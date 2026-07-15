"""Test regime persistence gate on May 22 live trades."""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys
from datetime import datetime

sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate

mt5.initialize()
config = BTCConfig()
device = torch.device('cuda')

encoder = CNNLSTMEncoder(n_features=config.n_features, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
    bidirectional=True).to(device).eval()
classifier = RegimeClassifier(embedding_dim=config.embedding_dim,
    n_classes=config.regime_classes).to(device).eval()
ckpt = torch.load(config.model_dir + '/btc_h1_encoder.pt', map_location=device, weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])
engine = BTCFeatureEngine()
gate = EntryGate()

END = datetime(2026, 5, 23, 12, 0, 0)
START = datetime(2026, 5, 16, 0, 0, 0)  # Need 96+ H1 bars for encoder
h1_rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, START, END)
h1f = pd.DataFrame(h1_rates)
h1f = h1f.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
h1f['timestamp'] = pd.to_datetime(h1f['timestamp'], unit='s', utc=True)

print(f'H1 bars: {len(h1f)} ({h1f["timestamp"].min()} -> {h1f["timestamp"].max()})')

# Run regime detection on all bars
regimes = []
for i in range(config.seq_len_h1, len(h1f)):
    ts = h1f['timestamp'].iloc[i]
    h1_slice = h1f.iloc[:i+1]
    feats = engine.compute(h1_slice)
    seq = engine.compute_sequence(feats, len(feats)-1, config.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0).to(device)

    rd = RuleBasedRegimeDetector()
    for _, row in h1_slice.iloc[-20:].iterrows():
        rd.update(row['high'], row['low'], row['close'])

    rr = classify_regime(encoder, classifier, t, rd, model_confidence_threshold=config.min_regime_confidence)
    g = gate.evaluate(rr['regime'], rr['confidence'], rr.get('atr_percentile', 0.5), bb_position=feats[-1, 4])

    regimes.append({
        'ts': ts,
        'regime': rr['regime'],
        'confidence': rr['confidence'],
        'direction': g.direction if g.entry_signal else 0,
        'entry_signal': g.entry_signal
    })

reg_df = pd.DataFrame(regimes)
print(f'reg_df columns: {reg_df.columns.tolist()}')
print(f'reg_df shape: {reg_df.shape}')
print(f'First 2 rows: {reg_df.head(2).to_dict()}')

# Filter to May 22
may22_start = pd.Timestamp('2026-05-22 00:00:00', tz='UTC')
may23_start = pd.Timestamp('2026-05-23 00:00:00', tz='UTC')
may22 = reg_df[(reg_df['ts'] >= may22_start) & (reg_df['ts'] < may23_start)]

print(f'\nMay 22 H1 regime sequence:')
print(f'{"Time":20s} {"Regime":15s} {"Conf":6s} {"Signal"}')
print('-' * 55)
for _, r in may22.iterrows():
    sig = f'DIR={r["direction"]:+.0f}' if r['entry_signal'] else 'no signal'
    print(f'{str(r["ts"]):20s} {r["regime"]:15s} {r["confidence"]:.2f}   {sig}')

# Actual trades from MT5 history on May 22
# (time, direction, pnl)
actual_trades = [
    ('00:00', 'SHORT', +10),      # 00:00 entry, 01:02 exit SL
    ('04:00', 'SHORT', -179),     # 04:00 entry, 04:15 exit SL
    ('08:00', 'LONG', -208),      # 08:00 entry, 12:00 exit SL
    ('16:00', 'LONG', -141),      # 16:00 entry, 17:56 exit SL
    ('20:00', 'SHORT', -122),     # 20:00 entry, 23:12 exit SL
    ('21:00', 'SHORT', -113),     # 21:00 entry, 23:12 exit SL
]

# Also check May 23 trades (after midnight) for completeness
may23 = reg_df[(reg_df['ts'] >= may23_start) & (reg_df['ts'] < pd.Timestamp('2026-05-23 18:00:00', tz='UTC'))]
print(f'\nMay 23 H1 regime sequence (for comparison):')
print(f'{"Time":20s} {"Regime":15s} {"Conf":6s} {"Signal"}')
print('-' * 55)
for _, r in may23.iterrows():
    sig = f'DIR={r["direction"]:+.0f}' if r['entry_signal'] else 'no signal'
    print(f'{str(r["ts"]):20s} {r["regime"]:15s} {r["confidence"]:.2f}   {sig}')

# Persistence gate analysis
print('\n' + '='*60)
print('PERSISTENCE GATE — What if we required N consistent bars?')
print('='*60)

for N in [2, 3]:
    print(f'\n--- N={N} (regime must be same for {N} bars) ---')
    blocked = []
    allowed = []
    for i in range(N-1, len(may22)):
        last_n = may22['regime'].iloc[i-N+1:i+1].tolist()
        consistent = len(set(last_n)) == 1
        has_signal = may22['entry_signal'].iloc[i]
        if has_signal:
            ts_str = str(may22['ts'].iloc[i])[11:16]  # HH:MM from ISO
            if consistent:
                allowed.append((ts_str, may22['regime'].iloc[i], may22['direction'].iloc[i]))
            else:
                blocked.append((ts_str, may22['regime'].iloc[i], may22['direction'].iloc[i], last_n))

    print(f'  BLOCKED signals ({len(blocked)}):')
    for b in blocked:
        print(f'    {b[0]} | regime={b[1]:15s} DIR={b[2]:+.0f} | recent: {b[3]}')
    print(f'  ALLOWED signals ({len(allowed)}):')
    for a in allowed:
        print(f'    {a[0]} | regime={a[1]:15s} DIR={a[2]:+.0f}')

    # Match signals to actual trades by hour
    blocked_pnl = 0
    prevented = 0
    for trade_time, trade_dir, trade_pnl in actual_trades:
        trade_hour = int(trade_time.split(':')[0])
        for b_ts, b_regime, b_dir, *rest in blocked:
            b_hour = int(b_ts.split(':')[0])
            # Match if within 1 hour (signals fire on H1 close, trades enter on M15 confirm)
            if abs(b_hour - trade_hour) <= 1:
                blocked_pnl += trade_pnl
                prevented += 1
                print(f'    PREVENTED: {trade_time} {trade_dir} (PnL={trade_pnl:+})')
                break

    total_orig = sum(t[2] for t in actual_trades)
    new_pnl = total_orig - blocked_pnl
    print(f'\n  Original PnL: ${total_orig}')
    print(f'  PnL saved:    ${-blocked_pnl:+}')
    print(f'  New PnL:      ${new_pnl:+}')
    print(f'  Trades: {len(actual_trades)} -> {len(actual_trades) - prevented}')

# Also test: what about the WINNING session (May 23 early morning)?
print('\n' + '='*60)
print('EFFECT ON WINNING SESSION (May 23 00:00-12:00)')
print('='*60)
may23_wins = [
    ('01:00', 'SHORT', +193),
    ('05:00', 'SHORT', +266),
    ('09:00', 'SHORT', -50),
    ('13:45', 'SHORT', +182),
]
for N in [2, 3]:
    blocked_win = 0
    for i in range(N-1, len(may23)):
        last_n = may23['regime'].iloc[i-N+1:i+1].tolist()
        consistent = len(set(last_n)) == 1
        has_signal = may23['entry_signal'].iloc[i]
        if has_signal and not consistent:
            ts_str = str(may23['ts'].iloc[i])[11:16]
            b_hour = int(ts_str.split(':')[0])
            for trade_time, trade_dir, trade_pnl in may23_wins:
                trade_hour = int(trade_time.split(':')[0])
                if abs(b_hour - trade_hour) <= 1:
                    blocked_win += trade_pnl
                    print(f'  N={N}: Would BLOCK {trade_time} {trade_dir} (PnL={trade_pnl:+})')
    print(f'  N={N}: Winning trades blocked PnL = ${blocked_win:+}')
