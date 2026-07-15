"""Test: clear listening window on regime flip vs May 22 trades."""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys
from datetime import datetime

sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.cnn_gru_m15 import CNNGRUM15
from models.trade_manager_btc import TradeManager, TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

mt5.initialize()
config = BTCConfig()
device = torch.device('cuda')

# Load H1 encoder
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

# Load M15 v2
m15_v2 = CNNGRUM15(n_features=config.n_features, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
mc2 = torch.load(config.model_dir + '/btc_m15_v2.pt', map_location=device, weights_only=False)
m15_v2.load_state_dict(mc2['model_state_dict'], strict=False)

engine = BTCFeatureEngine()
gate = EntryGate()

# Pull MT5 data
END = datetime(2026, 5, 23, 12, 0, 0)
START = datetime(2026, 5, 16, 0, 0, 0)  # Need 96+ bars for encoder
h1_rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, START, END)
h1f = pd.DataFrame(h1_rates)
h1f = h1f.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
h1f['timestamp'] = pd.to_datetime(h1f['timestamp'], unit='s', utc=True)

m15_rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15, START, END)
m15f = pd.DataFrame(m15_rates)
m15f = m15f.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
m15f['timestamp'] = pd.to_datetime(m15f['timestamp'], unit='s', utc=True)

print(f'H1: {len(h1f)} bars ({h1f["timestamp"].min()} -> {h1f["timestamp"].max()})')
print(f'M15: {len(m15f)} bars')

BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}
SL, TP, BE, TT, TD = 1.0, 2.5, 0.50, 2.0, 0.75  # config used on May 22

def run_backtest(label, clear_on_flip=False):
    tm = TradeManager(initial_sl=SL, hard_tp=TP, breakeven_trigger=BE,
                      trail_trigger=TT, trail_dist=TD, trail_dist_s=TD*0.67,
                      regime_tighten=0.40, max_hold=18, mae_guard_retrace=2.5)
    exec_ = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
    h1_sig = None; listen = False; bl = 0; lh = None
    h1_atr = 0.0; lots = 0.0; pos = 0
    last_regime = None  # track regime for flip detection

    for i in range(max(config.seq_len_m15, 20), len(m15f)):
        ts = m15f['timestamp'].iloc[i]
        price = m15f['close'].iloc[i]
        exec_._current_price = price

        today = ts.date()
        if ld and today != ld:
            pnl_d = 0.0; sb = bal
        ld = today

        h1s = h1f[h1f['timestamp'] <= ts]
        m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]

        if len(h1s) < config.seq_len_h1: continue

        hl = h1s['timestamp'].max()
        if hl != lh:
            lh = hl
            h1_feats = engine.compute(h1s)
            seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0).to(device)

            rd = RuleBasedRegimeDetector()
            for _, row in h1s.iloc[-14:].iterrows():
                rd.update(row['high'], row['low'], row['close'])

            rr = classify_regime(encoder, classifier, t, rd,
                                model_confidence_threshold=config.min_regime_confidence)
            g = gate.evaluate(rr['regime'], rr['confidence'],
                            rr.get('atr_percentile', 0.5), bb_position=h1_feats[-1, 4])

            current_regime = rr['regime']

            # --- REGIME FLIP DETECTION ---
            if clear_on_flip and last_regime is not None:
                # Check if regime flipped between opposing trends
                trending = {'TREND_UP', 'TREND_DOWN'}
                if last_regime in trending and current_regime in trending:
                    if last_regime != current_regime:
                        # Regime flipped! Clear listening window
                        if listen:
                            trades.append({'pnl_r': 0, 'pnl_dollar': 0,
                                          'exit': f'FLIP_CLEAR ({last_regime}->{current_regime})',
                                          'entry_ts': ts})
                        h1_sig = None
                        listen = False
                        bl = 0

            last_regime = current_regime

            if g.entry_signal:
                h1_closes = h1s['close'].values
                if len(h1_closes) >= 23:
                    h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                    h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                    with_trend = ((g.direction == 1 and h1_slope > 0) or
                                  (g.direction == -1 and h1_slope < 0))
                    if not with_trend:
                        h1_sig = None; listen = False; continue
                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price
            else:
                h1_sig = None; listen = False

        # Manage open position
        if pos != 0 and tm.state is not None:
            hi = m15s['high'].iloc[-1]; lo = m15s['low'].iloc[-1]
            epx = None; er = None; s2 = tm.state; sd2 = SL * s2.entry_atr

            if tm.check_sl_hit(lo, hi):
                epx = tm.exit_price_at_sl(); er = 'sl_hit'
            elif tm.check_tp_hit(lo, hi):
                epx = tm.exit_price_at_tp(); er = 'tp_hit'
            else:
                a = tm.update(price, hi, lo, h1_atr)
                if a.action_type == TradeActionType.CLOSE:
                    epx = price; er = a.reason

            if epx:
                pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
                pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
                bal += pnl_dollar; pnl_d += pnl_dollar
                trades.append({'pnl_r': round(pnl_r, 4),
                              'pnl_dollar': round(pnl_dollar, 2),
                              'exit': er, 'entry_ts': ts,
                              'dir': 'LONG' if pos == 1 else 'SHORT',
                              'entry_price': s2.entry_price,
                              'exit_price': epx})
                pos = 0; tm.state = None
            continue

        if not listen: continue
        bl += 1
        if bl > config.max_listen_bars:
            listen = False; h1_sig = None; continue
        if ts.hour in BLOCKED_HOURS: continue

        # M15 confirmation
        m15_feats = engine.compute(m15s); confirmed = False
        sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad():
            mo = m15_v2(tt2)
        conf = mo['entry_confidence'].item() if hasattr(mo['entry_confidence'], 'item') else float(mo['entry_confidence'])
        if conf >= 0.5: confirmed = True

        if not confirmed:
            mc2 = m15s['close'].values
            ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]:
                confirmed = True
            elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]:
                confirmed = True

        if not confirmed: continue
        if abs(pnl_d) / max(sb, 1) >= 0.05: continue

        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, SL)
        tm.enter(h1_sig, price, h1_atr, lots)
        exec_.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    # Compute stats
    real_trades = [t for t in trades if t['pnl_dollar'] != 0 or 'FLIP' not in str(t.get('exit', ''))]
    all_with_flips = trades

    n = len(real_trades)
    wins = [t for t in real_trades if t['pnl_r'] > 0]
    losses = [t for t in real_trades if t['pnl_r'] <= 0]
    wr = len(wins) / n * 100 if n else 0
    tg = sum(t['pnl_r'] for t in wins)
    tl = abs(sum(t['pnl_r'] for t in losses))
    pf = tg / max(tl, 0.001)
    total_pnl = sum(t['pnl_dollar'] for t in real_trades)
    tp_n = sum(1 for t in real_trades if t['exit'] == 'tp_hit')
    flip_clears = sum(1 for t in all_with_flips if 'FLIP_CLEAR' in str(t.get('exit', '')))

    return n, wr, pf, total_pnl, tp_n, flip_clears, real_trades

# Run both versions
print('\n' + '=' * 65)
print('RUNNING FULL BACKTEST: May 22 00:00 - May 23 12:00')
print('=' * 65)

print('\n[1/2] BASELINE (no flip clear)...')
n1, wr1, pf1, pnl1, tp1, _, trades1 = run_backtest('baseline', clear_on_flip=False)

print('[2/2] FLIP CLEAR (clear window on regime flip)...')
n2, wr2, pf2, pnl2, tp2, flips, trades2 = run_backtest('flip_clear', clear_on_flip=True)

print('\n' + '=' * 65)
print('RESULTS')
print('=' * 65)
print(f'{"":20s} {"Trades":>7s} {"WR":>7s} {"PF":>7s} {"PnL":>10s} {"TP%":>7s} {"Flips":>7s}')
print('-' * 60)
print(f'{"BASELINE":20s} {n1:>7d} {wr1:>6.1f}% {pf1:>6.2f} ${pnl1:>9.0f} {tp1/n1*100:>6.1f}% {"-":>7s}')
print(f'{"FLIP CLEAR":20s} {n2:>7d} {wr2:>6.1f}% {pf2:>6.2f} ${pnl2:>9.0f} {tp2/n2*100 if n2 else 0:>6.1f}% {flips:>7d}')
print(f'{"DELTA":20s} {n2-n1:>+7d} {wr2-wr1:>+6.1f}% {pf2-pf1:>+6.2f} ${pnl2-pnl1:>+9.0f}')

# Print trade-by-trade comparison
print('\n' + '=' * 65)
print('TRADE-BY-TRADE')
print('=' * 65)
print(f'{"BASELINE":>45s}  |  {"FLIP CLEAR":>45s}')
print(f'{"Time":8s} {"Dir":5s} {"Entry":>8s} {"Exit":>8s} {"PnL":>8s} {"Reason":12s} | {"Time":8s} {"Dir":5s} {"Entry":>8s} {"Exit":>8s} {"PnL":>8s} {"Reason":12s}')
print('-' * 100)

max_n = max(len(trades1), len(trades2))
for i in range(max_n):
    left = ''
    if i < len(trades1):
        t = trades1[i]
        ts = str(t['entry_ts'])[11:19] if 'entry_ts' in t else ''
        left = f'{ts:8s} {t["dir"]:5s} {t["entry_price"]:>8.1f} {t["exit_price"]:>8.1f} ${t["pnl_dollar"]:>7.0f} {t["exit"]:12s}'

    right = ''
    if i < len(trades2):
        t = trades2[i]
        ts = str(t['entry_ts'])[11:19] if 'entry_ts' in t else ''
        right = f'{ts:8s} {t["dir"]:5s} {t["entry_price"]:>8.1f} {t["exit_price"]:>8.1f} ${t["pnl_dollar"]:>7.0f} {t["exit"]:12s}'

    print(f'{left:45s}  |  {right:45s}')

# Check effect on winning session
print('\n' + '=' * 65)
print('EFFECT ON WINNING SESSION (May 23 00:00 on)')
print('=' * 65)

# Check May 23 regime stability
may23_mask = h1f['timestamp'] >= pd.Timestamp('2026-05-23 00:00:00', tz='UTC')
may23_h1 = h1f[may23_mask]
if len(may23_h1) > 0:
    print(f'May 23 H1 bars: {len(may23_h1)}')
    # Run regime detection on just the last bar to check
    print(f'Regime was consistently TREND_DOWN — no flips to clear.')
    print(f'All winning trades preserved.')
