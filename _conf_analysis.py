"""Analyze M15 confidence vs trade outcomes for May 2026."""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys, os
from datetime import datetime
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.cnn_gru_m15 import CNNGRUM15
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType

mt5.initialize()
cfg = BTCConfig()

# Load H1
encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir + '/btc_h1_encoder.pt', map_location='cpu', weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])

# Load FIXED M15
m15_model = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
mc2 = torch.load(cfg.model_dir + '/btc_m15_v2.pt', map_location='cpu', weights_only=False)
sd = mc2['model_state_dict']
block_starts = {'conv1': 0, 'conv2': 4, 'conv3': 8}
remapped = {}
for old_key, val in sd.items():
    prefix = old_key.split('.')[0]
    if prefix in block_starts:
        rest = old_key.split('.', 1)[1]; sub_idx = int(rest.split('.')[0])
        param = rest.split('.', 1)[1]; flat_idx = block_starts[prefix] + sub_idx
        new_key = f'cnn.{flat_idx}.{param}'
    elif old_key.startswith('entry_head.'):
        new_key = old_key.replace('entry_head.', 'entry_conf.', 1)
    else:
        new_key = old_key
    remapped[new_key] = val
m15_model.load_state_dict(remapped, strict=False)

engine = BTCFeatureEngine()
gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
rd = RuleBasedRegimeDetector()
BLOCKED = {2, 11, 18, 19, 21, 22, 23}

# Data
h1_rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1,
    datetime(2026, 5, 1), datetime(2026, 5, 25, 12))
m15_rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15,
    datetime(2026, 5, 1), datetime(2026, 5, 25, 12))
h1 = pd.DataFrame(h1_rates).rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
m15 = pd.DataFrame(m15_rates).rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
h1['timestamp'] = pd.to_datetime(h1['timestamp'], unit='s', utc=True)
m15['timestamp'] = pd.to_datetime(m15['timestamp'], unit='s', utc=True)
h1 = h1.sort_values('timestamp').reset_index(drop=True)
m15 = m15.sort_values('timestamp').reset_index(drop=True)

print(f'H1: {len(h1)}  M15: {len(m15)}')

h1_feats = engine.compute(h1)
m15_feats = engine.compute(m15)

# Pre-warm rule detector
for i in range(cfg.seq_len_h1):
    rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

# Pre-compute M15 confs
print('Computing M15 confidence...')
m15_confs = np.zeros(len(m15), dtype=np.float32)
for i in range(cfg.seq_len_m15, len(m15)):
    seq = engine.compute_sequence(m15_feats, i, cfg.seq_len_m15)
    with torch.no_grad():
        o = m15_model(torch.from_numpy(seq).unsqueeze(0))
    m15_confs[i] = float(o['entry_confidence'].squeeze().numpy())

# Run simulation
listening = False; h1_sig = 0; bars_listened = 0
position = 0; tm = None; entry_info = {}; trades = []
last_h1_processed = -1
pre_entry_confs = []  # track confs during listening window

for m15_i in range(cfg.seq_len_h1 * 4, len(m15)):
    ts = m15['timestamp'].iloc[m15_i]
    price = float(m15['close'].iloc[m15_i])
    m15_conf = float(m15_confs[m15_i])

    # H1 bar detection
    h1_i = int((h1['timestamp'] <= ts).sum() - 1)
    if h1_i >= cfg.seq_len_h1 and h1_i != last_h1_processed:
        last_h1_processed = h1_i
        seq = engine.compute_sequence(h1_feats, h1_i, cfg.seq_len_h1)
        t = torch.from_numpy(seq).unsqueeze(0)
        for j in range(max(0, h1_i - 13), h1_i + 1):
            rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
        rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
        gd = gate.evaluate(rr['regime'], rr['confidence'],
                           float(rr.get('atr_percentile', 0.5)),
                           bb_position=float(h1_feats[h1_i, 4]))

        if gd.entry_signal:
            h1_cl = h1['close'].values[:h1_i + 1]
            if len(h1_cl) >= 23:
                ema22 = pd.Series(h1_cl).ewm(span=22, adjust=False).mean().values
                slope = (ema22[-1] - ema22[-2]) / max(abs(float(ema22[-2])), 1e-12)
                with_trend = ((gd.direction == 1 and slope > 0) or
                              (gd.direction == -1 and slope < 0))
                if with_trend:
                    h1_sig = gd.direction; listening = True; bars_listened = 0
                    pre_entry_confs = []
                else:
                    listening = False
            else:
                listening = False
        else:
            listening = False

    # Manage open position
    if position != 0 and tm is not None and tm.state is not None:
        s = tm.state
        entry_info.setdefault('hold_confs', []).append(round(float(m15_conf), 4))

        hi = float(m15['high'].iloc[m15_i])
        lo = float(m15['low'].iloc[m15_i])
        exit_px = None; reason = ''

        if tm.check_sl_hit(lo, hi):
            exit_px = tm.exit_price_at_sl(); reason = 'sl_hit'
        elif tm.check_tp_hit(lo, hi):
            exit_px = tm.exit_price_at_tp(); reason = 'tp_hit'
        else:
            action = tm.update(price, hi, lo, h1_feats[h1_i, 6] * price)
            if action.action_type == TradeActionType.CLOSE:
                exit_px = price; reason = action.reason

        if exit_px:
            d = 1 if position == 1 else -1
            if d == 1:
                pnl_d = (exit_px - s.entry_price) * s.lots
            else:
                pnl_d = (s.entry_price - exit_px) * s.lots
            pnl_r = pnl_d / max(s.entry_atr * s.lots * cfg.initial_sl, 1e-9)

            entry_info['exit_conf'] = round(float(m15_conf), 4)
            entry_info['pnl_d'] = round(pnl_d, 2)
            entry_info['pnl_r'] = round(float(pnl_r), 4)
            entry_info['exit_reason'] = reason
            entry_info['exit_ts'] = str(ts)[:19]
            entry_info['exit_price'] = round(exit_px, 1)
            entry_info['bars_held'] = s.bars_held
            trades.append(entry_info)
            position = 0; tm = None
            entry_info = {}
            continue

    if not listening:
        continue

    bars_listened += 1
    if bars_listened > cfg.max_listen_bars:
        listening = False; pre_entry_confs = []; continue
    if ts.hour in BLOCKED:
        continue

    pre_entry_confs.append(round(float(m15_conf), 4))

    if m15_conf >= 0.5:
        h1_atr = float(h1_feats[h1_i, 6] * price)
        tm = TradeManager(
            initial_sl=cfg.initial_sl, hard_tp=cfg.hard_tp,
            breakeven_trigger=cfg.breakeven_trigger,
            trail_trigger=cfg.trail_trigger,
            trail_dist=cfg.trail_dist, trail_dist_s=cfg.trail_dist_s,
            regime_tighten=cfg.regime_tighten,
            max_hold=cfg.max_hold_bars, mae_guard_retrace=cfg.mae_guard_retrace)
        lots = TradeManager.compute_position_size(10000.0, h1_atr, price, cfg.risk_pct, cfg.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots, regime=rr['regime'])
        position = h1_sig; listening = False

        entry_info = {
            'entry_ts': str(ts)[:19],
            'entry_price': round(price, 1),
            'direction': 'LONG' if h1_sig == 1 else 'SHORT',
            'regime': rr['regime'],
            'm15_conf_at_entry': round(float(m15_conf), 4),
            'pre_entry_confs': pre_entry_confs.copy(),
            'hold_confs': [round(float(m15_conf), 4)],
            'exit_conf': 0, 'pnl_d': 0, 'pnl_r': 0,
            'exit_reason': '', 'exit_ts': '', 'exit_price': 0, 'bars_held': 0,
        }
        pre_entry_confs = []

# Print results
print(f'\n{"#":>3s} {"Entry":19s} {"Dir":5s} {"Entry$":>9s} {"PreAvg":>7s} {"@Entry":>7s} {"HoldAvg":>7s} {"@Exit":>7s} {"PnL":>9s} {"R":>6s} {"Exit":12s}')
print('-' * 115)

wins = losses = 0
for i, t in enumerate(trades):
    pre_avg = np.mean(t['pre_entry_confs']) if t['pre_entry_confs'] else 0
    hold_avg = np.mean(t['hold_confs']) if t['hold_confs'] else 0
    if t['pnl_d'] > 0: wins += 1
    elif t['pnl_d'] < 0: losses += 1
    marker = ' LOSS' if t['pnl_d'] < 0 else ''

    ep = t['entry_price']; ce = t['m15_conf_at_entry']; pd_ = t['pnl_d']
    pr = t['pnl_r']; ex = t['exit_reason']; ts_str = t['entry_ts']; dr = t['direction']
    print(f'{i+1:>3d} {ts_str:19s} {dr:5s} ${ep:>8.1f} {pre_avg:>6.4f} {ce:>7.4f} '
          f'{hold_avg:>7.4f} {t["exit_conf"]:>7.4f} ${pd_:>+8.1f} {pr:>+5.3f} {ex:12s}{marker}')

# Summary
n = len(trades)
if n > 0:
    pnls = np.array([t['pnl_d'] for t in trades])
    rs = np.array([t['pnl_r'] for t in trades])
    print(f'\n{"="*60}')
    print(f'Trades: {n}  Wins: {wins}  Losses: {losses}  WR: {wins/n*100:.1f}%')
    print(f'Total PnL: \${pnls.sum():+.1f}  Total R: {rs.sum():+.3f}')

    # By entry confidence
    high = [t for t in trades if t['m15_conf_at_entry'] >= 0.5]
    low = [t for t in trades if t['m15_conf_at_entry'] < 0.5]
    hi_pnl = sum(t['pnl_d'] for t in high)
    lo_pnl = sum(t['pnl_d'] for t in low)
    print()
    print('High conf (>=0.5): {} trades  PnL: ${:+.1f}'.format(len(high), hi_pnl))
    if high:
        hw = sum(1 for t in high if t['pnl_d'] > 0)
        hi_confs = [t['m15_conf_at_entry'] for t in high]
        print('  WR: {:.1f}%  Avg conf: {:.4f}'.format(hw/len(high)*100, np.mean(hi_confs)))

    print('Low conf (<0.5):  {} trades  PnL: ${:+.1f}'.format(len(low), lo_pnl))
    if low:
        lw = sum(1 for t in low if t['pnl_d'] > 0)
        lo_confs = [t['m15_conf_at_entry'] for t in low]
        print('  WR: {:.1f}%  Avg conf: {:.4f}'.format(lw/len(low)*100, np.mean(lo_confs)))

    # Correlation
    confs_arr = np.array([t['m15_conf_at_entry'] for t in trades])
    corr = np.corrcoef(confs_arr, pnls)[0, 1]
    print()
    print('Confidence-PnL correlation: r={:.3f}'.format(corr))
    print('(positive = higher conf -> better PnL, negative = opposite)')

    # Show the 5 best and 5 worst trades by confidence
    sorted_trades = sorted(trades, key=lambda t: t['m15_conf_at_entry'], reverse=True)
    print()
    print('Top 5 highest-confidence entries:')
    for t in sorted_trades[:5]:
        ts2 = t['entry_ts']; cf2 = t['m15_conf_at_entry']; pn2 = t['pnl_d']; ex2 = t['exit_reason']
        print('  {} conf={:.4f} PnL=${:+.1f} {}'.format(ts2, cf2, pn2, ex2))
    print('Bottom 5 lowest-confidence entries:')
    for t in sorted_trades[-5:]:
        ts2 = t['entry_ts']; cf2 = t['m15_conf_at_entry']; pn2 = t['pnl_d']; ex2 = t['exit_reason']
        print('  {} conf={:.4f} PnL=${:+.1f} {}'.format(ts2, cf2, pn2, ex2))

mt5.shutdown()
