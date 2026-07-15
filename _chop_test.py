"""Check if regime/gate detects choppy markets."""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys, os
from datetime import datetime
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate

mt5.initialize()
cfg = BTCConfig()
engine = BTCFeatureEngine()

encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir + '/btc_h1_encoder.pt', map_location='cpu', weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])
gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
rd = RuleBasedRegimeDetector()

h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, datetime(2026,4,25), datetime(2026,5,25,12))
h1 = pd.DataFrame(h1r).rename(columns={'time':'ts','tick_volume':'volume'})
h1['ts'] = pd.to_datetime(h1['ts'], unit='s', utc=True)
h1 = h1.sort_values('ts').reset_index(drop=True)
h1f = engine.compute(h1)

for i in range(cfg.seq_len_h1):
    rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

rc = {'TREND_UP': 0, 'TREND_DOWN': 0, 'RANGE': 0, 'TRANSITION': 0}
blocks = 0; range_blocks = 0; vol_blocks = 0; trans_blocks = 0

print('{:22s} {:>8s} {:12s} {:>6s} {:12s} {:8s} {:20s}'.format(
    'Time', 'Close', 'Model', 'Prob', 'Rule', 'Gate', 'Reason'))
print('-' * 95)

for i in range(cfg.seq_len_h1, len(h1)):
    ts = h1['ts'].iloc[i]; close = float(h1['close'].iloc[i])
    seq = engine.compute_sequence(h1f, i, cfg.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0)
    for j in range(max(0,i-13), i+1):
        rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
    rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
    rule_out = rd._classify()
    atr_pct = float(rr.get('atr_percentile', 0.5))
    bb_pos = float(h1f[i, 4])
    gd = gate.evaluate(rr['regime'], rr['confidence'], atr_pct, bb_position=bb_pos)
    rc[rr['regime']] = rc.get(rr['regime'], 0) + 1
    if not gd.entry_signal:
        blocks += 1
        if 'vol' in gd.reason: vol_blocks += 1
        if rr['regime'] == 'RANGE': range_blocks += 1
        if rr['regime'] == 'TRANSITION': trans_blocks += 1
    signal = 'S' if (gd.entry_signal and gd.direction==-1) else 'L' if gd.entry_signal else '.'
    reason = gd.reason[:20] if not gd.entry_signal else 'entry'
    if i % 3 == 0:
        print('{:22s} ${:>7.1f} {:12s} {:>5.3f} {:12s} {:8s} {:20s}'.format(
            str(ts)[:19], close, rr['regime'], rr['confidence'],
            rule_out.get('regime','?'), signal, reason))

total = sum(rc.values())
print()
print('Regime distribution (last month):')
for r in ['TREND_UP','TREND_DOWN','RANGE','TRANSITION']:
    c = rc.get(r, 0)
    print('  {}: {} bars ({:.1f}%)'.format(r, c, c/total*100))
print()
print('Gate blocked: {} / {} bars ({:.1f}%)'.format(blocks, total, blocks/total*100))
print('  RANGE blocks: {}'.format(range_blocks))
print('  TRANSITION blocks: {}'.format(trans_blocks))
print('  Volatility blocks: {}'.format(vol_blocks))

# Check: during RANGE/TRANSITION, does gate ever incorrectly give entry?
print()
print('RANGE/TRANSITION bars where gate incorrectly allows entry:')
bad = 0
for i in range(cfg.seq_len_h1, len(h1)):
    ts = h1['ts'].iloc[i]
    seq = engine.compute_sequence(h1f, i, cfg.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0)
    for j in range(max(0,i-13), i+1):
        rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
    rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
    gd = gate.evaluate(rr['regime'], rr['confidence'],
                       float(rr.get('atr_percentile',0.5)),
                       bb_position=float(h1f[i,4]))
    if rr['regime'] in ('RANGE', 'TRANSITION') and gd.entry_signal:
        bad += 1
        if bad <= 10:
            print('  {} regime={} gate=ENTER reason={}'.format(
                str(ts)[:19], rr['regime'], gd.reason))
if bad == 0:
    print('  None — gate correctly blocks all RANGE/TRANSITION bars')

# Check: what about TREND bars? Do they all produce signals?
print()
print('TREND bars with NO signal:')
no_sig = 0
for i in range(cfg.seq_len_h1, len(h1)):
    ts = h1['ts'].iloc[i]
    seq = engine.compute_sequence(h1f, i, cfg.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0)
    for j in range(max(0,i-13), i+1):
        rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
    rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
    gd = gate.evaluate(rr['regime'], rr['confidence'],
                       float(rr.get('atr_percentile',0.5)),
                       bb_position=float(h1f[i,4]))
    if rr['regime'] in ('TREND_UP', 'TREND_DOWN') and not gd.entry_signal:
        no_sig += 1
        if no_sig <= 10:
            print('  {} regime={} gate=BLOCK reason={}'.format(
                str(ts)[:19], rr['regime'], gd.reason))
print('Total TREND bars blocked: {}'.format(no_sig))

# VIX-style: ATR percentile as chop detector
print()
print('ATR percentile distribution:')
# Compute ATR percentiles across all bars
atr_pcts = []
for i in range(cfg.seq_len_h1, len(h1)):
    atr_pcts.append(float(h1f[i, 6]))
atr_pcts = np.array(atr_pcts)
for p in [10, 25, 50, 75, 90]:
    print('  P{}: {:.4f} (ATR% of price)'.format(p, np.percentile(atr_pcts, p)))
print('  Current: {:.4f}'.format(atr_pcts[-1]))

# BB width as chop detector
bb_widths = []
for i in range(cfg.seq_len_h1, len(h1)):
    bb_widths.append(float(h1f[i, 7]))
bb_widths = np.array(bb_widths)
print()
print('BB width distribution (narrow = chop):')
for p in [10, 25, 50, 75, 90]:
    print('  P{}: {:.4f}'.format(p, np.percentile(bb_widths, p)))
print('  Current: {:.4f}'.format(bb_widths[-1]))

mt5.shutdown()
