"""Trace wrong-direction entries: model or rule detector?"""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys, os, pickle
from datetime import datetime, timedelta
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate

cfg = BTCConfig()
engine = BTCFeatureEngine()

encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir+'/btc_h1_encoder.pt', map_location='cpu', weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)

mt5.initialize()
h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, datetime(2026,4,25), datetime(2026,5,25,12))
mt5.shutdown()
h1 = pd.DataFrame(h1r).rename(columns={'time':'ts','tick_volume':'volume'})
h1['ts'] = pd.to_datetime(h1['ts'], unit='s', utc=True)
h1 = h1.sort_values('ts').reset_index(drop=True)
h1f = engine.compute(h1)

rd = RuleBasedRegimeDetector()
for i in range(cfg.seq_len_h1):
    rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

signals = []
for hi in range(cfg.seq_len_h1, len(h1)):
    seq = engine.compute_sequence(h1f, hi, cfg.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0)
    for j in range(max(0, hi-13), hi+1):
        rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
    rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
    rule_out = rd._classify()
    gd = gate.evaluate(rr['regime'], rr['confidence'],
                       float(rr.get('atr_percentile', 0.5)),
                       bb_position=float(h1f[hi, 4]))

    ema22_blocked = False
    if gd.entry_signal:
        hc = h1['close'].values[:hi+1]
        if len(hc) >= 23:
            e22 = pd.Series(hc).ewm(span=22, adjust=False).mean().values
            slp = (e22[-1]-e22[-2])/max(abs(float(e22[-2])),1e-12)
            if not ((gd.direction==1 and slp>0) or (gd.direction==-1 and slp<0)):
                ema22_blocked = True

    trending = {'TREND_UP', 'TREND_DOWN'}
    model_vs_rule = 'AGREE'
    if rr['regime'] in trending and rule_out['regime'] in trending:
        if rr['regime'] != rule_out['regime']:
            model_vs_rule = 'CONFLICT'

    signals.append({
        'ts': str(h1['ts'].iloc[hi])[:19],
        'close': float(h1['close'].iloc[hi]),
        'model_regime': rr['regime'],
        'model_conf': rr['confidence'],
        'source': rr.get('source', '?'),
        'rule_regime': rule_out['regime'],
        'rule_conf': rule_out['confidence'],
        'gate_dir': gd.direction if gd.entry_signal else 0,
        'gate_signal': gd.entry_signal,
        'gate_reason': gd.reason if not gd.entry_signal else 'entry',
        'ema22_blocked': ema22_blocked,
        'model_vs_rule': model_vs_rule,
    })

# Analysis
entries = [s for s in signals if s['gate_signal']]
model_entries = [s for s in entries if s['source'] == 'model']
rule_entries = [s for s in entries if s['source'] == 'rule']
conflicts = [s for s in entries if s['model_vs_rule'] == 'CONFLICT']

print('H1 bars: {}  Entry signals: {}'.format(len(signals), len(entries)))
print()
print('Decision source:')
print('  Model (conf >= 0.6): {} entries ({:.0f}%)'.format(len(model_entries), len(model_entries)/len(entries)*100 if entries else 0))
print('  Rule  (conf < 0.6):  {} entries ({:.0f}%)'.format(len(rule_entries), len(rule_entries)/len(entries)*100 if entries else 0))
print()
print('Model-Rule TREND conflicts: {} bars'.format(len(conflicts)))
for s in conflicts:
    print('  {}  model={} ({:.3f}) rule={} ({:.3f})  gate={}  ema22_blocked={}'.format(
        s['ts'], s['model_regime'], s['model_conf'], s['rule_regime'], s['rule_conf'],
        'L' if s['gate_dir']==1 else 'S' if s['gate_dir']==-1 else '-',
        s['ema22_blocked']))

# Who is responsible for wrong-direction entries?
print()
print('=== WRONG-DIRECTION TRACEBACK ===')
wrong_entries = [
    ('2026-05-22 07:00', 'LONG', 'SHORT_WIN'),
    ('2026-05-23 16:00', 'SHORT', 'LONG_WIN'),
    ('2026-05-23 20:30', 'SHORT', 'LONG_WIN'),
    ('2026-05-24 14:00', 'LONG', 'SHORT_WIN'),
    ('2026-05-24 16:30', 'LONG', 'SHORT_WIN'),
    ('2026-05-24 20:30', 'LONG', 'SHORT_WIN'),
]
for wt, wdir, oracle_wants in wrong_entries:
    wt_ts = pd.Timestamp(wt, tz='UTC')
    best = None; best_dt = timedelta(hours=3)
    for s in signals:
        dt = abs(pd.Timestamp(s['ts'], tz='UTC') - wt_ts)
        if dt < best_dt: best_dt = dt; best = s
    if best:
        who = 'MODEL' if best['source'] == 'model' else 'RULE'
        print('{}  bot={} oracle={}  -> H1 signal: model={} ({:.3f}) rule={} ({:.3f}) decided by: {}'.format(
            wt, wdir, oracle_wants, best['model_regime'], best['model_conf'],
            best['rule_regime'], best['rule_conf'], who))
        if best['model_vs_rule'] == 'CONFLICT':
            print('    CONFLICT: model and rule disagreed — {} won because confidence {} 0.6'.format(
                'model' if best['source']=='model' else 'rule',
                '>=' if best['source']=='model' else '<'))

# EMA22 blame
ema22_blocks = [s for s in signals if s['ema22_blocked']]
print()
print('EMA22 trend filter blocks: {} bars'.format(len(ema22_blocks)))

# Source distribution over time
print()
print('=== MODEL vs RULE — by week ===')
df = pd.DataFrame(signals)
df['week'] = pd.to_datetime(df['ts']).dt.strftime('%Y-W%W')
for week, grp in df.groupby('week'):
    ents = grp[grp['gate_signal']]
    m_e = (ents['source'] == 'model').sum()
    r_e = (ents['source'] == 'rule').sum()
    conflicts_w = (ents['model_vs_rule'] == 'CONFLICT').sum()
    print('  {}: {} entries ({} model, {} rule, {} conflicts)'.format(week, len(ents), m_e, r_e, conflicts_w))
