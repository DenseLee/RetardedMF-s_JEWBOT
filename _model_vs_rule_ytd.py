"""Model vs Rule Detector — YTD accuracy and PnL impact."""
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

print('Fetching YTD data...')
mt5.initialize()
h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, datetime(2026,1,1), datetime(2026,5,25,12))
mt5.shutdown()
h1 = pd.DataFrame(h1r).rename(columns={'time':'ts','tick_volume':'volume'})
h1['ts'] = pd.to_datetime(h1['ts'], unit='s', utc=True)
h1 = h1.sort_values('ts').reset_index(drop=True)
h1f = engine.compute(h1)
print('H1 bars: {}'.format(len(h1)))

# Build oracle lookup from M15 oracle (best available)
# For H1-level oracle, use the M15 oracle at the H1 bar timestamp
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from benchmark.oracle_m15 import M15Oracle, M15OracleLabeler
import __main__; __main__.M15Oracle = M15Oracle
try:
    with open('D:/FiananceBot/BTC_BOT/benchmark/m15_oracle.pkl', 'rb') as f:
        m15_labels = pickle.load(f)
    oracle_by_ts = {}
    for l in m15_labels:
        oracle_by_ts[l.timestamp] = l
    print('Oracle: {} M15 labels'.format(len(m15_labels)))
except:
    oracle_by_ts = {}
    print('No oracle available — computing H1-level oracle instead')

# Walk every H1 bar, compare model vs rule
rd = RuleBasedRegimeDetector()
for i in range(cfg.seq_len_h1):
    rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

bars = []
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

    # Oracle truth at this bar: what direction actually works?
    h1_ts_str = str(h1['ts'].iloc[hi])[:19]
    oracle = oracle_by_ts.get(h1_ts_str)
    oracle_dir = 0  # 1=long wins, -1=short wins, 0=chop
    oracle_long_r = 0; oracle_short_r = 0
    if oracle:
        if oracle.long_r > 0.5 and oracle.short_r <= 0: oracle_dir = 1
        elif oracle.short_r > 0.5 and oracle.long_r <= 0: oracle_dir = -1
        oracle_long_r = oracle.long_r; oracle_short_r = oracle.short_r

    trending = {'TREND_UP', 'TREND_DOWN'}
    model_dir = 1 if rr['regime'] == 'TREND_UP' else (-1 if rr['regime'] == 'TREND_DOWN' else 0)
    rule_dir = 1 if rule_out['regime'] == 'TREND_UP' else (-1 if rule_out['regime'] == 'TREND_DOWN' else 0)

    agree = 1 if rr['regime'] == rule_out['regime'] else 0
    conflict = 1 if (rr['regime'] in trending and rule_out['regime'] in trending
                     and rr['regime'] != rule_out['regime']) else 0

    # Who would the gate follow?
    gate_dir = gd.direction if gd.entry_signal else 0

    # Is the model right about direction?
    model_correct = 1 if model_dir != 0 and model_dir == oracle_dir else 0
    rule_correct = 1 if rule_dir != 0 and rule_dir == oracle_dir else 0
    gate_correct = 1 if gate_dir != 0 and gate_dir == oracle_dir else 0

    # What R would you get following each?
    model_r = oracle_long_r if model_dir == 1 else (oracle_short_r if model_dir == -1 else 0)
    rule_r = oracle_long_r if rule_dir == 1 else (oracle_short_r if rule_dir == -1 else 0)
    gate_r = oracle_long_r if gate_dir == 1 else (oracle_short_r if gate_dir == -1 else 0)

    bars.append({
        'ts': h1_ts_str, 'close': float(h1['close'].iloc[hi]),
        'model_regime': rr['regime'], 'model_conf': rr['confidence'],
        'rule_regime': rule_out['regime'], 'rule_conf': rule_out['confidence'],
        'agree': agree, 'conflict': conflict,
        'gate_dir': gate_dir, 'model_dir': model_dir, 'rule_dir': rule_dir,
        'oracle_dir': oracle_dir,
        'model_correct': model_correct, 'rule_correct': rule_correct, 'gate_correct': gate_correct,
        'model_r': model_r, 'rule_r': rule_r, 'gate_r': gate_r,
        'source': rr.get('source', '?'),
    })

df = pd.DataFrame(bars)
n = len(df)

# Summary stats
agree_bars = df[df['agree'] == 1]
conflict_bars = df[df['conflict'] == 1]
trend_bars = df[(df['model_dir'] != 0) & (df['rule_dir'] != 0)]
conflict_in_trend = df[df['conflict'] == 1]

print()
print('=' * 60)
print('MODEL vs RULE DETECTOR — YTD Accuracy')
print('=' * 60)
print('Total H1 bars: {}'.format(n))
print()
print('Agreement:')
print('  Agree on regime:   {} bars ({:.1f}%)'.format(len(agree_bars), len(agree_bars)/n*100))
print('  TREND conflicts:   {} bars ({:.1f}%)'.format(len(conflict_bars), len(conflict_bars)/n*100))
print()

# Accuracy when they agree vs conflict
# Use oracle_dir as ground truth (only where oracle is available)
oracle_bars = df[df['oracle_dir'] != 0]
if len(oracle_bars) > 0:
    agree_oracle = oracle_bars[oracle_bars['agree'] == 1]
    conflict_oracle = oracle_bars[oracle_bars['conflict'] == 1]

    print('Oracle-verified bars (where oracle shows a clear direction): {}'.format(len(oracle_bars)))
    print()

    if len(agree_oracle) > 0:
        print('When model & rule AGREE:')
        print('  Model correct: {:.1f}%  Rule correct: {:.1f}%  Gate correct: {:.1f}%'.format(
            agree_oracle['model_correct'].mean()*100,
            agree_oracle['rule_correct'].mean()*100,
            agree_oracle['gate_correct'].mean()*100))
        print('  Avg model R: {:+.3f}  Avg rule R: {:+.3f}  Avg gate R: {:+.3f}'.format(
            agree_oracle['model_r'].mean(), agree_oracle['rule_r'].mean(), agree_oracle['gate_r'].mean()))

    if len(conflict_oracle) > 0:
        print()
        print('When model & rule CONFLICT:')
        print('  Model correct: {:.1f}%  Rule correct: {:.1f}%  Gate follows: MODEL ({:.1f}% correct)'.format(
            conflict_oracle['model_correct'].mean()*100,
            conflict_oracle['rule_correct'].mean()*100,
            conflict_oracle['gate_correct'].mean()*100))
        print('  Avg model R: {:+.3f}  Avg rule R: {:+.3f}  Avg gate R: {:+.3f}'.format(
            conflict_oracle['model_r'].mean(), conflict_oracle['rule_r'].mean(), conflict_oracle['gate_r'].mean()))

# Cumulative R if following model vs rule
print()
print('=' * 60)
print('CUMULATIVE R — Following Model vs Rule Detector')
print('=' * 60)

# Only count bars where oracle has a direction AND either model/rule says trending
trend_oracle = oracle_bars[(oracle_bars['model_dir'] != 0) | (oracle_bars['rule_dir'] != 0)]

if len(trend_oracle) > 0:
    model_total_r = trend_oracle['model_r'].sum()
    rule_total_r = trend_oracle['rule_r'].sum()
    gate_total_r = trend_oracle['gate_r'].sum()

    # Perfect oracle: always pick right direction
    perfect_r = trend_oracle.apply(lambda r: max(r['oracle_long_r'] if 'oracle_long_r' in r.index else 0,
                                                   r['oracle_short_r'] if 'oracle_short_r' in r.index else 0), axis=1).sum()

    print('Bars where direction matters: {}'.format(len(trend_oracle)))
    print()
    print('  Model detector:      {:+.1f}R'.format(model_total_r))
    print('  Rule detector:       {:+.1f}R'.format(rule_total_r))
    print('  Gate (current bot):  {:+.1f}R'.format(gate_total_r))
    print('  Perfect oracle:      {:+.1f}R'.format(perfect_r))
    print()

    # During conflicts specifically
    conflict_trend = trend_oracle[trend_oracle['conflict'] == 1]
    if len(conflict_trend) > 0:
        cm_r = conflict_trend['model_r'].sum()
        cr_r = conflict_trend['rule_r'].sum()
        cg_r = conflict_trend['gate_r'].sum()
        print('During TREND conflicts only ({} bars):'.format(len(conflict_trend)))
        print('  Model (always chosen): {:+.1f}R'.format(cm_r))
        print('  Rule (always ignored): {:+.1f}R'.format(cr_r))
        print('  Gate (follows model):  {:+.1f}R'.format(cg_r))
        if cr_r > cm_r:
            print('  *** Rule detector WOULD have made {:+.1f}R more! ***'.format(cr_r - cm_r))

# Monthly conflict rate
print()
print('=' * 60)
print('MONTHLY CONFLICT RATE')
print('=' * 60)
df['month'] = pd.to_datetime(df['ts']).dt.strftime('%Y-%m')
for month, grp in df.groupby('month'):
    conflicts_m = grp['conflict'].sum()
    entries_m = (grp['gate_dir'] != 0).sum()
    conflict_entry = ((grp['conflict'] == 1) & (grp['gate_dir'] != 0)).sum()
    print('  {}: {} conflicts, {} entries, {} conflict-entries'.format(
        month, conflicts_m, entries_m, conflict_entry))
