"""
Test: Is H1 model "confidence" actually trend exhaustion?

Hypothesis: High confidence = overfit to existing trend = about to reverse.
           Falling confidence = regime weakening = reversal signal.
           Low confidence = transition = trust rule detector direction.
"""
import sys, os, pickle
import numpy as np, pandas as pd, torch
from datetime import datetime, timedelta

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import (RegimeClassifier, RuleBasedRegimeDetector,
                                       classify_regime, REGIME_NAMES)
from models.entry_gate import EntryGate

cfg = BTCConfig()
device = torch.device("cpu")
fe = BTCFeatureEngine()

# Load H1 model
encoder_path = os.path.join(cfg.model_dir, "btc_h1_encoder.pt")
ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
encoder = CNNLSTMEncoder(
    n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).to(device).eval()
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim,
                               n_classes=cfg.regime_classes).to(device).eval()
classifier.load_state_dict(ckpt["classifier_state_dict"])

# Load oracle
with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle_labels = pickle.load(f)
# Filter to May
oracle_labels = [ol for ol in oracle_labels if ol.timestamp.startswith("2026-05")]
oracle_by_ts = {}
for ol in oracle_labels:
    oracle_by_ts[ol.timestamp] = ol

# Load backtest data
from backtest.data_manager import BacktestDataManager
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-05-01", "2026-05-26", use_cache=True)
h1 = ds.h1_df; m15 = ds.m15_df
h1_feats = ds.h1_features
h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile
h1_rule_regime = ds.h1_rule_regime

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile,
                 max_atr_pct=cfg.max_atr_percentile)
rule_det = RuleBasedRegimeDetector()

# Warm rule detector
for _, row in h1.iloc[:cfg.seq_len_h1].iterrows():
    rule_det.update(row["high"], row["low"], row["close"])

# ── Evaluate every H1 bar ──
print(f"{'='*100}")
print("TESTING: Model 'confidence' as trend exhaustion indicator")
print(f"{'='*100}")

records = []
for h1_i in range(cfg.seq_len_h1, len(h1)):
    ts = h1['timestamp'].iloc[h1_i]
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts_naive = ts.tz_localize(None)
    else:
        ts_naive = ts
    price = float(h1['close'].iloc[h1_i])

    # Update rule detector
    for _, row in h1.iloc[max(0, h1_i-14):h1_i+1].iterrows():
        rule_det.update(row["high"], row["low"], row["close"])

    # Model inference
    seq = fe.compute_sequence(h1_feats, h1_i, cfg.seq_len_h1)
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = encoder(tensor)
        raw_logits = classifier.raw_logits(enc_out["embedding"])
        probs_t4 = torch.softmax(raw_logits / cfg.regime_temperature, dim=1)
        max_prob, pred_class = probs_t4.max(dim=1)
        model_conf = max_prob.item()
        model_regime = REGIME_NAMES[pred_class.item()]

    # Rule detector
    rule_out = rule_det._classify()
    rule_regime = rule_out["regime"]
    rule_conf = rule_out["confidence"]
    rule_slope = rule_out.get("ema_slope", 0)

    # Full classify_regime (with fallback)
    regime_result = classify_regime(encoder, classifier, tensor, rule_det,
                                     model_confidence_threshold=cfg.min_regime_confidence)
    # Rule-wins-conflict
    trending = {"TREND_UP", "TREND_DOWN"}
    if (regime_result["regime"] in trending and
        rule_out["regime"] in trending and
        regime_result["regime"] != rule_out["regime"]):
        regime_result["regime"] = rule_out["regime"]
        regime_result["confidence"] = rule_out["confidence"]

    final_regime = regime_result["regime"]
    final_conf = regime_result["confidence"]

    # Oracle: what happened in the NEXT 12 H1 bars?
    # For each H1 bar, find oracle for the closest M15 bar
    ts_str = str(ts_naive)[:19]
    ol = oracle_by_ts.get(ts_str)
    if ol is None:
        for offset_h in [-1, 1]:
            adj = ts_naive + timedelta(hours=offset_h)
            ol = oracle_by_ts.get(str(adj)[:19])
            if ol: break

    if ol is None:
        continue

    # Future price change (next 4 H1 bars ≈ next 16 M15 lookahead)
    future_close = None
    future_high = None
    future_low = None
    for offset in [1, 2, 3, 4, 6, 8, 12]:
        fut_idx = h1_i + offset
        if fut_idx < len(h1):
            if future_close is None:
                future_close = float(h1['close'].iloc[fut_idx])
            future_high = max(future_high or 0, float(h1['high'].iloc[fut_idx]))
            future_low = min(future_low or 1e9, float(h1['low'].iloc[fut_idx]))

    if future_close is None:
        continue

    future_change_pct = (future_close / price - 1) * 100
    future_high_pct = (future_high / price - 1) * 100 if future_high else 0
    future_low_pct = (future_low / price - 1) * 100 if future_low else 0

    records.append({
        "ts": ts_naive,
        "price": price,
        "model_regime": model_regime,
        "model_conf": round(model_conf, 4),
        "rule_regime": rule_regime,
        "rule_conf": round(rule_conf, 4),
        "rule_slope": round(rule_slope, 6),
        "final_regime": final_regime,
        "final_conf": round(final_conf, 4),
        "oracle_label": ol.label,
        "oracle_long": ol.long_r,
        "oracle_short": ol.short_r,
        "future_pct": round(future_change_pct, 4),
        "future_high_pct": round(future_high_pct, 4),
        "future_low_pct": round(future_low_pct, 4),
        "oracle_best_dir": "LONG" if ol.long_r > ol.short_r else "SHORT",
        "oracle_best_r": max(ol.long_r, ol.short_r),
    })

df = pd.DataFrame(records)
n = len(df)
print(f"\n  {n} H1 bars analyzed")

# ── Analysis 1: Does model confidence predict future direction? ──
print(f"\n{'─'*80}")
print("TEST 1: Model confidence vs next 4-12 bar price change")
print(f"{'─'*80}")

# Split by confidence bins
for label, lo, hi in [("Low (0.0-0.4)", 0, 0.4), ("Med (0.4-0.7)", 0.4, 0.7),
                        ("High (0.7-0.9)", 0.7, 0.9), ("Extreme (0.9-1.0)", 0.9, 1.0)]:
    subset = df[(df['model_conf'] >= lo) & (df['model_conf'] < hi)]
    if len(subset) == 0: continue
    up = (subset['future_pct'] > 0).sum()
    dn = (subset['future_pct'] < 0).sum()
    avg_fwd = subset['future_pct'].mean()
    avg_high = subset['future_high_pct'].mean()
    avg_low = subset['future_low_pct'].mean()

    # What oracle says
    oracle_long = subset['oracle_long'].mean()
    oracle_short = subset['oracle_short'].mean()

    print(f"\n  {label} (n={len(subset)}):")
    print(f"    Future close: {avg_fwd:+.2f}%  (up={up}, dn={dn})")
    print(f"    Future high:  {avg_high:+.2f}%  Future low: {avg_low:+.2f}%")
    print(f"    Oracle long:  {oracle_long:+.2f}R  Oracle short: {oracle_short:+.2f}R")

# ── Analysis 2: Confidence CHANGE as reversal signal ──
print(f"\n{'─'*80}")
print("TEST 2: Falling confidence as reversal signal")
print(f"{'─'*80}")

# Compute confidence delta from previous bar
df['conf_delta'] = df['model_conf'].diff()
df['conf_prev'] = df['model_conf'].shift(1)

# Bars where confidence dropped significantly
drop_05 = df[df['conf_delta'] < -0.05]
drop_10 = df[df['conf_delta'] < -0.10]
drop_20 = df[df['conf_delta'] < -0.20]
stable = df[(df['conf_delta'].abs() < 0.02)]

for label, subset in [("Drop > 0.05", drop_05), ("Drop > 0.10", drop_10),
                        ("Drop > 0.20", drop_20), ("Stable (±0.02)", stable)]:
    if len(subset) == 0: continue
    up = (subset['future_pct'] > 0).sum()
    dn = (subset['future_pct'] < 0).sum()
    print(f"\n  {label} (n={len(subset)}):")
    print(f"    Future close: {subset['future_pct'].mean():+.2f}%  (up={up}, dn={dn})")
    print(f"    Future high:  {subset['future_high_pct'].mean():+.2f}%")
    print(f"    Oracle best R: {subset['oracle_best_r'].mean():+.2f}")

# ── Analysis 3: The exhaustion hypothesis — fade high confidence ──
print(f"\n{'─'*80}")
print("TEST 3: Fade the model — invert high confidence signals")
print(f"{'─'*80}")

# Strategy A: Follow model (current bot behavior)
#   Enter when model is confident in TREND_UP → go LONG
# Strategy B: Fade model (exhaustion hypothesis)
#   When model is EXTREMELY confident in TREND_UP → go SHORT (trend exhausted)
#   When model confidence FALLS → go with rule detector direction

# For each bar where model says TREND_UP:
trend_up_bars = df[df['model_regime'] == 'TREND_UP']
trend_dn_bars = df[df['model_regime'] == 'TREND_DOWN']

print(f"\n  When model says TREND_UP (n={len(trend_up_bars)}):")
print(f"    Oracle says LONG:  {(trend_up_bars['oracle_best_dir']=='LONG').mean()*100:.1f}% of the time")
print(f"    Oracle best R:     {trend_up_bars['oracle_best_r'].mean():.2f}")

# But when model says TREND_UP with EXTREME confidence:
extreme_up = trend_up_bars[trend_up_bars['model_conf'] > 0.9]
print(f"    When EXTREME confidence (>0.9, n={len(extreme_up)}):")
print(f"      Oracle LONG: {(extreme_up['oracle_best_dir']=='LONG').mean()*100:.1f}%")
print(f"      Oracle best R: {extreme_up['oracle_best_r'].mean():.2f}")

low_up = trend_up_bars[trend_up_bars['model_conf'] < 0.5]
print(f"    When LOW confidence (<0.5, n={len(low_up)}):")
print(f"      Oracle LONG: {(low_up['oracle_best_dir']=='LONG').mean()*100:.1f}%")
print(f"      Oracle best R: {low_up['oracle_best_r'].mean():.2f}")

print(f"\n  When model says TREND_DOWN (n={len(trend_dn_bars)}):")
print(f"    Oracle says SHORT: {(trend_dn_bars['oracle_best_dir']=='SHORT').mean()*100:.1f}% of the time")
print(f"    Oracle best R:     {trend_dn_bars['oracle_best_r'].mean():.2f}")

extreme_dn = trend_dn_bars[trend_dn_bars['model_conf'] > 0.9]
print(f"    When EXTREME confidence (>0.9, n={len(extreme_dn)}):")
print(f"      Oracle SHORT: {(extreme_dn['oracle_best_dir']=='SHORT').mean()*100:.1f}%")
print(f"      Oracle best R: {extreme_dn['oracle_best_r'].mean():.2f}")
print(f"      Future close: {extreme_dn['future_pct'].mean():+.2f}%")
print(f"      Model was {'RIGHT' if extreme_dn['oracle_best_dir'].iloc[0]=='SHORT' else 'WRONG — fade it!' if len(extreme_dn)>0 else 'N/A'}")

low_dn = trend_dn_bars[trend_dn_bars['model_conf'] < 0.5]
print(f"    When LOW confidence (<0.5, n={len(low_dn)}):")
print(f"      Oracle SHORT: {(low_dn['oracle_best_dir']=='SHORT').mean()*100:.1f}%")
print(f"      Oracle best R: {low_dn['oracle_best_r'].mean():.2f}")
print(f"      Future close: {low_dn['future_pct'].mean():+.2f}%")

# ── Analysis 4: Conf delta vs rule detector — who's right? ──
print(f"\n{'─'*80}")
print("TEST 4: Confidence collapse + rule detector flip = reversal?")
print(f"{'─'*80}")

# Find bars where: conf drops > 0.1 AND rule detector disagrees with model
df['model_vs_rule'] = np.where(
    (df['model_regime'].isin(['TREND_UP','TREND_DOWN'])) &
    (df['rule_regime'].isin(['TREND_UP','TREND_DOWN'])) &
    (df['model_regime'] != df['rule_regime']),
    'conflict', 'agree')

conflict_bars = df[df['model_vs_rule'] == 'conflict']
print(f"\n  Model-rule conflicts: {len(conflict_bars)} bars")
if len(conflict_bars) > 0:
    # Who was right?
    rule_right = 0; model_right = 0; neither = 0
    for _, r in conflict_bars.iterrows():
        if r['rule_regime'] == r['oracle_best_dir']:
            rule_right += 1
        elif r['model_regime'] == r['oracle_best_dir']:
            model_right += 1
        else:
            neither += 1
    print(f"  Rule right: {rule_right} ({rule_right/len(conflict_bars)*100:.0f}%)")
    print(f"  Model right: {model_right} ({model_right/len(conflict_bars)*100:.0f}%)")
    print(f"  Neither: {neither}")

    # When confidence is FALLING during a conflict
    conflict_drop = conflict_bars[conflict_bars['conf_delta'] < -0.05]
    if len(conflict_drop) > 0:
        rule_right_d = 0; model_right_d = 0
        for _, r in conflict_drop.iterrows():
            if r['rule_regime'] == r['oracle_best_dir']:
                rule_right_d += 1
            elif r['model_regime'] == r['oracle_best_dir']:
                model_right_d += 1
        print(f"\n  During confidence DROP + conflict (n={len(conflict_drop)}):")
        print(f"  Rule right: {rule_right_d} ({rule_right_d/len(conflict_drop)*100:.0f}%)")
        print(f"  Model right: {model_right_d} ({model_right_d/len(conflict_drop)*100:.0f}%)")

# ── Summary ──
print(f"\n{'='*80}")
print("SUMMARY: Is 'confidence' actually trend exhaustion?")
print("=" * 80)

# The key test: fade extreme confidence
extreme_all = df[df['model_conf'] > 0.9]
fade_correct = (extreme_all['oracle_best_dir'] != extreme_all['model_regime']).sum() if len(extreme_all) > 0 else 0
fade_pct = fade_correct / len(extreme_all) * 100 if len(extreme_all) > 0 else 0

drop_all = df[df['conf_delta'] < -0.10]
drop_oracle = drop_all['oracle_best_r'].mean() if len(drop_all) > 0 else 0
stable_oracle = df[df['conf_delta'].abs() < 0.02]['oracle_best_r'].mean()

print(f"""
  Extreme confidence (>0.9): fade would be correct {fade_correct}/{len(extreme_all)} ({fade_pct:.0f}%) of the time
  Confidence drop >0.10: avg oracle R = {drop_oracle:.2f}
  Stable confidence:     avg oracle R = {stable_oracle:.2f}
""")

if fade_pct > 55:
    print("  → YES: Model confidence works better as a CONTRARIAN indicator.")
    print("    High confidence = trend about to exhaust = FADE the signal.")
elif fade_pct < 45:
    print("  → NO: Model confidence is not contrarian — it's just random/broken.")
else:
    print("  → INCONCLUSIVE: Model confidence is close to 50/50.")
