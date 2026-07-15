"""
M15 Timing Audit — Full Month (May 2026).

For every H1 listening window during the month:
  Compare strategies for picking the best M15 entry bar.
"""
import sys, os, json
import numpy as np, pandas as pd, torch
from datetime import datetime, timedelta

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import (RuleBasedRegimeDetector, classify_regime,
                                       REGIME_NAMES)
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier
from models.entry_gate import EntryGate
from benchmark.oracle_m15 import M15OracleLabeler
from backtest.data_manager import BacktestDataManager

cfg = BTCConfig()
device = torch.device("cpu")
fe = BTCFeatureEngine()

# ── Load models ──
print("Loading models...")

# M15 model
m15_path = os.path.join(cfg.model_dir, "btc_m15_v2.pt")
if not os.path.exists(m15_path):
    m15_path = os.path.join(cfg.model_dir, "btc_m15_model.pt")
m15_model = CNNGRUM15(
    n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).to(device).eval()
ckpt = torch.load(m15_path, map_location=device, weights_only=False)
sd = ckpt["model_state_dict"]
if "conv1.0.weight" in sd:
    remapped = {}
    bs = {"conv1":0,"conv2":4,"conv3":8}
    for ok, v in sd.items():
        pfx = ok.split(".")[0]
        if pfx in bs:
            rest = ok.split(".",1)[1]; si = int(rest.split(".")[0])
            nk = f"cnn.{bs[pfx]+si}.{rest.split('.',1)[1]}"
        elif ok.startswith("entry_head."): nk = ok.replace("entry_head.","entry_conf.",1)
        else: nk = ok
        remapped[nk] = v
    sd = remapped
m15_model.load_state_dict(sd, strict=False)
print(f"  M15 model loaded")

# H1 models
encoder_path = os.path.join(cfg.model_dir, "btc_h1_encoder.pt")
ckpt_h1 = torch.load(encoder_path, map_location=device, weights_only=False)
encoder = CNNLSTMEncoder(
    n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).to(device).eval()
encoder.load_state_dict(ckpt_h1["encoder_state_dict"])
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim,
                               n_classes=cfg.regime_classes).to(device).eval()
classifier.load_state_dict(ckpt_h1["classifier_state_dict"])
print(f"  H1 models loaded")

# ── Load oracle ──
print("\nLoading cached oracle...")
import pickle
with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle_labels = pickle.load(f)
# Filter to May
oracle_labels = [ol for ol in oracle_labels if ol.timestamp.startswith("2026-05")]
oracle_by_ts = {}
for ol in oracle_labels:
    oracle_by_ts[ol.timestamp] = ol
print(f"  {len(oracle_labels)} May oracle labels ({oracle_labels[0].timestamp} → {oracle_labels[-1].timestamp})")

# ── Load backtest data for May ──
print("\nLoading backtest data...")
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-05-01", "2026-05-26", use_cache=True)
h1 = ds.h1_df; m15 = ds.m15_df
h1_feats = ds.h1_features
h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile
h1_rule_regime = ds.h1_rule_regime
print(f"  H1: {len(h1)} bars, M15: {len(m15)} bars")

# ── Walk through timeline building listening windows ──
# This is a pure data-collection pass: evaluate M15 model at EVERY bar
# within EVERY listening window. No trade simulation, no position tracking.
print("\nWalking timeline to find H1 listening windows...")

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile,
                 max_atr_pct=cfg.max_atr_percentile)
rule_det = RuleBasedRegimeDetector()

# Pre-warm rule detector
warm_h1 = h1.iloc[:cfg.seq_len_h1]
for _, row in warm_h1.iterrows():
    rule_det.update(row["high"], row["low"], row["close"])

min_h1 = cfg.seq_len_h1
warm_h1_ts = h1['timestamp'].iloc[min_h1 - 1]
# Ensure timezone compatibility
if hasattr(warm_h1_ts, 'tz') and warm_h1_ts.tz is not None:
    warm_m15 = int((m15['timestamp'].dt.tz_localize(None) < warm_h1_ts.tz_localize(None)).sum())
else:
    warm_m15 = int((m15['timestamp'] < warm_h1_ts).sum())
warm_m15 = max(warm_m15, cfg.seq_len_m15)
print(f"  warm_m15 = {warm_m15} / {len(m15)}")

# State for listening windows
last_h1_idx = -1
listening = False
h1_signal = 0
bars_listened = 0
max_listen = cfg.max_listen_bars

# Results per listening window
window_data = {}  # window_id → list of bar dicts
window_id = 0
active_window = None

for m15_i in range(warm_m15, len(m15)):
    ts = m15['timestamp'].iloc[m15_i]
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts_naive = ts.tz_localize(None)
    else:
        ts_naive = ts
    price = float(m15['close'].iloc[m15_i])

    # Get corresponding H1 index
    if hasattr(h1['timestamp'].iloc[0], 'tz') and h1['timestamp'].iloc[0].tz is not None:
        h1_i = int((h1['timestamp'].dt.tz_localize(None) <= ts_naive).sum() - 1)
    else:
        h1_i = int((h1['timestamp'] <= ts_naive).sum() - 1)

    # ── H1 bar close ──
    if h1_i != last_h1_idx and h1_i >= min_h1:
        last_h1_idx = h1_i

        # Update rule detector
        for _, row in h1.iloc[max(0, h1_i-14):h1_i+1].iterrows():
            rule_det.update(row["high"], row["low"], row["close"])

        # classify_regime
        seq = fe.compute_sequence(h1_feats, h1_i, cfg.seq_len_h1)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
        regime_result = classify_regime(encoder, classifier, tensor, rule_det,
                                         model_confidence_threshold=cfg.min_regime_confidence)

        # Rule-wins-conflict
        rule_classification = rule_det._classify()
        trending = {"TREND_UP", "TREND_DOWN"}
        if (regime_result["regime"] in trending and
            rule_classification["regime"] in trending and
            regime_result["regime"] != rule_classification["regime"]):
            regime_result["regime"] = rule_classification["regime"]
            regime_result["confidence"] = rule_classification["confidence"]

        # Gate
        atr_pct = float(h1_atr_pctl[h1_i])
        bb_pos = float(h1_feats[h1_i, 4])
        gd = gate.evaluate(regime_result["regime"], regime_result["confidence"],
                           atr_pct, bb_position=bb_pos)

        # Start or end listening window
        if gd.entry_signal:
            h1_signal = gd.direction
            listening = True
            bars_listened = 0
            active_window = window_id
            window_data[window_id] = []
            window_id += 1
        else:
            listening = False
            active_window = None

    if not listening:
        continue

    bars_listened += 1
    if bars_listened > max_listen:
        listening = False
        active_window = None
        continue

    # ── M15 model inference ──
    if m15_i >= cfg.seq_len_m15:
        window_m15 = m15.iloc[m15_i - cfg.seq_len_m15 + 1:m15_i + 1]
        if len(window_m15) < cfg.seq_len_m15:
            continue
        m15_feats = fe.compute(window_m15)
        seq_m15 = fe.compute_sequence(m15_feats, len(m15_feats)-1, cfg.seq_len_m15)
        tensor_m15 = torch.from_numpy(seq_m15).unsqueeze(0).to(device)
        with torch.no_grad():
            out = m15_model(tensor_m15)
            m15_conf = out["entry_confidence"].item()
            m15_dir_bias = out["direction_bias"].item()
    else:
        m15_conf = 0.0
        m15_dir_bias = 0.0

    # EMA turning
    closes = m15['close'].values[:m15_i+1]
    if len(closes) >= 3:
        if h1_signal == 1:
            ema_turn = closes[-1] > closes[-2]
        else:
            ema_turn = closes[-1] < closes[-2]
    else:
        ema_turn = False

    # Oracle label — try exact match then ± offsets
    ts_str = str(ts_naive)[:19]
    ol = oracle_by_ts.get(ts_str)
    if ol is None:
        for offset_m in [15, -15, 30, -30]:
            adj = ts_naive + timedelta(minutes=offset_m)
            ol = oracle_by_ts.get(str(adj)[:19])
            if ol: break

    if ol:
        oracle_r = ol.long_r if h1_signal == 1 else ol.short_r
        oracle_label = ol.label
    else:
        oracle_r = 0.0
        oracle_label = "?"

    if active_window is not None:
        window_data[active_window].append({
            "ts": ts_naive, "price": price,
            "m15_conf": round(m15_conf, 4),
            "m15_dir_bias": round(m15_dir_bias, 4),
            "ema_turn": ema_turn,
            "oracle_r": round(oracle_r, 4),
            "oracle_label": oracle_label,
            "h1_signal": h1_signal,
        })

print(f"  {len(window_data)} listening windows found")
total_bars = sum(len(bars) for bars in window_data.values())
print(f"  {total_bars} M15 bars evaluated")

# ── Filter: only windows with 2+ bars ──
valid_windows = {k: v for k, v in window_data.items() if len(v) >= 2}
print(f"  {len(valid_windows)} valid windows (2+ bars)")

# ── Compare strategies ──
print(f"\n{'='*80}")
print(f"STRATEGY COMPARISON — {len(valid_windows)} LISTENING WINDOWS (MAY 2026)")
print("=" * 80)

strategies = {
    "FIRST": lambda bars: bars[0],
    "EMA": lambda bars: next((b for b in bars if b["ema_turn"]), bars[0]),
    "M15_CONF": lambda bars: max(bars, key=lambda b: b["m15_conf"]),
    "M15_CONF_LOW": lambda bars: min(bars, key=lambda b: b["m15_conf"]),
    "RANDOM": lambda bars: bars[np.random.randint(0, len(bars))],
    "ORACLE": lambda bars: max(bars, key=lambda b: b["oracle_r"]),
}

results = {name: [] for name in strategies}
for wi, bars in valid_windows.items():
    best_r = max(b["oracle_r"] for b in bars)
    n_bars = len(bars)
    for name, pick_fn in strategies.items():
        picked = pick_fn(bars)
        results[name].append({
            "oracle_r": picked["oracle_r"],
            "is_best": picked["oracle_r"] == best_r,
            "rank": sorted(bars, key=lambda b: b["oracle_r"], reverse=True).index(picked) + 1,
            "n_bars": n_bars,
            "h1_signal": bars[0]["h1_signal"],
        })

print(f"\n{'Strategy':<18s} {'Windows':>8s} {'Avg OracleR':>12s} {'% of Best':>10s} {'Avg Rank':>10s} {'Hit Best':>10s}")
print("-" * 80)

for name in ["FIRST", "EMA", "M15_CONF", "M15_CONF_LOW", "RANDOM", "ORACLE"]:
    vals = results[name]
    avg_r = np.mean([v["oracle_r"] for v in vals])
    best_rs = []
    for wi, bars in valid_windows.items():
        best_rs.append(max(b["oracle_r"] for b in bars))
    avg_best = np.mean(best_rs)
    pct = (avg_r / avg_best * 100) if avg_best > 0 else 0
    avg_rank = np.mean([v["rank"] for v in vals])
    hit_best = sum(1 for v in vals if v["is_best"])
    print(f"{name:<18s} {len(vals):>8d} {avg_r:>+12.4f}R {pct:>9.1f}% {avg_rank:>10.2f} {hit_best:>8d}/{len(vals)}")

# ── EMA vs M15 head-to-head ──
ema_wins = 0; m15_wins = 0; ties = 0
for wi, bars in valid_windows.items():
    ema_pick = next((b for b in bars if b["ema_turn"]), bars[0])
    m15_pick = max(bars, key=lambda b: b["m15_conf"])
    if ema_pick["oracle_r"] > m15_pick["oracle_r"]: ema_wins += 1
    elif m15_pick["oracle_r"] > ema_pick["oracle_r"]: m15_wins += 1
    else: ties += 1

print(f"\nEMA vs M15_CONF Head-to-Head:")
print(f"  EMA wins:      {ema_wins}")
print(f"  M15_CONF wins: {m15_wins}")
print(f"  Ties:          {ties}")
if ema_wins + m15_wins > 0:
    print(f"  M15 win rate:  {m15_wins/(ema_wins+m15_wins)*100:.1f}%")

# ── Spearman rank correlation ──
all_conf = []; all_oracle = []
for wi, bars in valid_windows.items():
    for b in bars:
        all_conf.append(b["m15_conf"])
        all_oracle.append(b["oracle_r"])

from scipy.stats import spearmanr
rho, pval = spearmanr(all_conf, all_oracle)
print(f"\nOverall Spearman correlation (M15 conf vs oracle R):")
print(f"  rho = {rho:+.4f}  p = {pval:.6f}  n = {len(all_conf)}")
if pval < 0.01: print(f"  *** Statistically significant at p<0.01 ***")
elif pval < 0.05: print(f"  ** Statistically significant at p<0.05 **")
else: print(f"  — Not statistically significant")

# ── By direction ──
print(f"\n{'='*80}")
print("BY SIGNAL DIRECTION")
print("=" * 80)

for direction, label in [(1, "LONG"), (-1, "SHORT")]:
    dir_windows = {k: v for k, v in valid_windows.items() if v[0]["h1_signal"] == direction}
    if len(dir_windows) < 3:
        print(f"\n  {label}: only {len(dir_windows)} windows — skipping")
        continue

    dir_results = {}
    for name in ["FIRST", "EMA", "M15_CONF", "ORACLE"]:
        fn = strategies[name]
        vals = []
        for wi, bars in dir_windows.items():
            picked = fn(bars)
            best_r = max(b["oracle_r"] for b in bars)
            vals.append({"oracle_r": picked["oracle_r"], "rank": sorted(
                bars, key=lambda b: b["oracle_r"], reverse=True).index(picked)+1})
        dir_results[name] = vals

    best_rs = [max(b["oracle_r"] for b in bars) for bars in dir_windows.values()]
    avg_best = np.mean(best_rs)

    ema_w = 0; m15_w = 0
    for wi, bars in dir_windows.items():
        ema_pick = next((b for b in bars if b["ema_turn"]), bars[0])
        m15_pick = max(bars, key=lambda b: b["m15_conf"])
        if ema_pick["oracle_r"] > m15_pick["oracle_r"]: ema_w += 1
        elif m15_pick["oracle_r"] > ema_pick["oracle_r"]: m15_w += 1

    print(f"\n  {label} signals: {len(dir_windows)} windows")
    for name in ["FIRST", "EMA", "M15_CONF", "ORACLE"]:
        vals = dir_results[name]
        avg_r = np.mean([v["oracle_r"] for v in vals])
        pct = avg_r / avg_best * 100 if avg_best > 0 else 0
        avg_rank = np.mean([v["rank"] for v in vals])
        print(f"    {name:<14s}: avg={avg_r:+.4f}R ({pct:.1f}%)  rank={avg_rank:.2f}")
    print(f"    EMA vs M15: EMA wins {ema_w}, M15 wins {m15_w} → M15={m15_w/(ema_w+m15_w)*100:.0f}%") if ema_w+m15_w > 0 else None

print(f"\nDone.")
