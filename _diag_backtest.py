"""Diagnose backtester signal flow vs test script."""
import sys, os, numpy as np, pandas as pd, torch
from datetime import datetime
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from backtest.data_manager import BacktestDataManager

cfg = BTCConfig()
engine = BTCFeatureEngine()

# Load models (same as backtester)
encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir+"/btc_h1_encoder.pt", map_location="cpu", weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)

# Load data via the fixed data manager
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-05-25", use_cache=False, force_refresh=False)

h1_df = ds.h1_df
h1_feats = ds.h1_features

# Walk H1 bars and count gate signals
rd = RuleBasedRegimeDetector()
for i in range(cfg.seq_len_h1):
    rd.update(float(h1_df["high"].iloc[i]), float(h1_df["low"].iloc[i]), float(h1_df["close"].iloc[i]))

model_signals = 0
rule_signals = 0
range_signals = 0
transition_signals = 0
blocked_ema22 = 0
model_regimes = {"TREND_UP":0,"TREND_DOWN":0,"RANGE":0,"TRANSITION":0}
rule_regimes = {"TREND_UP":0,"TREND_DOWN":0,"RANGE":0,"TRANSITION":0}
conflicts = 0
ema22_blocks = 0

for hi in range(cfg.seq_len_h1, len(h1_df)):
    seq = engine.compute_sequence(h1_feats, hi, cfg.seq_len_h1)
    t = torch.from_numpy(seq).unsqueeze(0)
    for j in range(max(0, hi-13), hi+1):
        rd.update(float(h1_df["high"].iloc[j]), float(h1_df["low"].iloc[j]), float(h1_df["close"].iloc[j]))
    rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
    rule_out = rd._classify()

    model_regimes[rr["regime"]] = model_regimes.get(rr["regime"], 0) + 1
    rule_regimes[rule_out["regime"]] = rule_regimes.get(rule_out["regime"], 0) + 1

    # Count conflicts
    trending = {"TREND_UP", "TREND_DOWN"}
    if rr["regime"] in trending and rule_out["regime"] in trending and rr["regime"] != rule_out["regime"]:
        conflicts += 1

    gd = gate.evaluate(rr["regime"], rr["confidence"],
                       float(rr.get("atr_percentile", 0.5)),
                       bb_position=float(h1_feats[hi, 4]))

    if gd.entry_signal:
        if rr["source"] == "model":
            model_signals += 1
        else:
            rule_signals += 1

        # Check EMA22
        hc = h1_df["close"].values[:hi+1]
        if len(hc) >= 23:
            e22 = pd.Series(hc).ewm(span=22, adjust=False).mean().values
            slp = (e22[-1]-e22[-2])/max(abs(float(e22[-2])),1e-12)
            if not ((gd.direction==1 and slp>0) or (gd.direction==-1 and slp<0)):
                ema22_blocks += 1

    # Count by gate result
    if not gd.entry_signal:
        if rr["regime"] in ("TREND_UP", "TREND_DOWN"):
            pass  # blocked by confidence or vol
        elif rr["regime"] == "RANGE":
            range_signals += 1
        else:
            transition_signals += 1

total = len(h1_df) - cfg.seq_len_h1
print(f"H1 bars analyzed: {total}")
print(f"Model regime dist: {model_regimes}")
print(f"Rule regime dist:  {rule_regimes}")
print(f"TREND conflicts: {conflicts}")
print(f"Gate entry signals: {model_signals + rule_signals} (model={model_signals}, rule={rule_signals})")
print(f"  EMA22 would block: {ema22_blocks}")
print(f"Gate blocked: RANGE bars={range_signals}, TRANSITION={transition_signals}")
print(f"")

# M15 stats
m15c = ds.m15_confidence
m15c_pos = m15c[m15c > 0]
print(f"M15 confidence stats:")
print(f"  N with signal: {len(m15c_pos)} / {len(m15c)}")
if len(m15c_pos) > 0:
    print(f"  Mean: {m15c_pos.mean():.4f}  Median: {np.median(m15c_pos):.4f}")
    print(f"  >0.5: {(m15c_pos>0.5).sum()} ({((m15c_pos>0.5).sum()/len(m15c_pos)*100):.1f}%)")
    print(f"  >0.6: {(m15c_pos>0.6).sum()} ({((m15c_pos>0.6).sum()/len(m15c_pos)*100):.1f}%)")
    print(f"  Distribution: p10={np.percentile(m15c_pos,10):.4f} p25={np.percentile(m15c_pos,25):.4f} p50={np.percentile(m15c_pos,50):.4f} p75={np.percentile(m15c_pos,75):.4f} p90={np.percentile(m15c_pos,90):.4f}")
