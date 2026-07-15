"""Trace regime/confidence at the live bot's entry times."""
import sys, os, json
import numpy as np, pandas as pd, torch

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import (RegimeClassifier, RuleBasedRegimeDetector,
                                       classify_regime, REGIME_NAMES)
from models.entry_gate import EntryGate

cfg = BTCConfig()
device = torch.device("cpu")

# Load H1 and M15 data
h1_path = os.path.join(cfg.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
m15_path = os.path.join(cfg.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")

h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
m15 = pd.read_csv(m15_path); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)

# Load models
encoder_path = os.path.join(cfg.model_dir, "btc_h1_encoder.pt")
ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
encoder = CNNLSTMEncoder(
    n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim,
    regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).to(device).eval()
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim,
                               n_classes=cfg.regime_classes).to(device).eval()
classifier.load_state_dict(ckpt["classifier_state_dict"])

fe = BTCFeatureEngine()
gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile,
                 max_atr_pct=cfg.max_atr_percentile)

# Live bot trade times from the status (approximate — last_trade_time)
# The bot reset its trade_history on restart, so we don't have exact times.
# Instead, reconstruct: the status shows balance=$446.6, daily_pnl=-$91.38
# starting balance ~$538 (from earlier context).
# We'll trace the entire period the bot was running: May 25 22:58 UTC → now
# and show every H1 bar evaluation.

print("=" * 90)
print("TRACING REGIME DETECTION FOR LIVE BOT (May 25 22:00 UTC onward)")
print("=" * 90)

# Filter to recent period — use last 4 days of data
print(f"Data range: {h1['timestamp'].min()} → {h1['timestamp'].max()}")
print(f"M15 range: {m15['timestamp'].min()} → {m15['timestamp'].max()}")

# Take last 200 H1 bars (enough context + recent bars)
h1_full = h1.tail(200).reset_index(drop=True)
m15_recent = m15[m15["timestamp"] >= h1_full["timestamp"].iloc[0]].reset_index(drop=True)

# Initialize rule detector with history before the window
rule_det = RuleBasedRegimeDetector()
pre_start = h1_full["timestamp"].iloc[0]
pre_bars = h1[h1["timestamp"] < pre_start].tail(50)
for _, row in pre_bars.iterrows():
    rule_det.update(row["high"], row["low"], row["close"])

# Walk through H1 bars
print(f"\n{'H1 Time':20s} {'Model':12s} {'ModelConf':10s} {'Rule':12s} {'RuleConf':10s} {'Final':12s} {'Gate':8s} {'Dir':>4s} {'Reason'}")
print("-" * 110)

for i in range(cfg.seq_len_h1, len(h1_full)):
    window = h1_full.iloc[:i+1]
    ts = window["timestamp"].iloc[-1]

    # Update rule detector
    row = window.iloc[-1]
    rule_det.update(row["high"], row["low"], row["close"])

    # Feature computation
    feats = fe.compute(window)
    seq = fe.compute_sequence(feats, len(feats) - 1, cfg.seq_len_h1)
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

    # Model regime
    with torch.no_grad():
        enc_out = encoder(tensor)
        raw_logits = classifier.raw_logits(enc_out["embedding"])
        probs = torch.softmax(raw_logits / cfg.regime_temperature, dim=1)
        max_prob, pred_class = probs.max(dim=1)
        max_prob = max_prob.item(); pred_class = pred_class.item()

    model_regime = REGIME_NAMES[pred_class]
    model_conf = max_prob
    all_probs = probs.squeeze(0).tolist()

    # Rule detector
    rule_out = rule_det._classify()
    rule_regime = rule_out["regime"]
    rule_conf = rule_out["confidence"]

    # Full classify_regime logic
    if max_prob >= cfg.min_regime_confidence:
        final_regime = model_regime
        final_conf = model_conf
        source = "model"
    else:
        final_regime = rule_regime
        final_conf = rule_conf
        source = "rule"

    # Rule-wins-conflict
    trending = {"TREND_UP", "TREND_DOWN"}
    if (final_regime in trending and rule_regime in trending
            and final_regime != rule_regime):
        final_regime = rule_regime
        final_conf = rule_conf
        source = "rule(wins)"

    # Gate evaluation
    atr_pct = rule_out.get("atr_percentile", 0.5)
    bb_pos = feats[-1, 4] if feats.shape[1] > 4 else 0.0
    gd = gate.evaluate(final_regime, final_conf, atr_pct, bb_position=bb_pos)

    dir_str = f"{'LONG' if gd.direction==1 else 'SHORT' if gd.direction==-1 else '-'}"
    signal_str = "SIGNAL" if gd.entry_signal else "blocked"

    print(f"{str(ts)[:19]:20s} {model_regime:12s} {model_conf:.3f}      {rule_regime:12s} {rule_conf:.3f}      {final_regime:12s} {signal_str:8s} {dir_str:>4s}   {gd.reason}")

    # Also show M15 entry opportunities within this H1 bar
    if gd.entry_signal:
        h1_bar_end = ts + pd.Timedelta(hours=1)
        m15_in_h1 = m15_recent[(m15_recent["timestamp"] > ts) & (m15_recent["timestamp"] <= h1_bar_end)]

        if len(m15_in_h1) > 0:
            closes = pd.concat([m15_recent[m15_recent["timestamp"] <= ts].tail(3), m15_in_h1])["close"].values
            for j in range(len(m15_in_h1)):
                m15_ts = m15_in_h1["timestamp"].iloc[j]
                m15_price = m15_in_h1["close"].iloc[j]
                # turning check
                all_closes = np.append(m15_recent[m15_recent["timestamp"] <= m15_ts].tail(2)["close"].values, m15_price)
                if len(all_closes) >= 2:
                    if gd.direction == 1:
                        turning = all_closes[-1] > all_closes[-2]
                    elif gd.direction == -1:
                        turning = all_closes[-1] < all_closes[-2]
                    else:
                        turning = False
                    turn_str = "✓ TURN" if turning else "  no turn"
                    print(f"  └─ M15 {str(m15_ts)[:19]}  price={m15_price:.1f}  {turn_str}")

print("\nDone.")
