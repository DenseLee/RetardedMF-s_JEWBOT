"""Collect M15 training data: feature sequences + direction-specific trade outcome labels."""
import sys, os, numpy as np, pandas as pd, torch, pickle
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType, Phase

config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = CNNLSTMEncoder(n_features=config.n_features, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
    bidirectional=True).to(device).eval()
classifier = RegimeClassifier(embedding_dim=config.embedding_dim, n_classes=config.regime_classes).to(device).eval()
ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder.pt"), map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"]); classifier.load_state_dict(ckpt["classifier_state_dict"])
engine = BTCFeatureEngine(); gate = EntryGate()

h1f = pd.read_csv(os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv"))
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)

# Use training window: 2022-2025
ft = pd.Timestamp("2022-01-01", tz="UTC"); et = pd.Timestamp("2025-12-31", tz="UTC")
h1f = h1f[(h1f["timestamp"] >= ft) & (h1f["timestamp"] < et)].reset_index(drop=True)
m15f = m15f[(m15f["timestamp"] >= ft) & (m15f["timestamp"] < et)].reset_index(drop=True)

BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}

print(f"Collecting M15 labels from {ft.date()} to {et.date()}")
print(f"H1 bars: {len(h1f)}, M15 bars: {len(m15f)}")
print(f"At each listening bar, simulating trade outcome via trade manager...")

# Simulate trade outcome for a hypothetical entry
def simulate_trade_outcome(entry_price, direction, entry_atr, m15f, start_idx, max_bars=18):
    """Run trade manager forward from start_idx to determine PnL outcome."""
    sl_mult = 1.0; tp_mult = 3.0; be_trigger = 0.50
    trail_trigger = 2.5; trail_dist = 0.75; trail_dist_s = 0.50
    max_hold = max_bars
    regime_tighten = 0.40

    sl_price = entry_price - direction * sl_mult * entry_atr
    tp_price = entry_price + direction * tp_mult * entry_atr
    best_price = entry_price
    be_activated = False; trail_active = False
    current_sl = sl_price

    end_idx = min(start_idx + max_hold, len(m15f))

    for j in range(start_idx, end_idx):
        hi = m15f["high"].iloc[j]; lo = m15f["low"].iloc[j]; close = m15f["close"].iloc[j]

        # Update best price
        if direction == 1:
            best_price = max(best_price, hi)
            profit_r = (hi - entry_price) / entry_atr
        else:
            best_price = min(best_price, lo)
            profit_r = (entry_price - lo) / entry_atr

        # Check SL hit
        if direction == 1 and lo <= current_sl:
            return (current_sl - entry_price) / entry_atr, "sl_hit", j - start_idx + 1
        elif direction == -1 and hi >= current_sl:
            return (entry_price - current_sl) / entry_atr, "sl_hit", j - start_idx + 1

        # Check TP hit
        if direction == 1 and hi >= tp_price:
            return (tp_price - entry_price) / entry_atr, "tp_hit", j - start_idx + 1
        elif direction == -1 and lo <= tp_price:
            return (entry_price - tp_price) / entry_atr, "tp_hit", j - start_idx + 1

        # Phase transitions
        if not be_activated and profit_r >= be_trigger:
            be_activated = True
            be_sl = entry_price + direction * 0.05 * entry_atr  # slightly above entry
            current_sl = be_sl

        if not trail_active and profit_r >= trail_trigger:
            trail_active = True

        if trail_active:
            if direction == 1:
                trail_sl = best_price - trail_dist * entry_atr
                trail_sl = max(trail_sl, current_sl)  # don't loosen
            else:
                trail_sl = best_price + trail_dist * entry_atr
                trail_sl = min(trail_sl, current_sl)
            current_sl = trail_sl

    # Time stop — exit at close
    return (close - entry_price) / entry_atr if direction == 1 else (entry_price - close) / entry_atr, "time_stop", max_hold


# State
bal = 10000.0; pnl_d = 0.0; ld = None; sb = 10000.0
h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
lh = None; h1_atr = 0.0; pos = 0; ab = []
entry_regime = ""; entry_conf = 0.0

# Data collection
sequences = []   # list of (20, 17) numpy arrays
labels = []      # 0 or 1
directions = []  # +1 or -1 (H1 signal direction)
pnl_rs = []      # actual PnL R-multiple
regimes = []     # H1 regime at signal time
hours = []       # UTC hour

total_checked = 0
total_positive = 0
total_negative = 0

for i in range(max(config.seq_len_m15, 20), len(m15f) - 18):
    ts = m15f["timestamp"].iloc[i]; price = m15f["close"].iloc[i]

    if i % 50000 == 0 and i > 0:
        print(f"  bar {i}/{len(m15f)} ({i/len(m15f)*100:.0f}%) — collected {len(labels)} labels "
              f"({total_positive} pos, {total_negative} neg)")

    today = ts.date()
    if ld and today != ld: pnl_d = 0.0; sb = bal
    ld = today

    h1s = h1f[h1f["timestamp"] <= ts]
    m15s_window = m15f.iloc[max(0, i - config.seq_len_m15 * 2):i + 1]
    if len(h1s) < config.seq_len_h1: continue

    hl = h1s["timestamp"].max()
    if hl != lh:
        lh = hl; h1_feats = engine.compute(h1s)
        seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
        t = torch.from_numpy(seq).unsqueeze(0).to(device)
        for _, row in h1s.iloc[-14:].iterrows(): rd.update(row["high"], row["low"], row["close"])
        rr = classify_regime(encoder, classifier, t, rd, model_confidence_threshold=config.min_regime_confidence)
        g = gate.evaluate(rr["regime"], rr["confidence"], rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])
        if g.entry_signal:
            h1_closes = h1s["close"].values
            if len(h1_closes) >= 23:
                h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                with_trend = ((g.direction == 1 and h1_slope > 0) or (g.direction == -1 and h1_slope < 0))
                if not with_trend: h1_sig = None; listen = False; continue
            if ts.hour in BLOCKED_HOURS: h1_sig = None; listen = False; continue
            h1_sig = g.direction; listen = True; bl = 0
            h1_atr = h1_feats[-1, 6] * price
            entry_regime = rr["regime"]; entry_conf = g.confidence
        else:
            h1_sig = None; listen = False

    # Manage position (simple — we don't actually enter, just track if we would be in one)
    # For data collection, we skip bars where we'd be in a position
    if pos != 0:
        # Check if we'd be exited (simplified — just time-based)
        pos_bars += 1
        if pos_bars >= 18:
            pos = 0  # force close after max_hold
        continue

    if not listen: continue

    bl += 1
    if bl > config.max_listen_bars: listen = False; h1_sig = None; continue
    if ts.hour in BLOCKED_HOURS: continue

    # ── At this bar, simulate what would happen if we entered ──
    total_checked += 1

    # Get M15 feature sequence (20 bars ending at current bar)
    m15_feats = engine.compute(m15s_window)
    feat_seq = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)

    # Simulate trade outcome
    pnl_r, exit_reason, bars_held = simulate_trade_outcome(
        price, h1_sig, h1_atr, m15f, i + 1, max_bars=config.max_hold_bars)

    label = 1 if pnl_r > 0.25 else 0
    if label == 1: total_positive += 1
    else: total_negative += 1

    sequences.append(feat_seq.astype(np.float32))
    labels.append(label)
    directions.append(h1_sig)
    pnl_rs.append(pnl_r)
    regimes.append(entry_regime)
    hours.append(ts.hour)

# ═══════════════════════════════════════════════════════════════════
print(f"\nCollection complete: {len(labels)} samples")
print(f"  Positive (PnL > 0.25R): {total_positive} ({total_positive/len(labels)*100:.1f}%)")
print(f"  Negative (PnL <= 0.25R): {total_negative} ({total_negative/len(labels)*100:.1f}%)")
print(f"  Avg PnL of positives: {np.mean([pnl_rs[i] for i in range(len(pnl_rs)) if labels[i]==1]):+.3f}R")
print(f"  Avg PnL of negatives: {np.mean([pnl_rs[i] for i in range(len(pnl_rs)) if labels[i]==0]):+.3f}R")

# Save
out_path = os.path.join(config.data_dir, "m15_training_data_v2.pkl")
data = {
    "sequences": np.array(sequences, dtype=np.float32),  # (N, 20, 17)
    "labels": np.array(labels, dtype=np.int64),
    "directions": np.array(directions, dtype=np.int64),
    "pnl_rs": np.array(pnl_rs, dtype=np.float32),
    "regimes": regimes,
    "hours": np.array(hours, dtype=np.int64),
}
with open(out_path, "wb") as f:
    pickle.dump(data, f)
print(f"\nSaved to {out_path}")
print(f"Array shapes: sequences={data['sequences'].shape}, labels={data['labels'].shape}")
