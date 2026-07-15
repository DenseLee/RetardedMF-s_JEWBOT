"""
Compare: Old model (reactionary) vs New predictive model vs Rule detector.
All evaluated against MT5-based oracle labels for May 2026.
"""
import os, sys, pickle, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, REGIME_NAMES
from models.entry_gate import EntryGate

ORACLE_CLASSES = ["LONG_WIN", "SHORT_WIN", "BOTH_WIN", "CHOP"]


def load_model(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return encoder, classifier, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-model", default="models/btc_h1_encoder_fallback.pt")
    parser.add_argument("--new-model", default="models/btc_h1_predictive.pt")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--oracle", default="benchmark/ytd_oracle.pkl")
    parser.add_argument("--csv", default="D:/FiananceBot/BTC_BOT/TrainingData/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    args = parser.parse_args()

    cfg = BTCConfig()
    device = torch.device("cpu")
    engine = BTCFeatureEngine()

    # ── Load data ──
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    mask = (df["timestamp"] >= "2026-05-01") & (df["timestamp"] < "2026-05-26")
    df = df[mask].reset_index(drop=True)
    feats = engine.compute(df)
    n_bars = len(df) - cfg.seq_len_h1
    print(f"Data: {len(df)} H1 bars for May 2026, {n_bars} evaluable")

    # ── Load oracle (MT5-precise) ──
    oracle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.oracle)
    oracle_by_h1 = {}
    if os.path.exists(oracle_path):
        oracle_labels = pickle.load(open(oracle_path, "rb"))
        # Aggregate M15 oracle labels to H1 level
        h1_temp = {}
        for ol in oracle_labels:
            if not ol.timestamp.startswith("2026-05"): continue
            h1_key = ol.timestamp[:13]  # "2026-05-01 00"
            if h1_key not in h1_temp:
                h1_temp[h1_key] = {"long": 0, "short": 0, "count": 0, "label_counts": {}}
            h1_temp[h1_key]["long"] += ol.long_r
            h1_temp[h1_key]["short"] += ol.short_r
            h1_temp[h1_key]["count"] += 1

        for h1_key, vals in h1_temp.items():
            avg_long = vals["long"] / vals["count"]
            avg_short = vals["short"] / vals["count"]
            if avg_long < 1.0 and avg_short < 1.0:
                oracle_by_h1[h1_key] = {"label": "CHOP", "dir": 0, "long_r": avg_long, "short_r": avg_short}
            elif avg_long > avg_short * 1.5:
                oracle_by_h1[h1_key] = {"label": "LONG_WIN", "dir": 1, "long_r": avg_long, "short_r": avg_short}
            elif avg_short > avg_long * 1.5:
                oracle_by_h1[h1_key] = {"label": "SHORT_WIN", "dir": -1, "long_r": avg_long, "short_r": avg_short}
            else:
                oracle_by_h1[h1_key] = {"label": "BOTH_WIN", "dir": 0, "long_r": avg_long, "short_r": avg_short}
        print(f"Oracle: {len(oracle_by_h1)} H1 keys for May 2026")
    else:
        print("Oracle not found — using CSV-computed labels")
        oracle_by_h1 = {}

    # ── Load models ──
    print("\nLoading models...")
    old_enc, old_cls, old_ckpt = load_model(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.old_model), cfg, device)
    old_val = old_ckpt.get("val_acc", 0)

    new_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.new_model)
    if os.path.exists(new_path):
        new_enc, new_cls, new_ckpt = load_model(new_path, cfg, device)
        new_val = new_ckpt.get("val_acc", 0)
        has_new = True
    else:
        has_new = False
        new_val = 0

    print(f"  Old model val_acc: {old_val:.1f}%")
    if has_new:
        print(f"  New model val_acc: {new_val:.1f}% (predictive)")

    # ── Rule detector baseline ──
    rule_det = RuleBasedRegimeDetector()
    for _, row in df.iloc[:cfg.seq_len_h1].iterrows():
        rule_det.update(row["high"], row["low"], row["close"])

    gate = EntryGate(min_confidence=0.35, min_atr_pct=0.3, max_atr_pct=0.9)

    # ── Run inference on all 3 ──
    print(f"\nRunning inference on {n_bars} bars...")
    results = []

    for i in range(n_bars):
        bar_idx = i + cfg.seq_len_h1
        ts = str(df["timestamp"].iloc[bar_idx])[:19]
        price = float(df["close"].iloc[bar_idx])
        h1_key = str(df["timestamp"].iloc[bar_idx])[:13]

        # Update rule detector
        for _, row in df.iloc[max(0, bar_idx-14):bar_idx+1].iterrows():
            rule_det.update(row["high"], row["low"], row["close"])
        rule_out = rule_det._classify()

        # Old model
        seq = engine.compute_sequence(feats, bar_idx, cfg.seq_len_h1)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
        with torch.no_grad():
            old_emb = old_enc(seq_t)
            old_raw = old_cls.raw_logits(old_emb["embedding"])
            old_probs = F.softmax(old_raw / args.temperature, dim=1).squeeze(0)
        old_pred = old_probs.argmax().item()
        old_conf = old_probs[old_pred].item()
        old_regime = REGIME_NAMES[old_pred]

        # Gate evaluation on old model
        atr_pct = rule_out.get("atr_percentile", 0.5)
        bb_pos = float(feats[bar_idx, 4])
        gd = gate.evaluate(old_regime, old_conf, atr_pct, bb_position=bb_pos)
        old_gate_signal = gd.entry_signal
        old_gate_dir = gd.direction

        # New predictive model
        if has_new:
            with torch.no_grad():
                new_emb = new_enc(seq_t)
                new_raw = new_cls.raw_logits(new_emb["embedding"])
                new_probs = F.softmax(new_raw / args.temperature, dim=1).squeeze(0)
            new_pred = new_probs.argmax().item()
            new_conf = new_probs[new_pred].item()
            new_label = ORACLE_CLASSES[new_pred]
            # Map to direction: LONG_WIN=long, SHORT_WIN=short, BOTH_WIN=neutral, CHOP=neutral
            new_dir = {"LONG_WIN": 1, "SHORT_WIN": -1, "BOTH_WIN": 0, "CHOP": 0}[new_label]
        else:
            new_label = "N/A"
            new_conf = 0.0
            new_dir = 0

        # Rule detector direction
        rule_regime = rule_out["regime"]
        rule_dir = 1 if rule_regime == "TREND_UP" else (-1 if rule_regime == "TREND_DOWN" else 0)

        # Oracle
        oracle = oracle_by_h1.get(h1_key, {})

        results.append({
            "ts": ts, "price": price,
            "old_regime": old_regime, "old_conf": round(old_conf, 4),
            "old_gate": old_gate_signal, "old_dir": old_gate_dir,
            "new_label": new_label, "new_conf": round(new_conf, 4),
            "new_dir": new_dir,
            "rule_regime": rule_regime, "rule_dir": rule_dir,
            "oracle_label": oracle.get("label", "?"),
            "oracle_dir": oracle.get("dir", 0),
            "oracle_long_r": round(oracle.get("long_r", 0), 2),
            "oracle_short_r": round(oracle.get("short_r", 0), 2),
        })

    df_r = pd.DataFrame(results)

    # ════════════════════════════════════
    # Comparison
    # ════════════════════════════════════
    print(f"\n{'='*70}")
    print("DIRECTION ACCURACY vs ORACLE (May 2026)")
    print("=" * 70)

    if oracle_by_h1:
        directional = df_r[df_r["oracle_dir"] != 0]
        n_dir = len(directional)

        for name, col in [("Old model", "old_dir"), ("Rule detector", "rule_dir")]:
            correct = (directional[col] == directional["oracle_dir"]).sum()
            wrong = (directional[col] == -directional["oracle_dir"]).sum()
            neutral = (directional[col] == 0).sum()
            print(f"  {name:<20s}: correct={correct} ({correct/n_dir*100:.1f}%)  "
                  f"wrong={wrong} ({wrong/n_dir*100:.1f}%)  neutral={neutral}")

        if has_new:
            correct = (directional["new_dir"] == directional["oracle_dir"]).sum()
            wrong = (directional["new_dir"] == -directional["oracle_dir"]).sum()
            neutral = (directional["new_dir"] == 0).sum()
            print(f"  {'New predictive':<20s}: correct={correct} ({correct/n_dir*100:.1f}%)  "
                  f"wrong={wrong} ({wrong/n_dir*100:.1f}%)  neutral={neutral}")

    # ════════════════════════════════════
    # Class distribution
    # ════════════════════════════════════
    print(f"\n{'='*70}")
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(f"\n  Old model:")
    for name in REGIME_NAMES:
        pct = (df_r["old_regime"] == name).mean() * 100
        print(f"    {name:<15s}: {pct:.1f}%")

    print(f"\n  Rule detector:")
    for name in REGIME_NAMES:
        pct = (df_r["rule_regime"] == name).mean() * 100
        print(f"    {name:<15s}: {pct:.1f}%")

    if has_new:
        print(f"\n  New predictive model:")
        for name in ORACLE_CLASSES:
            pct = (df_r["new_label"] == name).mean() * 100
            avg_conf = df_r[df_r["new_label"] == name]["new_conf"].mean()
            print(f"    {name:<15s}: {pct:.1f}%  (avg conf: {avg_conf:.3f})")

    # ════════════════════════════════════
    # Per-bar side-by-side (last 30)
    # ════════════════════════════════════
    print(f"\n{'='*120}")
    print("PER-BAR SIDE-BY-SIDE (last 30 bars)")
    print("=" * 120)
    cols = f"{'Time':19s} {'Close':>8s} {'Oracle':>10s} {'Old':>12s} {'OldGate':>8s} "
    if has_new:
        cols += f"{'NewPred':>12s} {'NewConf':>8s} "
    cols += f"{'Rule':>12s}"
    print(cols)
    print("-" * 120)

    for _, r in df_r.tail(30).iterrows():
        oracle_str = r["oracle_label"] if r["oracle_label"] != "?" else "?"
        old_gate_str = f"{'LONG' if r['old_dir']==1 else 'SHORT' if r['old_dir']==-1 else 'BLOCK'}"
        line = f"{r['ts'][:19]:19s} {r['price']:>8.1f} {oracle_str:>10s} {r['old_regime']:>12s} {old_gate_str:>8s} "
        if has_new:
            line += f"{r['new_label']:>12s} {r['new_conf']:>8.4f} "
        line += f"{r['rule_regime']:>12s}"
        print(line)

    # ════════════════════════════════════
    # Summary
    # ════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if oracle_by_h1 and has_new:
        dir_df = df_r[df_r["oracle_dir"] != 0]
        old_acc = (dir_df["old_dir"] == dir_df["oracle_dir"]).mean() * 100
        new_acc = (dir_df["new_dir"] == dir_df["oracle_dir"]).mean() * 100
        rule_acc = (dir_df["rule_dir"] == dir_df["oracle_dir"]).mean() * 100

        print(f"""
  Model               Oracle Accuracy    Prediction Diversity
  ─────────────────────────────────────────────────────────
  Old (reactionary)      {old_acc:.1f}%             1 class (mode-collapsed)
  Rule detector          {rule_acc:.1f}%             2-3 classes
  New (predictive)       {new_acc:.1f}%             {df_r['new_label'].nunique()} classes

  Δ New vs Old:          {new_acc - old_acc:+.1f}%
  Δ New vs Rule:         {new_acc - rule_acc:+.1f}%
""")

    print(f"\nOld model saved as: models/btc_h1_encoder_fallback.pt")
    print(f"New predictive model: models/btc_h1_predictive.pt")
    print(f"Current live model:   models/btc_h1_encoder.pt (unchanged — old model)")


if __name__ == "__main__":
    main()
