"""
Compare old (fallback) vs new H1 encoder+classifier models.
Metrics: class distribution, confidence calibration, oracle alignment, per-bar side-by-side.
"""
import os, sys, pickle, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, REGIME_NAMES


def load_model(ckpt_path, cfg, device):
    """Load encoder + classifier from checkpoint."""
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
    val_acc = ckpt.get("val_acc", 0.0)
    return encoder, classifier, val_acc


def run_inference(encoder, classifier, feats, seq_len, temperature, device, engine):
    """Run inference over all bars, return per-bar predictions."""
    results = []
    for i in range(seq_len, len(feats)):
        seq = engine.compute_sequence(feats, i, seq_len)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
        with torch.no_grad():
            enc = encoder(seq_t)
            raw = classifier.raw_logits(enc["embedding"])
            probs = F.softmax(raw / temperature, dim=1).squeeze(0)
        pred = probs.argmax().item()
        conf = probs[pred].item()
        results.append({
            "regime": REGIME_NAMES[pred],
            "confidence": round(conf, 4),
            "probs": [round(p, 4) for p in probs.tolist()],
        })
    return results


def compute_ece(predictions, oracle_labels_by_idx, seq_len):
    """Expected Calibration Error."""
    confidences = []
    correct = []
    for i, pred in enumerate(predictions):
        bar_idx = i + seq_len
        oracle_dir = oracle_labels_by_idx.get(bar_idx, 0)
        model_dir = 1 if pred["regime"] == "TREND_UP" else (-1 if pred["regime"] == "TREND_DOWN" else 0)
        confidences.append(pred["confidence"])
        correct.append(1 if model_dir == oracle_dir and oracle_dir != 0 else 0)

    confidences = np.array(confidences)
    correct = np.array(correct)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() == 0: continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece, float(confidences.mean()), float(correct.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-model", default="models/btc_h1_encoder_fallback.pt")
    parser.add_argument("--new-model", default="models/btc_h1_encoder.pt")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--oracle", default="benchmark/ytd_oracle.pkl")
    parser.add_argument("--from", dest="start_date", default="2026-05-01")
    parser.add_argument("--to", dest="end_date", default="2026-05-26")
    args = parser.parse_args()

    cfg = BTCConfig()
    device = torch.device("cpu")
    engine = BTCFeatureEngine()

    # ── Load data ──
    data_path = os.path.join(cfg.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    mask = (df["timestamp"] >= args.start_date) & (df["timestamp"] < args.end_date)
    df = df[mask].reset_index(drop=True)
    feats = engine.compute(df)
    print(f"Data: {len(df)} bars ({args.start_date} → {args.end_date})")

    # ── Load oracle ──
    oracle_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.oracle)
    oracle_labels_by_idx = {}
    if os.path.exists(oracle_path):
        oracle_labels = pickle.load(open(oracle_path, "rb"))
        h1_oracle = {}
        for ol in oracle_labels:
            h1_key = ol.timestamp[:13]
            if h1_key not in h1_oracle:
                h1_oracle[h1_key] = {"long": 0, "short": 0, "count": 0}
            h1_oracle[h1_key]["long"] += ol.long_r
            h1_oracle[h1_key]["short"] += ol.short_r
            h1_oracle[h1_key]["count"] += 1

        for i, row in df.iterrows():
            h1_key = str(row["timestamp"])[:13]
            ho = h1_oracle.get(h1_key)
            if ho and ho["count"] > 0:
                avg_long = ho["long"] / ho["count"]
                avg_short = ho["short"] / ho["count"]
                if avg_long > avg_short * 1.5 and avg_long > 1.0:
                    oracle_labels_by_idx[i] = 1   # LONG
                elif avg_short > avg_long * 1.5 and avg_short > 1.0:
                    oracle_labels_by_idx[i] = -1  # SHORT
                else:
                    oracle_labels_by_idx[i] = 0   # neutral/chop
        print(f"Oracle: {len(h1_oracle)} H1 keys loaded, {len(oracle_labels_by_idx)} bars labeled")
    else:
        print("Oracle not found — skipping oracle metrics")

    # ── Load models ──
    print("\nLoading models...")
    old_enc, old_cls, old_acc = load_model(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.old_model),
        cfg, device)
    new_enc, new_cls, new_acc = load_model(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.new_model),
        cfg, device)
    print(f"  Old model val_acc: {old_acc:.1f}%")
    print(f"  New model val_acc: {new_acc:.1f}%")

    # ── Run inference ──
    print(f"\nRunning inference on {len(df)-cfg.seq_len_h1} bars...")
    old_preds = run_inference(old_enc, old_cls, feats, cfg.seq_len_h1, args.temperature, device, engine)
    new_preds = run_inference(new_enc, new_cls, feats, cfg.seq_len_h1, args.temperature, device, engine)
    n = len(old_preds)

    # ═══════════════════════════════════════
    # Metric 1: Class Distribution
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"  {'Regime':<15s} {'Old %':>8s} {'Old Conf':>10s} {'New %':>8s} {'New Conf':>10s}")
    print(f"  {'-'*55}")

    from scipy.stats import entropy as scipy_entropy
    old_dist = pd.Series([p["regime"] for p in old_preds]).value_counts(normalize=True)
    new_dist = pd.Series([p["regime"] for p in new_preds]).value_counts(normalize=True)

    for name in REGIME_NAMES:
        old_pct = old_dist.get(name, 0) * 100
        new_pct = new_dist.get(name, 0) * 100
        old_idx = [i for i, p in enumerate(old_preds) if p["regime"] == name]
        new_idx = [i for i, p in enumerate(new_preds) if p["regime"] == name]
        old_avg_conf = np.mean([old_preds[i]["confidence"] for i in old_idx]) if old_idx else 0
        new_avg_conf = np.mean([new_preds[i]["confidence"] for i in new_idx]) if new_idx else 0
        print(f"  {name:<15s} {old_pct:>7.1f}% {old_avg_conf:>9.4f}  {new_pct:>7.1f}% {new_avg_conf:>9.4f}")

    old_entropy = scipy_entropy(old_dist.values)
    new_entropy = scipy_entropy(new_dist.values)
    print(f"\n  Distribution entropy: Old={old_entropy:.3f}  New={new_entropy:.3f}  (higher=more diverse)")

    # ═══════════════════════════════════════
    # Metric 2: Confidence Calibration
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("CONFIDENCE CALIBRATION")
    print("=" * 70)

    if oracle_labels_by_idx:
        old_ece, old_avg_conf, old_avg_acc = compute_ece(old_preds, oracle_labels_by_idx, cfg.seq_len_h1)
        new_ece, new_avg_conf, new_avg_acc = compute_ece(new_preds, oracle_labels_by_idx, cfg.seq_len_h1)
        print(f"  Old: ECE={old_ece:.4f}  avg_conf={old_avg_conf:.3f}  oracle_acc={old_avg_acc:.3f}")
        print(f"  New: ECE={new_ece:.4f}  avg_conf={new_avg_conf:.3f}  oracle_acc={new_avg_acc:.3f}")
        print(f"  Δ:   ECE={new_ece - old_ece:+.4f}  acc={new_avg_acc - old_avg_acc:+.3f}")
        if new_ece < old_ece:
            print(f"  → New model is better calibrated (lower ECE)")
        else:
            print(f"  → Old model has lower ECE (but may be due to always predicting one class)")
    else:
        old_conf_vals = [p["confidence"] for p in old_preds]
        new_conf_vals = [p["confidence"] for p in new_preds]
        print(f"  Old: avg_conf={np.mean(old_conf_vals):.3f}  std={np.std(old_conf_vals):.3f}")
        print(f"  New: avg_conf={np.mean(new_conf_vals):.3f}  std={np.std(new_conf_vals):.3f}")

    # ═══════════════════════════════════════
    # Metric 3: Oracle Alignment
    # ═══════════════════════════════════════
    if oracle_labels_by_idx:
        print(f"\n{'='*70}")
        print("ORACLE ALIGNMENT")
        print("=" * 70)

        for label, preds in [("Old", old_preds), ("New", new_preds)]:
            correct_d = 0; wrong_d = 0; neutral = 0
            total_directional = 0
            for i, p in enumerate(preds):
                bar_idx = i + cfg.seq_len_h1
                oracle_dir = oracle_labels_by_idx.get(bar_idx, 0)
                if oracle_dir == 0: continue
                total_directional += 1
                model_dir = 1 if p["regime"] == "TREND_UP" else (-1 if p["regime"] == "TREND_DOWN" else 0)
                if model_dir == oracle_dir: correct_d += 1
                elif model_dir == -oracle_dir: wrong_d += 1
                else: neutral += 1

            if total_directional > 0:
                print(f"  {label}: correct={correct_d} ({correct_d/total_directional*100:.1f}%)  "
                      f"wrong={wrong_d} ({wrong_d/total_directional*100:.1f}%)  "
                      f"neutral={neutral}  (n={total_directional})")

    # ═══════════════════════════════════════
    # Metric 4: Per-bar side-by-side (last 40 bars)
    # ═══════════════════════════════════════
    print(f"\n{'='*140}")
    print(f"PER-BAR SIDE-BY-SIDE (last 40 bars)")
    print("=" * 140)
    hdr = f"{'Time':22s} {'Close':>8s} {'Old Regime':>14s} {'Old Conf':>8s} {'New Regime':>14s} {'New Conf':>8s}  {'Old Raw Logits':>35s}  {'New Raw Logits':>35s}"
    print(hdr)
    print("-" * 140)

    start_i = max(0, n - 40)
    for i in range(start_i, n):
        ts = str(df["timestamp"].iloc[i + cfg.seq_len_h1])[:19]
        close = float(df["close"].iloc[i + cfg.seq_len_h1])
        op = old_preds[i]; np_ = new_preds[i]

        seq = engine.compute_sequence(feats, i + cfg.seq_len_h1, cfg.seq_len_h1)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
        with torch.no_grad():
            old_raw = old_cls.raw_logits(old_enc(seq_t)["embedding"]).squeeze(0).tolist()
            new_raw = new_cls.raw_logits(new_enc(seq_t)["embedding"]).squeeze(0).tolist()
        old_logit_str = f"TU={old_raw[0]:+.1f} TD={old_raw[1]:+.1f} R={old_raw[2]:+.1f} TR={old_raw[3]:+.1f}"
        new_logit_str = f"TU={new_raw[0]:+.1f} TD={new_raw[1]:+.1f} R={new_raw[2]:+.1f} TR={new_raw[3]:+.1f}"

        print(f"{ts:22s} {close:>8.1f} {op['regime']:>14s} {op['confidence']:>8.4f} "
              f"{np_['regime']:>14s} {np_['confidence']:>8.4f}  {old_logit_str:>35s}  {new_logit_str:>35s}")

    # ═══════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    old_unique = len(set(p["regime"] for p in old_preds))
    new_unique = len(set(p["regime"] for p in new_preds))

    print(f"""
  {"Metric":<35s} {"Old":>12s} {"New":>12s} {"Δ":>12s}
  {"-"*70}
  {"Unique regimes predicted":<35s} {old_unique:>12d} {new_unique:>12d} {new_unique - old_unique:>+12d}
  {"Distribution entropy":<35s} {old_entropy:>12.3f} {new_entropy:>12.3f} {new_entropy - old_entropy:>+12.3f}""")

    if oracle_labels_by_idx:
        print(f"  {'ECE (calibration error)':<35s} {old_ece:>12.4f} {new_ece:>12.4f} {new_ece - old_ece:>+12.4f}")
        print(f"  {'Oracle direction accuracy':<35s} {old_avg_acc*100:>11.1f}% {new_avg_acc*100:>11.1f}% {(new_avg_acc - old_avg_acc)*100:>+11.1f}%")

    old_conf_mean = np.mean([p["confidence"] for p in old_preds])
    new_conf_mean = np.mean([p["confidence"] for p in new_preds])
    print(f"  {'Avg confidence':<35s} {old_conf_mean:>12.4f} {new_conf_mean:>12.4f} {new_conf_mean - old_conf_mean:>+12.4f}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", f"model_comparison_{args.start_date[:7]}.csv")
    rows = []
    for i in range(n):
        bar_idx = i + cfg.seq_len_h1
        rows.append({
            "timestamp": str(df["timestamp"].iloc[bar_idx])[:19],
            "close": float(df["close"].iloc[bar_idx]),
            "old_regime": old_preds[i]["regime"],
            "old_conf": old_preds[i]["confidence"],
            "old_prob_TU": old_preds[i]["probs"][0],
            "old_prob_TD": old_preds[i]["probs"][1],
            "old_prob_R": old_preds[i]["probs"][2],
            "old_prob_TR": old_preds[i]["probs"][3],
            "new_regime": new_preds[i]["regime"],
            "new_conf": new_preds[i]["confidence"],
            "new_prob_TU": new_preds[i]["probs"][0],
            "new_prob_TD": new_preds[i]["probs"][1],
            "new_prob_R": new_preds[i]["probs"][2],
            "new_prob_TR": new_preds[i]["probs"][3],
            "oracle_dir": oracle_labels_by_idx.get(bar_idx, 0) if oracle_labels_by_idx else 0,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved per-bar comparison to {csv_path}")


if __name__ == "__main__":
    main()
