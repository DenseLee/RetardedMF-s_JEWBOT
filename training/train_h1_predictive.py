"""
Train H1 encoder + classifier with FORWARD-LOOKING oracle labels.

Instead of mimicking the rule-based detector (reactionary), the model learns
to predict what the market will do in the next 18 hours:
  - LONG_WIN  (0): Long entry would have made >1.0R
  - SHORT_WIN (1): Short entry would have made >1.0R
  - BOTH_WIN  (2): Both directions would have made >1.0R
  - CHOP      (3): Neither direction made >1.0R

This is a predictive task: given the last 96 H1 bars, predict the opportunity
in the next 72 M15 bars.
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier

ORACLE_CLASSES = ["LONG_WIN", "SHORT_WIN", "BOTH_WIN", "CHOP"]


def compute_oracle_labels_from_csv(h1_df, m15_df, config):
    """Compute forward-looking oracle labels from CSV M15 data.

    For each H1 bar: look ahead max_hold M15 bars, measure best long/short
    excursion in ATR units using M15 bar high/low.

    Returns numpy array of class indices (0=LONG_WIN, 1=SHORT_WIN, 2=BOTH_WIN, 3=CHOP)
    """
    max_hold = 72  # M15 bars = 18 hours
    min_move = 1.0  # ATR minimum
    ratio = 1.5     # dominance ratio

    n_h1 = len(h1_df)
    n_m15 = len(m15_df)
    labels = np.full(n_h1, -1, dtype=np.int64)

    # Approximate ATR per H1 bar (from H1 features)
    from data.feature_engine_btc import BTCFeatureEngine
    engine = BTCFeatureEngine()
    h1_feats = engine.compute(h1_df)
    h1_atr_pct = h1_feats[:, 6]  # ATR as % of close

    for h1_i in range(config.seq_len_h1, n_h1):
        h1_ts = h1_df['timestamp'].iloc[h1_i]
        h1_close = float(h1_df['close'].iloc[h1_i])
        atr = h1_atr_pct[h1_i] * h1_close
        if atr < 1:
            atr = h1_close * 0.005

        # Find M15 bars in the lookahead window
        start_m15 = int((m15_df['timestamp'] > h1_ts).sum())
        if start_m15 >= n_m15:
            continue
        end_m15 = min(start_m15 + max_hold, n_m15)

        best_long = 0.0
        best_short = 0.0
        for m15_j in range(start_m15, end_m15):
            hi = float(m15_df['high'].iloc[m15_j])
            lo = float(m15_df['low'].iloc[m15_j])
            best_long = max(best_long, (hi - h1_close) / atr)
            best_short = max(best_short, (h1_close - lo) / atr)

        # Classify
        if best_long < min_move and best_short < min_move:
            labels[h1_i] = 3  # CHOP
        elif best_long > best_short * ratio and best_long >= min_move:
            labels[h1_i] = 0  # LONG_WIN
        elif best_short > best_long * ratio and best_short >= min_move:
            labels[h1_i] = 1  # SHORT_WIN
        else:
            labels[h1_i] = 2  # BOTH_WIN

    return labels


def build_data(config, engine):
    """Load data and compute oracle labels."""
    h1_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    m15_path = os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
    m15 = pd.read_csv(m15_path); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)

    # Split
    train_mask = (h1["timestamp"] >= config.h1_train_start) & (h1["timestamp"] < config.h1_train_end)
    val_mask   = (h1["timestamp"] >= config.val_start) & (h1["timestamp"] < config.val_end)
    h1_train = h1[train_mask].reset_index(drop=True)
    h1_val   = h1[val_mask].reset_index(drop=True)

    print(f"Train H1: {len(h1_train)} bars ({config.h1_train_start} → {config.h1_train_end})")
    print(f"Val H1:   {len(h1_val)} bars ({config.val_start} → {config.val_end})")

    # Build forward-looking oracle labels from CSV data
    print("Computing forward-looking oracle labels from M15 CSV data...")
    t0 = time.time()
    y_train = compute_oracle_labels_from_csv(h1_train, m15, config)
    y_val   = compute_oracle_labels_from_csv(h1_val, m15, config)
    print(f"  Done in {(time.time()-t0):.1f}s")

    n_train = (y_train >= 0).sum()
    n_val   = (y_val >= 0).sum()
    print(f"Train labeled: {n_train} ({n_train/len(h1_train)*100:.1f}%)")
    print(f"Val labeled:   {n_val} ({n_val/len(h1_val)*100:.1f}%)")

    for c, name in enumerate(ORACLE_CLASSES):
        n = (y_train == c).sum()
        print(f"  {name}: {n} bars ({n/n_train*100:.1f}%)" if n_train > 0 else f"  {name}: 0")

    # Features
    feats_train = engine.compute(h1_train)
    feats_val   = engine.compute(h1_val)

    def make_loader(feats, y, seq_len, batch, shuffle=True):
        X, Y = [], []
        for i in range(seq_len, len(feats)):
            if y[i] < 0: continue
            seq = engine.compute_sequence(feats, i, seq_len)
            X.append(seq); Y.append(y[i])
        X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True)

    train_loader = make_loader(feats_train, y_train, config.seq_len_h1, config.batch_size)
    val_loader   = make_loader(feats_val, y_val, config.seq_len_h1, config.batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train_epoch(encoder, classifier, loader, opt, crit, freeze_enc, device):
    encoder.train(); classifier.train()
    loss_sum, correct, total = 0.0, 0, 0
    params = list(classifier.parameters())
    if not freeze_enc: params += list(encoder.parameters())
    for X, y in loader:
        X, y = X.to(device), y.to(device); opt.zero_grad()
        if freeze_enc:
            with torch.no_grad(): emb = encoder(X)["embedding"]
        else:
            emb = encoder(X)["embedding"]
        logits = classifier.raw_logits(emb)
        loss_b = crit(logits, y); loss_b.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        loss_sum += loss_b.item()
        correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/len(loader), correct/total*100


@torch.no_grad()
def validate(encoder, classifier, loader, crit, device):
    encoder.eval(); classifier.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        emb = encoder(X)["embedding"]
        logits = classifier.raw_logits(emb)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/len(loader), correct/total*100


def run_stage(encoder, classifier, train_ldr, val_ldr, opt, crit,
              freeze_enc, device, n_epochs, print_freq=10,
              early_stop_patience=20, min_epochs=0):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=8, min_lr=1e-7)
    best_va, best_enc_sd, best_cls_sd, no_imp = 0.0, None, None, 0
    for ep in range(n_epochs):
        tl, ta = train_epoch(encoder, classifier, train_ldr, opt, crit, freeze_enc, device)
        vl, va = validate(encoder, classifier, val_ldr, crit, device)
        scheduler.step(va)
        if va > best_va:
            best_va = va
            best_enc_sd = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_cls_sd = {k: v.clone() for k, v in classifier.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if (ep+1) % print_freq == 0:
            print(f"  Ep {ep+1:4d}: train={ta:.1f}% val={va:.1f}% best={best_va:.1f}% "
                  f"lr={opt.param_groups[0]['lr']:.2e}")
        if ep >= min_epochs and no_imp >= early_stop_patience:
            print(f"  Early stop ep {ep+1}")
            break
    return best_enc_sd, best_cls_sd, best_va


def main():
    config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"); t_start = time.time()

    engine = BTCFeatureEngine()
    train_loader, val_loader = build_data(config, engine)

    encoder = CNNLSTMEncoder(
        n_features=17, seq_len=config.seq_len_h1, cnn_channels=config.cnn_channels,
        lstm_hidden=config.lstm_hidden, lstm_layers=config.lstm_layers,
        dropout=config.lstm_dropout, embedding_dim=config.embedding_dim,
        regime_classes=4, bidirectional=True).to(device)
    classifier = RegimeClassifier(embedding_dim=128, n_classes=4).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1)

    global_best_va = 0.0
    global_best_enc = None
    global_best_cls = None

    # Stage 1: unfrozen, higher LR
    S1 = 200
    print(f"\n=== STAGE 1: {S1} ep, unfrozen, LR=1e-3, min_epochs=50 ===")
    for p in encoder.parameters(): p.requires_grad = True
    opt = optim.AdamW(list(encoder.parameters())+list(classifier.parameters()),
                      lr=1e-3, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S1, print_freq=10,
                            early_stop_patience=30, min_epochs=50)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    # Stage 2: lower LR
    S2 = 100
    print(f"\n=== STAGE 2: {S2} ep, unfrozen, LR=1e-4 ===")
    opt = optim.AdamW(list(encoder.parameters())+list(classifier.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S2, print_freq=10,
                            early_stop_patience=25, min_epochs=20)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    # Stage 3: fine-tune
    S3 = 50
    print(f"\n=== STAGE 3: {S3} ep, unfrozen, LR=1e-5 ===")
    opt = optim.AdamW(list(encoder.parameters())+list(classifier.parameters()),
                      lr=1e-5, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S3, print_freq=10,
                            early_stop_patience=20)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    elapsed = (time.time() - t_start) / 60
    print(f"\nBest val acc: {global_best_va:.1f}%  |  Time: {elapsed:.1f} min")

    save_path = os.path.join(config.model_dir, "btc_h1_predictive.pt")
    torch.save({"encoder_state_dict": global_best_enc,
                "classifier_state_dict": global_best_cls,
                "oracle_classes": ORACLE_CLASSES,
                "config": config, "val_acc": global_best_va,
                "note": "Predictive model trained on oracle forward-looking labels"}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
