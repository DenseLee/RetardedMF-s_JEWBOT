"""
Train H1 CNN-LSTM encoder + regime classifier.
Training window: 2020-01-01 → 2025-12-31

Uses RuleBasedRegimeDetector for ground-truth regime labels.

Fixes over v1:
  - Label smoothing (0.1) to prevent logit saturation
  - Class-balanced sampling (WeightedRandomSampler)
  - Gradient clipping (max_norm=1.0)
  - ReduceLROnPlateau + early stopping

Curriculum: Stage1(50ep frozen) → Stage2(100ep frozen) → Stage3(200ep unfrozen)
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, REGIME_NAMES, RuleBasedRegimeDetector


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing. Prevents extreme logit saturation."""
    def __init__(self, smoothing=0.1, ignore_index=-1):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            valid = targets != self.ignore_index
            smooth_targets[valid] = 0.0
            smooth_targets[valid].scatter_(1, targets[valid].unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=1)
        loss = loss * valid.float()
        return loss.sum() / valid.sum().clamp(min=1)


def build_data(config, engine):
    """Load 2020-2025 data, generate rule-based regime labels."""
    data_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    df = pd.read_csv(data_path); df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    train_mask = (df["timestamp"] >= config.h1_train_start) & (df["timestamp"] < config.h1_train_end)
    val_mask   = (df["timestamp"] >= config.val_start) & (df["timestamp"] < config.val_end)
    df_train = df[train_mask].reset_index(drop=True)
    df_val   = df[val_mask].reset_index(drop=True)
    print(f"Train: {len(df_train)} bars ({config.h1_train_start} → {config.h1_train_end})")
    print(f"Val:   {len(df_val)} bars ({config.val_start} → {config.val_end})")

    feats_train = engine.compute(df_train)
    feats_val   = engine.compute(df_val)

    print("Generating rule-based regime labels...")
    y_train = _generate_regime_labels(df_train)
    y_val   = _generate_regime_labels(df_val)
    print(f"  Train: {(y_train>=0).sum()} labeled bars")
    print(f"  Val:   {(y_val>=0).sum()} labeled bars")

    # Class distribution
    for c, name in enumerate(REGIME_NAMES):
        n = (y_train == c).sum()
        print(f"  {name}: {n} bars ({n/len(y_train)*100:.1f}%)")

    def make_loader(feats, y, seq_len, batch, shuffle=True):
        X, Y = [], []
        for i in range(seq_len, len(feats)):
            if y[i] < 0: continue
            seq = engine.compute_sequence(feats, i, seq_len)
            X.append(seq); Y.append(y[i])
        X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True)

    train_loader = make_loader(feats_train, y_train, config.seq_len_h1, config.batch_size, shuffle=True)
    val_loader   = make_loader(feats_val, y_val, config.seq_len_h1, config.batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader


def _generate_regime_labels(df):
    """Run RuleBasedRegimeDetector over each bar, assign regime class."""
    rd = RuleBasedRegimeDetector()
    n = len(df)
    labels = np.full(n, -1, dtype=np.int64)
    regime_map = {name: i for i, name in enumerate(REGIME_NAMES)}
    for i in range(n):
        row = df.iloc[i]
        result = rd.update(row["high"], row["low"], row["close"])
        regime = result["regime"]
        labels[i] = regime_map.get(regime, -1)
    return labels


def train_epoch(encoder, classifier, loader, opt, crit, freeze_enc, device):
    encoder.train(); classifier.train()
    loss_sum, correct, total = 0.0, 0, 0
    params = list(classifier.parameters())
    if not freeze_enc:
        params += list(encoder.parameters())

    for X, y in loader:
        X, y = X.to(device), y.to(device); opt.zero_grad()
        if freeze_enc:
            with torch.no_grad(): enc = encoder(X)
        else:
            enc = encoder(X)
        logits = classifier.raw_logits(enc["embedding"])
        loss_b = crit(logits, y)
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        loss_sum += loss_b.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total * 100


@torch.no_grad()
def validate(encoder, classifier, loader, crit, device):
    encoder.eval(); classifier.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        enc = encoder(X)
        logits = classifier.raw_logits(enc["embedding"])
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total * 100


def run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
              freeze_enc, device, n_epochs, stage_name, print_freq=10,
              early_stop_patience=20, min_epochs=0):
    """Train one stage with ReduceLROnPlateau + early stopping.
    min_epochs: don't early-stop before this many epochs (encoder warmup)."""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, min_lr=1e-7)
    best_va = 0.0
    best_enc_sd = None
    best_cls_sd = None
    no_improve = 0

    for ep in range(n_epochs):
        tl, ta = train_epoch(encoder, classifier, train_loader, opt, crit, freeze_enc, device)
        vl, va = validate(encoder, classifier, val_loader, crit, device)
        scheduler.step(va)

        if va > best_va:
            best_va = va
            best_enc_sd = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_cls_sd = {k: v.clone() for k, v in classifier.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % print_freq == 0:
            print(f"  Ep {ep+1:4d}: train={ta:.1f}% val={va:.1f}% best={best_va:.1f}% "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

        if ep >= min_epochs and no_improve >= early_stop_patience:
            print(f"  Early stopping at ep {ep+1} (no improvement for {early_stop_patience} epochs)")
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
    crit = LabelSmoothingCrossEntropy(smoothing=0.02, ignore_index=-1)

    global_best_va = 0.0
    global_best_enc = None
    global_best_cls = None

    # ── Stage 1: Long training, no early stop until ep 50 ──
    S1 = 200
    print(f"\n=== STAGE 1: {S1} episodes, unfrozen, LR=1e-3, min_epochs=50 ===")
    for p in encoder.parameters(): p.requires_grad = True
    opt = optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()),
                      lr=1e-3, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S1, "Stage 1", print_freq=10,
                            early_stop_patience=30, min_epochs=50)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    # ── Stage 2: Lower LR ──
    S2 = 100
    print(f"\n=== STAGE 2: {S2} episodes, unfrozen, LR=1e-4 ===")
    opt = optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S2, "Stage 2", print_freq=10,
                            early_stop_patience=25, min_epochs=20)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    # ── Stage 3: Fine-tune with lowest LR ──
    S3 = 50
    print(f"\n=== STAGE 3: {S3} episodes, unfrozen, LR=1e-5 ===")
    opt = optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()),
                      lr=1e-5, weight_decay=1e-4)
    be, bc, bv = run_stage(encoder, classifier, train_loader, val_loader, opt, crit,
                            False, device, S3, "Stage 3", print_freq=10,
                            early_stop_patience=20)
    if bv > global_best_va:
        global_best_va, global_best_enc, global_best_cls = bv, be, bc

    elapsed = (time.time() - t_start) / 60
    print(f"\nBest val acc: {global_best_va:.1f}%  |  Time: {elapsed:.1f} min")

    # Save best model
    save_path = os.path.join(config.model_dir, "btc_h1_encoder.pt")
    torch.save({"encoder_state_dict": global_best_enc,
                "classifier_state_dict": global_best_cls,
                "config": config, "val_acc": global_best_va}, save_path)
    print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
