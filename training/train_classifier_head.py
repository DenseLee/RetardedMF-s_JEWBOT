"""
Retrain JUST the classifier head using the old (fallback) encoder, frozen.
The old encoder learned useful features but the classifier mode-collapsed.
Label smoothing on a new classifier should fix the overconfidence while
keeping the encoder's representations.
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, REGIME_NAMES, RuleBasedRegimeDetector


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05, ignore_index=-1):
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
    data_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    df = pd.read_csv(data_path); df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    train_mask = (df["timestamp"] >= config.h1_train_start) & (df["timestamp"] < config.h1_train_end)
    val_mask   = (df["timestamp"] >= config.val_start) & (df["timestamp"] < config.val_end)
    df_train = df[train_mask].reset_index(drop=True)
    df_val   = df[val_mask].reset_index(drop=True)
    print(f"Train: {len(df_train)} bars  Val: {len(df_val)} bars")

    feats_train = engine.compute(df_train); feats_val = engine.compute(df_val)

    print("Generating rule-based regime labels...")
    y_train = _generate_regime_labels(df_train)
    y_val   = _generate_regime_labels(df_val)

    for c, name in enumerate(REGIME_NAMES):
        n = (y_train == c).sum()
        print(f"  {name}: {n} bars ({n/len(y_train)*100:.1f}%)")

    def make_loader(feats, y, seq_len, batch, use_sampler=True):
        X, Y = [], []
        for i in range(seq_len, len(feats)):
            if y[i] < 0: continue
            seq = engine.compute_sequence(feats, i, seq_len)
            X.append(seq); Y.append(y[i])
        X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        if use_sampler:
            class_counts = Counter(Y)
            weights = [1.0 / class_counts[c] for c in Y]
            sampler = WeightedRandomSampler(weights, num_samples=len(Y), replacement=True)
            return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=True)
        else:
            return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    train_loader = make_loader(feats_train, y_train, config.seq_len_h1, config.batch_size)
    val_loader   = make_loader(feats_val, y_val, config.seq_len_h1, config.batch_size, use_sampler=False)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def _generate_regime_labels(df):
    rd = RuleBasedRegimeDetector(); n = len(df)
    labels = np.full(n, -1, dtype=np.int64)
    regime_map = {name: i for i, name in enumerate(REGIME_NAMES)}
    for i in range(n):
        row = df.iloc[i]
        result = rd.update(row["high"], row["low"], row["close"])
        labels[i] = regime_map.get(result["regime"], -1)
    return labels


def train_epoch(encoder, classifier, loader, opt, crit, device):
    encoder.train(); classifier.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device); opt.zero_grad()
        with torch.no_grad(): emb = encoder(X)["embedding"]
        logits = classifier.raw_logits(emb)
        loss_b = crit(logits, y); loss_b.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        opt.step()
        loss_sum += loss_b.item()
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / len(loader), correct / total * 100


@torch.no_grad()
def validate(encoder, classifier, loader, crit, device):
    encoder.eval(); classifier.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        emb = encoder(X)["embedding"]
        logits = classifier.raw_logits(emb)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / len(loader), correct / total * 100


def main():
    config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"); t_start = time.time()

    engine = BTCFeatureEngine()
    train_loader, val_loader = build_data(config, engine)

    # Load OLD encoder (frozen — keep its learned features)
    old_ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder_fallback.pt"),
                          map_location=device, weights_only=False)
    encoder = CNNLSTMEncoder(
        n_features=17, seq_len=config.seq_len_h1, cnn_channels=config.cnn_channels,
        lstm_hidden=config.lstm_hidden, lstm_layers=config.lstm_layers,
        dropout=config.lstm_dropout, embedding_dim=config.embedding_dim,
        regime_classes=4, bidirectional=True).to(device).eval()
    encoder.load_state_dict(old_ckpt["encoder_state_dict"])
    for p in encoder.parameters(): p.requires_grad = False
    print("Loaded old encoder (frozen)")

    # NEW classifier (random init)
    classifier = RegimeClassifier(embedding_dim=128, n_classes=4).to(device)
    crit = LabelSmoothingCrossEntropy(smoothing=0.05, ignore_index=-1)

    # Stage 1: train classifier head
    S1 = 50
    print(f"\n=== STAGE 1: {S1} ep, lr=1e-3 ===")
    opt = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-7)
    best_va, best_sd, no_imp = 0.0, None, 0
    for ep in range(S1):
        tl, ta = train_epoch(encoder, classifier, train_loader, opt, crit, device)
        vl, va = validate(encoder, classifier, val_loader, crit, device)
        scheduler.step(va)
        if va > best_va: best_va = va; best_sd = {k: v.clone() for k, v in classifier.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1:4d}: train={ta:.1f}% val={va:.1f}% best={best_va:.1f}% lr={opt.param_groups[0]['lr']:.2e}")
        if no_imp >= 10: print(f"  Early stop ep {ep+1}"); break

    # Stage 2: fine-tune with lower LR
    S2 = 30
    print(f"\n=== STAGE 2: {S2} ep, lr=1e-4 ===")
    classifier.load_state_dict(best_sd)
    opt = optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-7)
    no_imp = 0
    for ep in range(S2):
        tl, ta = train_epoch(encoder, classifier, train_loader, opt, crit, device)
        vl, va = validate(encoder, classifier, val_loader, crit, device)
        scheduler.step(va)
        if va > best_va: best_va = va; best_sd = {k: v.clone() for k, v in classifier.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1:4d}: train={ta:.1f}% val={va:.1f}% best={best_va:.1f}% lr={opt.param_groups[0]['lr']:.2e}")
        if no_imp >= 10: print(f"  Early stop ep {ep+1}"); break

    elapsed = (time.time() - t_start) / 60
    print(f"\nBest val acc: {best_va:.1f}%  |  Time: {elapsed:.1f} min")

    # Save: old encoder + new classifier
    classifier.load_state_dict(best_sd)
    save_path = os.path.join(config.model_dir, "btc_h1_encoder.pt")
    torch.save({"encoder_state_dict": old_ckpt["encoder_state_dict"],
                "classifier_state_dict": best_sd,
                "config": config, "val_acc": best_va,
                "note": "Old encoder (frozen) + new classifier with label smoothing 0.05"}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
