"""
Train a SIMPLE predictive model — shallow 1D conv + small MLP, no LSTM.

The CNNLSTMEncoder mode-collapses regardless of training recipe.
A simpler model with fewer parameters should be easier to train and
less prone to mode collapse.
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine

ORACLE_CLASSES = ["LONG_WIN", "SHORT_WIN", "BOTH_WIN", "CHOP"]


class SimplePredictor(nn.Module):
    """Lightweight 1D conv + MLP for oracle direction prediction.
    Input: (B, seq_len=96, n_features=17)
    """
    def __init__(self, n_features=17, seq_len=96, n_classes=4, dropout=0.3):
        super().__init__()
        # Aggressive downsampling via stride
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2, stride=2),  # 96→48
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 5, padding=2, stride=2),          # 48→24
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 64, 5, padding=2, stride=2),          # 24→12
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),                              # → (B, 64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features) → (B, n_features, seq_len)
        x = x.permute(0, 2, 1)
        feats = self.conv(x)
        return self.classifier(feats)


def compute_oracle_labels_from_csv(h1_df, m15_df, config):
    """Compute forward-looking oracle labels from CSV M15 data (same as before)."""
    max_hold, min_move, ratio = 72, 1.0, 1.5
    n_h1, n_m15 = len(h1_df), len(m15_df)
    labels = np.full(n_h1, -1, dtype=np.int64)
    engine = BTCFeatureEngine()
    h1_feats = engine.compute(h1_df)
    h1_atr_pct = h1_feats[:, 6]

    for h1_i in range(config.seq_len_h1, n_h1):
        h1_ts = h1_df['timestamp'].iloc[h1_i]
        h1_close = float(h1_df['close'].iloc[h1_i])
        atr = h1_atr_pct[h1_i] * h1_close
        if atr < 1: atr = h1_close * 0.005

        start_m15 = int((m15_df['timestamp'] > h1_ts).sum())
        if start_m15 >= n_m15: continue
        end_m15 = min(start_m15 + max_hold, n_m15)

        best_long, best_short = 0.0, 0.0
        for m15_j in range(start_m15, end_m15):
            hi = float(m15_df['high'].iloc[m15_j])
            lo = float(m15_df['low'].iloc[m15_j])
            best_long = max(best_long, (hi - h1_close) / atr)
            best_short = max(best_short, (h1_close - lo) / atr)

        if best_long < min_move and best_short < min_move:
            labels[h1_i] = 3
        elif best_long > best_short * ratio and best_long >= min_move:
            labels[h1_i] = 0
        elif best_short > best_long * ratio and best_short >= min_move:
            labels[h1_i] = 1
        else:
            labels[h1_i] = 2
    return labels


def build_data(config, engine):
    h1_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    m15_path = os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
    m15 = pd.read_csv(m15_path); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)

    train_mask = (h1["timestamp"] >= config.h1_train_start) & (h1["timestamp"] < config.h1_train_end)
    val_mask   = (h1["timestamp"] >= config.val_start) & (h1["timestamp"] < config.val_end)
    h1_train = h1[train_mask].reset_index(drop=True)
    h1_val   = h1[val_mask].reset_index(drop=True)

    print(f"Train H1: {len(h1_train)}  Val H1: {len(h1_val)}")

    t0 = time.time()
    print("Computing oracle labels...")
    y_train = compute_oracle_labels_from_csv(h1_train, m15, config)
    y_val   = compute_oracle_labels_from_csv(h1_val, m15, config)
    print(f"  Done in {time.time()-t0:.1f}s")

    n_tr = (y_train >= 0).sum(); n_vl = (y_val >= 0).sum()
    print(f"Train labeled: {n_tr} ({n_tr/len(h1_train)*100:.1f}%)")
    print(f"Val labeled:   {n_vl} ({n_vl/len(h1_val)*100:.1f}%)")
    for c, name in enumerate(ORACLE_CLASSES):
        n = (y_train == c).sum()
        print(f"  {name}: {n} ({n/n_tr*100:.1f}%)")

    feats_train = engine.compute(h1_train); feats_val = engine.compute(h1_val)

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
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
    return train_loader, val_loader


def main():
    config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    engine = BTCFeatureEngine()
    train_ldr, val_ldr = build_data(config, engine)

    model = SimplePredictor(n_features=17, seq_len=config.seq_len_h1, n_classes=4).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1)

    # Single-stage training with high LR, long patience
    n_epochs = 300
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-7)
    best_va, best_sd, no_imp = 0.0, None, 0

    print(f"\n=== Training {n_epochs} epochs, LR=1e-3 ===")
    for ep in range(n_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in train_ldr:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1)==y).sum().item(); tr_total += y.size(0)

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_ldr:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                vl_loss += crit(logits, y).item()
                vl_correct += (logits.argmax(1)==y).sum().item(); vl_total += y.size(0)

        ta = tr_correct/tr_total*100; va = vl_correct/vl_total*100
        scheduler.step(va)
        if va > best_va: best_va = va; best_sd = {k:v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1

        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:4d}: train={ta:.1f}% val={va:.1f}% best={best_va:.1f}% lr={opt.param_groups[0]['lr']:.2e}")

        if ep >= 50 and no_imp >= 30:
            print(f"  Early stop ep {ep+1}")
            break

    model.load_state_dict(best_sd)
    save_path = os.path.join(config.model_dir, "btc_h1_predictive.pt")
    torch.save({"model_state_dict": best_sd, "val_acc": best_va,
                "oracle_classes": ORACLE_CLASSES,
                "arch": "SimplePredictor"}, save_path)
    print(f"\nBest val acc: {best_va:.1f}% | Saved to {save_path}")


if __name__ == "__main__":
    main()
