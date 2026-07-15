"""
Train the M15 CNN-GRU model for entry confirmation.
Binary classification: should we enter at this M15 bar?

Uses barrier-based labels: for each M15 bar, check if a 1:3 R:R entry
would win within 8 M15 bars. Label=1 if win, 0 if loss.

Training window: 2022-01-01 → 2025-12-31 per spec.
Epochs: 50 (reduced from 100 for first pass).
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_gru_m15 import CNNGRUM15
from training.label_generator import AsymmetricLabelGenerator


def build_data(config, engine):
    """Load M15 data, generate barrier labels, build DataLoaders."""
    data_path = os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
    df = pd.read_csv(data_path); df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    train_mask = (df["timestamp"] >= config.m15_train_start) & (df["timestamp"] < config.m15_train_end)
    val_mask   = (df["timestamp"] >= config.val_start) & (df["timestamp"] < config.val_end)
    df_train = df[train_mask].reset_index(drop=True)
    df_val   = df[val_mask].reset_index(drop=True)
    print(f"Train M15: {len(df_train)} bars ({config.m15_train_start} → {config.m15_train_end})")
    print(f"Val M15:   {len(df_val)} bars ({config.val_start} → {config.val_end})")

    # Features
    print("Computing M15 features...")
    feats_train = engine.compute(df_train)
    feats_val   = engine.compute(df_val)

    # ATR
    atr_train = feats_train[:, 6] * df_train["close"].values
    atr_val   = feats_val[:, 6] * df_val["close"].values

    # Barrier labels
    print("Generating barrier labels (this may take a few minutes)...")
    label_gen = AsymmetricLabelGenerator(sl_atr_mult=1.0, tp_atr_mult=3.0, max_hold=8)
    labels_train = label_gen.create_labels(df_train, atr_train)
    labels_val   = label_gen.create_labels(df_val, atr_val, verbose=False)

    # Build per-bar binary labels: 1 if ANY direction wins at this bar
    bar_labels_train = np.zeros(len(df_train), dtype=np.float32)
    bar_labels_val   = np.zeros(len(df_val), dtype=np.float32)
    for _, row in labels_train.iterrows():
        if row["outcome"] == "win":
            bar_labels_train[int(row["bar_idx"])] = 1.0
    for _, row in labels_val.iterrows():
        if row["outcome"] == "win":
            bar_labels_val[int(row["bar_idx"])] = 1.0

    n_pos = (bar_labels_train == 1).sum(); n_neg = (bar_labels_train == 0).sum()
    print(f"Train labels: {n_pos} positive ({n_pos/len(bar_labels_train)*100:.1f}%), "
          f"{n_neg} negative")

    # Build sequences
    def make_loader(feats, labels, seq_len, batch, balance=True):
        X, Y, W = [], [], []
        for i in range(seq_len, len(feats)):
            seq = engine.compute_sequence(feats, i, seq_len)
            X.append(seq); Y.append(labels[i])
            W.append(5.0 if labels[i] == 1 else 1.0)  # weight pos class higher
        X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        if balance:
            weights = torch.tensor(W, dtype=torch.float)
            sampler = WeightedRandomSampler(weights, len(weights))
            return DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True)
        return DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)

    train_loader = make_loader(feats_train, bar_labels_train, config.seq_len_m15,
                               config.batch_size_m15)
    val_loader = make_loader(feats_val, bar_labels_val, config.seq_len_m15,
                             config.batch_size_m15, balance=False)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def main():
    config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"); t_start = time.time()

    engine = BTCFeatureEngine()
    train_loader, val_loader = build_data(config, engine)

    model = CNNGRUM15(
        n_features=17, seq_len=config.seq_len_m15, cnn_channels=config.gru_cnn_channels,
        gru_hidden=config.gru_hidden, gru_layers=config.gru_layers,
        dropout=config.gru_dropout).to(device)

    pos_weight = torch.tensor([3.0]).to(device)  # weight wins higher (fewer of them)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    EPOCHS = 50
    best_val_loss = float("inf"); patience = 0

    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device); optimizer.zero_grad()
            out = model(X); loss = criterion(out["entry_confidence"].squeeze(), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_loss += loss.item()

        model.eval(); val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X); preds = out["entry_confidence"].squeeze()
                val_loss += criterion(preds, y).item()
                val_correct += ((preds > 0.5) == y).sum().item(); val_total += y.size(0)

        train_loss /= len(train_loader); val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% lr={lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}"); break

    elapsed = (time.time() - t_start) / 60
    print(f"Best val loss: {best_val_loss:.4f}  |  Time: {elapsed:.1f} min")

    model.load_state_dict(best_state)
    save_path = os.path.join(config.model_dir, "btc_m15_model.pt")
    torch.save({"model_state_dict": best_state, "config": config}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
