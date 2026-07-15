"""
Train a REGRESSION model to predict expected R for long and short.

Input: 96 H1 bars of 17 features
Output: (predicted_long_r, predicted_short_r) — continuous values in ATR units

The model predicts "if I enter long now, I expect X R in the next 18 hours.
If I enter short now, I expect Y R."

Direction = whichever has higher predicted R.
Conviction = difference between the two.
Magnitude = predicted value itself (expected return).

This cannot mode-collapse because it's regression, not classification.
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine


class DirectionalRegressor(nn.Module):
    """Predict expected R for long and short from 96-bar context.
    Outputs two scalars: (long_r_pred, short_r_pred).
    """
    def __init__(self, n_features=17, seq_len=96, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2, stride=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 5, padding=2, stride=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 64, 5, padding=2, stride=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),  # long_r, short_r
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feats = self.conv(x)
        return self.head(feats)  # (B, 2)


def compute_targets(h1_df, m15_df, config):
    """Compute oracle long_r and short_r for each H1 bar (regression targets)."""
    max_hold = 72
    n_h1, n_m15 = len(h1_df), len(m15_df)
    targets = np.full((n_h1, 2), np.nan, dtype=np.float32)
    engine = BTCFeatureEngine()
    h1_feats = engine.compute(h1_df)
    h1_atr_pct = h1_feats[:, 6]

    for h1_i in range(config.seq_len_h1, n_h1):
        h1_close = float(h1_df['close'].iloc[h1_i])
        atr = h1_atr_pct[h1_i] * h1_close
        if atr < 1: atr = h1_close * 0.005

        h1_ts = h1_df['timestamp'].iloc[h1_i]
        # Find M15 bars after this H1 close using proper timestamp alignment
        m15_after = m15_df[m15_df['timestamp'] > h1_ts]
        if len(m15_after) == 0: continue
        start_m15 = m15_after.index[0]
        end_m15 = min(start_m15 + max_hold, n_m15)

        best_long, best_short = 0.0, 0.0
        for m15_j in range(start_m15, end_m15):
            hi = float(m15_df['high'].iloc[m15_j])
            lo = float(m15_df['low'].iloc[m15_j])
            best_long = max(best_long, (hi - h1_close) / atr)
            best_short = max(best_short, (h1_close - lo) / atr)

        targets[h1_i, 0] = best_long
        targets[h1_i, 1] = best_short

    return targets


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
    print("Computing regression targets (long_r, short_r) from M15 data...")
    y_train = compute_targets(h1_train, m15, config)
    y_val   = compute_targets(h1_val, m15, config)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Stats
    valid_tr = ~np.isnan(y_train[:, 0])
    valid_vl = ~np.isnan(y_val[:, 0])
    print(f"Train valid: {valid_tr.sum()}  Val valid: {valid_vl.sum()}")
    print(f"Train long_r: mean={np.nanmean(y_train[:,0]):.2f} median={np.nanmedian(y_train[:,0]):.2f} max={np.nanmax(y_train[:,0]):.1f}")
    print(f"Train short_r: mean={np.nanmean(y_train[:,1]):.2f} median={np.nanmedian(y_train[:,1]):.2f} max={np.nanmax(y_train[:,1]):.1f}")

    feats_train = engine.compute(h1_train); feats_val = engine.compute(h1_val)

    def make_loader(feats, targets, seq_len, batch, shuffle=True):
        X, Y = [], []
        for i in range(seq_len, len(feats)):
            if np.isnan(targets[i, 0]): continue
            seq = engine.compute_sequence(feats, i, seq_len)
            X.append(seq); Y.append(targets[i])
        X = np.stack(X).astype(np.float32); Y = np.stack(Y).astype(np.float32)
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

    model = DirectionalRegressor(n_features=17, seq_len=config.seq_len_h1).to(device)
    crit = nn.SmoothL1Loss(beta=2.0)  # Huber — less penalty on outliers
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    best_vl, best_sd, no_imp = float('inf'), None, 0

    n_epochs = 100
    print(f"\n=== Training {n_epochs} epochs, LR=1e-3 ===")
    for ep in range(n_epochs):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for X, y in train_ldr:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            pred = model(X)
            loss = crit(pred, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * y.size(0); tr_n += y.size(0)

        model.eval()
        vl_loss, vl_n, vl_dir_correct, vl_dir_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for X, y in val_ldr:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                vl_loss += crit(pred, y).item() * y.size(0); vl_n += y.size(0)
                # Direction accuracy: does predicted sign match actual sign?
                pred_dir = torch.sign(pred[:, 0] - pred[:, 1])
                actual_dir = torch.sign(y[:, 0] - y[:, 1])
                mask = actual_dir != 0
                vl_dir_correct += (pred_dir[mask] == actual_dir[mask]).sum().item()
                vl_dir_total += mask.sum().item()

        tr_rmse = (tr_loss/tr_n)**0.5; vl_rmse = (vl_loss/vl_n)**0.5
        vl_dir_acc = vl_dir_correct/vl_dir_total*100 if vl_dir_total > 0 else 0
        scheduler.step(vl_rmse)
        if vl_rmse < best_vl: best_vl = vl_rmse; best_sd = {k:v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1

        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:4d}: tr_rmse={tr_rmse:.3f} v_rmse={vl_rmse:.3f} v_dir={vl_dir_acc:.1f}% best={best_vl:.3f} lr={opt.param_groups[0]['lr']:.2e}")

        if ep >= 30 and no_imp >= 20:
            print(f"  Early stop ep {ep+1}")
            break

    model.load_state_dict(best_sd)
    # Final directional accuracy on val set
    model.eval()
    dir_correct, dir_total = 0, 0
    with torch.no_grad():
        for X, y in val_ldr:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pd_sign = torch.sign(pred[:, 0] - pred[:, 1])
            ad_sign = torch.sign(y[:, 0] - y[:, 1])
            mask = ad_sign != 0
            dir_correct += (pd_sign[mask] == ad_sign[mask]).sum().item()
            dir_total += mask.sum().item()
    dir_acc = dir_correct / dir_total * 100 if dir_total > 0 else 0

    save_path = os.path.join(config.model_dir, "btc_h1_regression.pt")
    torch.save({"model_state_dict": best_sd, "val_rmse": best_vl,
                "val_dir_acc": dir_acc, "arch": "DirectionalRegressor"}, save_path)
    print(f"\nBest val RMSE: {best_vl:.3f}R | Val dir acc: {dir_acc:.1f}% | Saved to {save_path}")


if __name__ == "__main__":
    main()
