"""Train H4 CNN-LSTM encoder for higher-timeframe trend classification."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector
from torch.utils.data import DataLoader, TensorDataset

config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Generate H4 data from H1 ──
h1f = pd.read_csv(os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)

# Resample H1 → H4: 4-hour OHLC bars
h1f.set_index("timestamp", inplace=True)
h4f = h1f.resample("4h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum"
}).dropna().reset_index()

print(f"H1 bars: {len(h1f):,}  →  H4 bars: {len(h4f):,}")

# Train/val split: 2020-2024 train, 2025 val
ft_train = pd.Timestamp("2020-01-01", tz="UTC")
ft_val = pd.Timestamp("2025-01-01", tz="UTC")
et_val = pd.Timestamp("2025-12-31", tz="UTC")

h4_train = h4f[(h4f["timestamp"] >= ft_train) & (h4f["timestamp"] < ft_val)].reset_index(drop=True)
h4_val = h4f[(h4f["timestamp"] >= ft_val) & (h4f["timestamp"] <= et_val)].reset_index(drop=True)
print(f"Train H4: {len(h4_train)}  |  Val H4: {len(h4_val)}")

# ── Generate labels: vectorized EMA slope + ATR regime detection ──
engine = BTCFeatureEngine()
regime_map = {"TREND_UP": 0, "TREND_DOWN": 1, "RANGE": 2, "TRANSITION": 3}

def label_h4_vectorized(df):
    features = engine.compute(df)
    closes = df["close"].values
    highs = df["high"].values; lows = df["low"].values

    # EMA9 and EMA21 slopes
    ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values
    ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().values

    # True range and ATR
    prev_close = np.roll(closes, 1); prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    labels = np.zeros(len(df), dtype=np.int64)
    for i in range(14, len(df)):
        # Slope over last 4 bars (16 hours — meaningful on H4)
        ema9_slope = (ema9[i] - ema9[i-4]) / max(abs(ema9[i-4]), 1e-12)
        ema21_slope = (ema21[i] - ema21[i-4]) / max(abs(ema21[i-4]), 1e-12)
        atr_pct = atr14[i] / closes[i]
        atr_pct_hist = atr14[max(0,i-50):i+1] / closes[max(0,i-50):i+1]
        atr_percentile = (atr_pct_hist < atr_pct).mean()

        if atr_percentile < 0.3:
            labels[i] = regime_map["RANGE"]
        elif ema9_slope > 0.0002 and ema21_slope > 0.0001:
            labels[i] = regime_map["TREND_UP"]
        elif ema9_slope < -0.0002 and ema21_slope < -0.0001:
            labels[i] = regime_map["TREND_DOWN"]
        elif abs(ema9_slope) < 0.0001 and abs(ema21_slope) < 0.0001:
            labels[i] = regime_map["RANGE"]
        else:
            labels[i] = regime_map["TRANSITION"]
    return features, labels

print("Labeling training data...")
train_feats, train_labels = label_h4_vectorized(h4_train)
print("Labeling validation data...")
val_feats, val_labels = label_h4_vectorized(h4_val)

# Convert to sequences
def make_sequences(features, labels, seq_len=96):
    X, y = [], []
    for i in range(seq_len, len(features)):
        seq = features[i-seq_len:i].copy()
        # Pad if needed
        if seq.shape[0] < seq_len:
            pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])
        X.append(seq)
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

print("Building sequences...")
X_train, y_train = make_sequences(train_feats, train_labels, config.seq_len_h1)
X_val, y_val = make_sequences(val_feats, val_labels, config.seq_len_h1)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Class distribution — Train: {np.bincount(y_train)}  Val: {np.bincount(y_val)}")

# ── Train H4 encoder ──
model = CNNLSTMEncoder(
    n_features=config.n_features, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
    bidirectional=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=config.learning_rate, weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2)

best_acc = 0; best_state = None; patience = 0
print(f"\nTraining {config.epochs} epochs...")
for epoch in range(config.epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        logits = out["regime_logits"]
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        train_loss += loss.item() * len(xb)

    model.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            logits = out["regime_logits"]
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += len(yb)
    val_acc = val_correct / val_total
    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_ds):.4f} val_acc={val_acc:.3f} best={best_acc:.3f}")

    if patience >= config.early_stop_patience:
        print(f"  Early stop at epoch {epoch+1}")
        break

# Save
save_path = os.path.join(config.model_dir, "btc_h4_encoder.pt")
torch.save({"encoder_state_dict": best_state, "val_acc": best_acc}, save_path)
print(f"\nBest val_acc: {best_acc:.3f}")
print(f"Saved to {save_path}")
