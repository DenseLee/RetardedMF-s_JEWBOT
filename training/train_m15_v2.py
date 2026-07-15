"""Train M15 model v2: binary classifier on direction-specific trade outcome labels."""
import sys, os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from models.cnn_gru_m15 import CNNGRUM15

config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
data_path = os.path.join(config.data_dir, "m15_training_data_v2.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

X = data["sequences"]  # (N, 20, 17)
y = data["labels"]     # (N,) 0/1

print(f"Loaded {len(y)} samples: {y.sum()} positive ({y.sum()/len(y)*100:.1f}%), "
      f"{(1-y).sum()} negative ({(1-y).sum()/len(y)*100:.1f}%)")

# Split: 2022-2024 train, 2025 val (80/20 approximate)
split_idx = int(len(y) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"Train: {len(y_train)} ({y_train.sum()} pos), Val: {len(y_val)} ({y_val.sum()} pos)")

# Convert to tensors
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).long()

# Weighted sampling for class imbalance
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sample_weights = np.where(y_train == 1, pos_weight, 1.0)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size_m15,
                          sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=config.batch_size_m15 * 2,
                        shuffle=False)

# Model: same CNN-GRU architecture, single output head for entry confidence only
# We modify CNNGRUM15 to add a mode that doesn't output direction_bias
class M15EntryClassifier(nn.Module):
    def __init__(self, n_features=17, seq_len=20, cnn_channels=(16, 32, 64), gru_hidden=64,
                 gru_layers=1, dropout=0.2):
        super().__init__()
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels[0], 3, padding=1),
            nn.BatchNorm1d(cnn_channels[0]), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels[0], cnn_channels[1], 3, padding=1),
            nn.BatchNorm1d(cnn_channels[1]), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(cnn_channels[1], cnn_channels[2], 3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]), nn.ReLU())
        cnn_out_len = seq_len // 4  # two MaxPool1d(2)
        self.gru = nn.GRU(cnn_channels[2], gru_hidden, gru_layers, batch_first=True, dropout=dropout if gru_layers > 1 else 0)
        self.shared = nn.Sequential(nn.Linear(gru_hidden, 32), nn.ReLU(), nn.Dropout(dropout))
        self.entry_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 17, 20)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = x.transpose(1, 2)  # (B, seq_out, channels)
        _, h = self.gru(x)
        h = h[-1]  # last layer hidden
        shared = self.shared(h)
        entry_conf = self.entry_head(shared)
        return {"entry_confidence": entry_conf}


model = M15EntryClassifier(
    n_features=config.n_features, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device)

# Loss with class weighting
pos_w = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]).to(device)
criterion = nn.BCELoss()  # No pos_weight since we use WeightedRandomSampler

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

best_val_acc = 0; best_epoch = 0; patience_counter = 0
best_state = None

print(f"\nTraining {config.epochs} epochs, batch_size={config.batch_size_m15}, "
      f"pos_weight={pos_weight.item():.1f}")

for epoch in range(config.epochs):
    model.train()
    train_loss = 0; train_correct = 0; train_total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).float()
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out["entry_confidence"].squeeze(), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
        preds = (out["entry_confidence"].squeeze() >= 0.5).long()
        train_correct += (preds == yb.long()).sum().item()
        train_total += len(xb)

    train_acc = train_correct / max(train_total, 1)

    # Validation
    model.eval()
    val_loss = 0; val_correct = 0; val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            out = model(xb)
            loss = criterion(out["entry_confidence"].squeeze(), yb)
            val_loss += loss.item() * len(xb)
            preds = (out["entry_confidence"].squeeze() >= 0.5).long()
            val_correct += (preds == yb.long()).sum().item()
            val_total += len(xb)

    val_acc = val_correct / max(val_total, 1)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc; best_epoch = epoch + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_acc:.3f} val_loss={val_loss/val_total:.4f} "
              f"val_acc={val_acc:.3f}  best={best_val_acc:.3f} @ {best_epoch}")

    if patience_counter >= config.early_stop_patience:
        print(f"  Early stop at epoch {epoch+1}")
        break

# Save
save_path = os.path.join(config.model_dir, "btc_m15_v2.pt")
torch.save({"model_state_dict": best_state, "val_acc": best_val_acc, "epoch": best_epoch}, save_path)
print(f"\nBest val_acc: {best_val_acc:.3f} at epoch {best_epoch}")
print(f"Saved to {save_path}")

# Quick evaluation
model.load_state_dict(best_state)
model.eval()
all_preds = []; all_labels = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = out["entry_confidence"].squeeze().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

all_preds = np.array(all_preds); all_labels = np.array(all_labels)
print(f"\nValidation distribution:")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pos = all_preds >= thresh
    if pos.sum() == 0: continue
    acc = (all_labels[pos] == 1).mean()
    print(f"  conf>={thresh:.1f}: {pos.sum():5d} samples ({pos.sum()/len(all_preds)*100:.1f}%), "
          f"precision={acc:.3f}")
