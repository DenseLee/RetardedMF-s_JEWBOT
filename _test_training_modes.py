"""Compare 3 training modes: full, frozen encoder, fine-tuned."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier
from training.label_generator import LabelGenerator

config = BTCConfig()
device = torch.device("cpu")
engine = BTCFeatureEngine()
label_gen = LabelGenerator(lookahead=12)
SEQ = config.seq_len_h1

# Load data (2024 train, 2025 val)
data_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
train_mask = (df["timestamp"] >= "2024-01-01") & (df["timestamp"] < "2025-01-01")
val_mask   = (df["timestamp"] >= "2025-01-01") & (df["timestamp"] < "2026-01-01")
df_train = df[train_mask].reset_index(drop=True)
df_val   = df[val_mask].reset_index(drop=True)
print(f"Train bars: {len(df_train)}, Val bars: {len(df_val)}")

# Features + labels
feats_train = engine.compute(df_train)
feats_val   = engine.compute(df_val)
c_train = df_train["close"].values; c_val = df_val["close"].values
atr_train = feats_train[:, 7] * c_train; atr_val = feats_val[:, 7] * c_val
labels_train = label_gen.generate_h1_labels(df_train, atr_train)
labels_val   = label_gen.generate_h1_labels(df_val, atr_val)

def labels_to_regime(labels, returns):
    r = np.full(len(labels), -1, dtype=np.int64)
    r[labels == 1]  = np.where(returns[labels == 1] > 2.0, 0, 1)
    r[labels == -1] = np.where(returns[labels == -1] > 2.0, 2, 3)
    return r

y_train = labels_to_regime(labels_train["label"].values, labels_train["return_r"].values)
y_val   = labels_to_regime(labels_val["label"].values, labels_val["return_r"].values)

def make_loader(feats, y, seq_len, batch=64):
    X, Y = [], []
    for i in range(seq_len, len(feats)):
        if y[i] < 0: continue
        seq = engine.compute_sequence(feats, i, seq_len)
        X.append(seq); Y.append(y[i])
    X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

train_loader = make_loader(feats_train, y_train, SEQ, config.batch_size)
val_loader   = make_loader(feats_val, y_val, SEQ, config.batch_size)
n_train = len(train_loader.dataset)
n_val   = len(val_loader.dataset)
print(f"Train samples: {n_train}, Val samples: {n_val}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Training helpers
def train_one_epoch(encoder, classifier, loader, optimizer, criterion, freeze_encoder):
    encoder.train(); classifier.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if freeze_encoder:
            with torch.no_grad():
                enc = encoder(Xb)
        else:
            enc = encoder(Xb)
        logp = classifier(enc["embedding"])
        loss = criterion(logp, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logp.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(loader), correct / total * 100

def validate(encoder, classifier, loader, criterion):
    encoder.eval(); classifier.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            enc = encoder(Xb)
            logp = classifier(enc["embedding"])
            loss += criterion(logp, yb).item()
            correct += (logp.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss / len(loader), correct / total * 100

def make_fresh_encoder():
    return CNNLSTMEncoder(n_features=17, seq_len=SEQ, cnn_channels=(32, 64, 128),
        lstm_hidden=128, lstm_layers=2, dropout=0.3, embedding_dim=128,
        regime_classes=4, bidirectional=True).to(device)

def make_fresh_classifier():
    return RegimeClassifier(embedding_dim=128, n_classes=4).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
EPOCHS_FULL = 30
EPOCHS_FROZEN = 30
EPOCHS_FINETUNE_HEAD = 10
EPOCHS_FINETUNE_ALL = 20

# ═══ MODE 1: Full training (encoder + classifier from scratch) ═══
print("\n" + "=" * 60)
print("MODE 1: Full training (encoder + classifier from scratch)")
print("=" * 60)
enc1 = make_fresh_encoder()
cls1 = make_fresh_classifier()
opt1 = optim.AdamW(list(enc1.parameters()) + list(cls1.parameters()), lr=1e-4, weight_decay=1e-4)

t0 = time.time(); best_val1 = 0
for epoch in range(EPOCHS_FULL):
    tl, ta = train_one_epoch(enc1, cls1, train_loader, opt1, criterion, freeze_encoder=False)
    vl, va = validate(enc1, cls1, val_loader, criterion)
    if va > best_val1: best_val1 = va
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: train_acc={ta:.1f}% val_acc={va:.1f}%")
t1 = time.time() - t0
print(f"  Best val acc: {best_val1:.1f}%  Time: {t1:.0f}s")

# ═══ MODE 2: Frozen encoder (random encoder, train classifier only) ═══
print("\n" + "=" * 60)
print("MODE 2: Frozen encoder (random weights, classifier head only)")
print("=" * 60)
enc2 = make_fresh_encoder()
for p in enc2.parameters():
    p.requires_grad = False
cls2 = make_fresh_classifier()
opt2 = optim.AdamW(cls2.parameters(), lr=1e-4, weight_decay=1e-4)

t0 = time.time(); best_val2 = 0
for epoch in range(EPOCHS_FROZEN):
    tl, ta = train_one_epoch(enc2, cls2, train_loader, opt2, criterion, freeze_encoder=True)
    vl, va = validate(enc2, cls2, val_loader, criterion)
    if va > best_val2: best_val2 = va
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: train_acc={ta:.1f}% val_acc={va:.1f}%")
t2 = time.time() - t0
print(f"  Best val acc: {best_val2:.1f}%  Time: {t2:.0f}s")

# ═══ MODE 3: Fine-tune (frozen head first, then unfreeze all) ═══
print("\n" + "=" * 60)
print("MODE 3: Fine-tune (frozen→unfrozen, curriculum-style)")
print("=" * 60)
enc3 = make_fresh_encoder()
cls3 = make_fresh_classifier()

# Phase A: freeze encoder, train classifier head
print("  Phase A: Frozen encoder, training classifier head...")
for p in enc3.parameters():
    p.requires_grad = False
opt3a = optim.AdamW(cls3.parameters(), lr=1e-4, weight_decay=1e-4)
best_head = 0
for epoch in range(EPOCHS_FINETUNE_HEAD):
    tl, ta = train_one_epoch(enc3, cls3, train_loader, opt3a, criterion, freeze_encoder=True)
    vl, va = validate(enc3, cls3, val_loader, criterion)
    if va > best_head: best_head = va
print(f"  Phase A best val acc: {best_head:.1f}%")

# Phase B: unfreeze all, train at lower LR
print("  Phase B: Unfrozen, fine-tuning all params...")
for p in enc3.parameters():
    p.requires_grad = True
opt3b = optim.AdamW(list(enc3.parameters()) + list(cls3.parameters()), lr=1e-5, weight_decay=1e-4)
t0 = time.time(); best_val3 = 0
for epoch in range(EPOCHS_FINETUNE_ALL):
    tl, ta = train_one_epoch(enc3, cls3, train_loader, opt3b, criterion, freeze_encoder=False)
    vl, va = validate(enc3, cls3, val_loader, criterion)
    if va > best_val3: best_val3 = va
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: train_acc={ta:.1f}% val_acc={va:.1f}%")
t3 = time.time() - t0
print(f"  Best val acc: {best_val3:.1f}%  Time: {t3:.0f}s")

# ═══ SUMMARY ═══
print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY  |  {n_train} train / {n_val} val samples  |  {SEQ}-bar sequences")
print(f"{'='*60}")
print(f"  {'Mode':<38s} {'Val Acc':>8s}  {'Time':>8s}  {'vs Random':>10s}")
print(f"  {'-'*38} {'-'*8}  {'-'*8}  {'-'*10}")
print(f"  {'1. Full training (scratch)':<38s} {best_val1:>7.1f}%  {t1:>7.0f}s  {best_val1-25:>+9.1f}%")
print(f"  {'2. Frozen encoder (head only)':<38s} {best_val2:>7.1f}%  {t2:>7.0f}s  {best_val2-25:>+9.1f}%")
print(f"  {'3. Fine-tune (frozen→unfrozen)':<38s} {best_val3:>7.1f}%  {t3:>7.0f}s  {best_val3-25:>+9.1f}%")
print(f"  {'Random baseline (4-class)':<38s} {'~25.0%':>8s}")
print(f"{'='*60}")
