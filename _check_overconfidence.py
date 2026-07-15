"""Show raw classifier logits + temperature effect to quantify overconfidence."""
import sys, os, numpy as np, pandas as pd, torch

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, REGIME_NAMES

cfg = BTCConfig()
device = torch.device("cpu")

h1_path = os.path.join(cfg.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)

encoder_path = os.path.join(cfg.model_dir, "btc_h1_encoder.pt")
ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
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

fe = BTCFeatureEngine()

# Take last 50 H1 bars from CSV
h1_subset = h1.tail(50 + cfg.seq_len_h1).reset_index(drop=True)

print(f"{'Time':20s} {'Raw logits (TU,TD,R,TR)':45s} {'T=1 prob':12s} {'T=4 prob':12s} {'T=8 prob':12s} {'Pred':12s}")
print("-" * 120)

for i in range(cfg.seq_len_h1, len(h1_subset)):
    window = h1_subset.iloc[:i+1]
    ts = window["timestamp"].iloc[-1]
    feats = fe.compute(window)
    seq = fe.compute_sequence(feats, len(feats) - 1, cfg.seq_len_h1)
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_out = encoder(tensor)
        raw = classifier.raw_logits(enc_out["embedding"])
        raw_np = raw.squeeze(0).cpu().numpy()

    p_t1 = torch.softmax(raw / 1.0, dim=1).squeeze(0)
    p_t4 = torch.softmax(raw / 4.0, dim=1).squeeze(0)
    p_t8 = torch.softmax(raw / 8.0, dim=1).squeeze(0)

    pred_t4 = REGIME_NAMES[p_t4.argmax().item()]
    max_t4 = p_t4.max().item()

    logit_str = f"TU={raw_np[0]:+.1f} TD={raw_np[1]:+.1f} R={raw_np[2]:+.1f} TR={raw_np[3]:+.1f}"
    print(f"{str(ts)[:19]:20s} {logit_str:45s} TU={p_t1[0]:.4f}  TU={p_t4[0]:.4f}  TU={p_t8[0]:.4f}  {pred_t4:12s} (T=4 max={max_t4:.4f})")

# Summary
print("\n--- Summary: fraction of bars model predicts TREND_UP ---")
count_tu_t4 = 0
count_tu_t8 = 0
count_tu_t16 = 0

for i in range(cfg.seq_len_h1, len(h1_subset)):
    window = h1_subset.iloc[:i+1]
    feats = fe.compute(window)
    seq = fe.compute_sequence(feats, len(feats) - 1, cfg.seq_len_h1)
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = encoder(tensor)
        raw = classifier.raw_logits(enc_out["embedding"])
    p4 = torch.softmax(raw / 4.0, dim=1)
    p8 = torch.softmax(raw / 8.0, dim=1)
    p16 = torch.softmax(raw / 16.0, dim=1)
    if p4.argmax().item() == 0: count_tu_t4 += 1
    if p8.argmax().item() == 0: count_tu_t8 += 1
    if p16.argmax().item() == 0: count_tu_t16 += 1

n = len(h1_subset) - cfg.seq_len_h1
print(f"T=4:  TREND_UP in {count_tu_t4}/{n} bars ({count_tu_t4/n*100:.0f}%)")
print(f"T=8:  TREND_UP in {count_tu_t8}/{n} bars ({count_tu_t8/n*100:.0f}%)")
print(f"T=16: TREND_UP in {count_tu_t16}/{n} bars ({count_tu_t16/n*100:.0f}%)")
