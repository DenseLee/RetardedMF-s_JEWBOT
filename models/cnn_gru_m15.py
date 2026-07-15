"""
M15 CNN-GRU Execution Encoder — lightweight, 20-bar input per spec.

Architecture:
  Input: (B, 20, 17)
    → Conv1D(17→16, k=3) + BN + ReLU + MaxPool  → (B, 16, 10)
    → Conv1D(16→32, k=3) + BN + ReLU + MaxPool  → (B, 32, 5)
    → Conv1D(32→64, k=3) + BN + ReLU            → (B, 64, 5)
    → permute to (B, 5, 64)
    → GRU(64→64, 1 layer) → last hidden: (B, 64)
    → Linear(64, 32) → ReLU → head
    → entry_conf (sigmoid) + direction_bias (tanh)
"""
import torch
import torch.nn as nn


class CNNGRUM15(nn.Module):
    def __init__(self, n_features=17, seq_len=20,
                 cnn_channels=(16, 32, 64), kernel_size=3,
                 gru_hidden=64, gru_layers=1, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(cnn_channels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if i < len(cnn_channels) - 1:  # no pool on last conv
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # CNN output length: seq // (2^(len-1)) = 20 // 4 = 5
        self.cnn_output_len = seq_len // (2 ** (len(cnn_channels) - 1))

        self.gru = nn.GRU(input_size=cnn_channels[-1], hidden_size=gru_hidden,
                          num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)

        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, 32), nn.ReLU(), nn.Dropout(dropout))
        self.entry_conf = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.direction_bias = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, 17, 20)
        cnn_out = self.cnn(x)            # (B, 64, 5)
        gru_in = cnn_out.permute(0, 2, 1)  # (B, 5, 64)
        _, h_n = self.gru(gru_in)
        h_last = h_n[-1]                 # (B, 64)
        shared = self.shared(h_last)     # (B, 32)
        return {"entry_confidence": self.entry_conf(shared),
                "direction_bias": self.direction_bias(shared)}
