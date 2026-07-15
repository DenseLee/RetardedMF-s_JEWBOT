"""
H1 CNN-LSTM Encoder for BTC regime/trend classification.

Architecture:
  Input: (B, seq_len, 17)
    → Conv1D(17→32, k=3) + BN + ReLU + MaxPool(2)  → (B, 32, seq/2)
    → Conv1D(32→64, k=3) + BN + ReLU + MaxPool(2)  → (B, 64, seq/4)
    → Conv1D(64→128, k=3) + BN + ReLU + MaxPool(2) → (B, 128, seq/8)
    → permute to (B, seq/8, 128)
    → BiLSTM(128→128, 2 layers, dropout=0.3)
    → last hidden concat: (B, 256)
    → embedding head: Linear(256, 128) → Tanh → embedding
    → regime head:   Linear(256, 64) → ReLU → Linear(64, 4) → logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_features=17, seq_len=96,
                 cnn_channels=(32, 64, 128), kernel_size=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 embedding_dim=128, regime_classes=4,
                 bidirectional=True):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.lstm_hidden_dim = lstm_hidden * (2 if bidirectional else 1)

        # CNN layers
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.append(ConvBlock(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output sequence length
        self.cnn_output_len = seq_len
        for _ in cnn_channels:
            self.cnn_output_len //= 2

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Dropout after LSTM
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.embedding_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, embedding_dim),
            nn.Tanh(),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, regime_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            dict with 'embedding' (B, embedding_dim) and 'regime_logits' (B, regime_classes)
        """
        # CNN expects (B, C, L)
        x = x.permute(0, 2, 1)  # (B, features, seq)
        cnn_out = self.cnn(x)   # (B, 128, cnn_output_len)

        # LSTM expects (B, L, C)
        lstm_in = cnn_out.permute(0, 2, 1)  # (B, cnn_output_len, 128)
        lstm_out, (h_n, _) = self.lstm(lstm_in)

        # Concatenate final hidden states from last layer
        if self.bidirectional:
            h_forward = h_n[-2, :, :]  # forward of last layer
            h_backward = h_n[-1, :, :]  # backward of last layer
            h_cat = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_cat = h_n[-1, :, :]

        h_cat = self.dropout(h_cat)

        return {
            "embedding": self.embedding_head(h_cat),
            "regime_logits": self.regime_head(h_cat),
        }
