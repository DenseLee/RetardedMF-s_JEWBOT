"""
BTC Bot — All parameters aligned with Listening Architecture spec.
"""
import os
from dataclasses import dataclass, field


@dataclass
class BTCConfig:
    # ── Symbol ──
    symbol: str = "BTCUSD"
    ccxt_symbol: str = "BTC/USDT"

    # ── Timeframes ──
    tf_h1: str = "H1"
    tf_m15: str = "M15"
    tf_h4: str = "4h"

    # ── Sequence lengths (M15=20 per spec: lightweight 20-bar encoder) ──
    seq_len_h1: int = 96       # 4 days of H1
    seq_len_m15: int = 20      # 5 hours of M15 (spec: lightweight on 20-bar)

    # ── Listening Architecture ──
    max_listen_bars: int = 8   # max M15 bars to wait for confirmation (2 hours)

    # ── Risk & Money Management ──
    risk_pct: float = 0.02          # 2% risk per trade (spec: dollar_risk = capital * risk_pct)
    max_daily_loss: float = 0.05    # 5% hard stop
    max_position_pct: float = 1.0   # never exceed full capital

    # ── Trade Manager (Phased: survive → detect → trail → pressure) ──
    # Phase 1: Wide SL gives trade room. Phase 2: Kill stalled trades at bar 4.
    # Phase 3: Trail profits. Phase 4: Tighten trail as time runs out.
    initial_sl: float = 1.5         # Phase 1 wide SL (ATR multiplier)
    hard_tp: float = 2.0            # TP at 2.0 ATR (wider than SL for positive R:R)
    phase1_bars: int = 4            # bars in Phase 1 (M15 bars)
    phase2_mfe_min: float = 0.1     # min MFE at bar 4 to survive (R units)
    phase3_trail: float = 0.40      # Phase 3 trail distance (ATR) — tight to lock in
    phase4_trail: float = 0.25      # Phase 4 tighter trail (ATR)
    phase4_start: int = 12          # when Phase 4 starts (M15 bars)
    max_hold_bars: int = 18         # time stop (M15 bars)
    # Legacy params kept for backward compat
    breakeven_trigger: float = 0.80
    trail_trigger: float = 0.80
    trail_dist: float = 0.75
    trail_dist_s: float = 0.75
    regime_tighten: float = 0.25
    mae_guard_retrace: float = 2.5

    # ── Entry Gating ──
    min_regime_confidence: float = 0.6
    regime_temperature: float = 4.0     # T>1 softens softmax, reduces overconfidence
    min_atr_percentile: float = 0.3
    max_atr_percentile: float = 0.9
    min_entry_confidence: float = 0.6
    h1_signal_ttl_bars: int = 8    # how long an H1 signal stays active

    # ── CNN-LSTM Encoder (H1) ──
    n_features: int = 17
    cnn_channels: tuple = (32, 64, 128)
    cnn_kernel_size: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    embedding_dim: int = 128
    regime_classes: int = 4        # TREND_UP / TREND_DOWN / RANGE / TRANSITION

    # ── CNN-GRU (M15) ──
    gru_cnn_channels: tuple = (16, 32, 64)
    gru_hidden: int = 64
    gru_layers: int = 1
    gru_dropout: float = 0.2
    m15_embedding_dim: int = 64    # spec: 64-dim M15 embedding

    # ── Training ──
    batch_size: int = 64
    batch_size_m15: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15

    # ── Curriculum ──
    # Stage 1: 200 episodes, strong trend only (ADX > 40)
    # Stage 2: 300 episodes, trend + moderate vol
    # Stage 3: 1000+ episodes, full dataset + halving weighting
    stage1_episodes: int = 200
    stage2_episodes: int = 300
    stage3_episodes: int = 1000
    stage1_min_adx: float = 40.0
    stage2_min_adx: float = 25.0
    halving_weight_mult: float = 3.0  # post-2024 halving weighted 3×

    # ── Training Windows ──
    h1_train_start: str = "2020-01-01"
    h1_train_end: str = "2025-12-31"
    m15_train_start: str = "2022-01-01"
    m15_train_end: str = "2025-12-31"
    val_start: str = "2026-01-01"
    val_end: str = "2026-05-31"

    # ── Walk-Forward ──
    train_months: int = 6
    test_months: int = 1
    min_train_bars: int = 2000

    # ── MT5 ──
    mt5_magic: int = 20260517
    mt5_deviation: int = 20
    mt5_filling: int = 1           # ORDER_FILLING_IOC

    # ── Sentiment (Alpha Vantage) ──
    sentiment_enabled: bool = False
    sentiment_cache_minutes: int = 60
    sentiment_block_threshold: float = 0.7

    # ── Polling ──
    poll_interval_seconds: int = 15
    dry_run: bool = True

    # ── Paths ──
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = "TrainingData"
    model_dir: str = "models"
    log_dir: str = "logs"
    status_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "status"))

    def __post_init__(self):
        self.data_dir = os.path.join(self.project_root, self.data_dir)
        self.model_dir = os.path.join(self.project_root, self.model_dir)
        self.log_dir = os.path.join(self.project_root, self.log_dir)
        self.status_dir = os.path.join(self.project_root, self.status_dir)
        for d in [self.data_dir, self.model_dir, self.log_dir, self.status_dir]:
            os.makedirs(d, exist_ok=True)
        self.lstm_hidden_dim = self.lstm_hidden * (2 if self.lstm_bidirectional else 1)
        self.cnn_output_len_h1 = self.seq_len_h1 // 8
        self.cnn_output_len_m15 = self.seq_len_m15 // 8
