"""
Curriculum training config — 3 stages progressing from easy to hard.

Stage 1 — Strong Trends Only:
  - Filter: ADX > 25, |EMA slope| > strong_trend threshold
  - Goal: Learn clear directional patterns
  - Epochs: 40, LR: 1e-4

Stage 2 — Add Weak Trends:
  - Filter: ADX > 20, |EMA slope| > weak_trend threshold
  - Goal: Generalize to moderate trends
  - Epochs: 30, LR: 5e-5

Stage 3 — Full Dataset (Gate-Filtered):
  - All bars that pass the entry gate
  - Goal: Learn when NOT to trade
  - Epochs: 30, LR: 1e-5
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class StageConfig:
    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    min_adx: float = 0.0
    min_ema_slope_abs: float = 0.0
    require_gate_open: bool = False
    description: str = ""


@dataclass
class CurriculumConfig:
    stages: List[StageConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.stages:
            self.stages = [
                StageConfig(
                    name="strong_trends",
                    epochs=40,
                    learning_rate=1e-4,
                    batch_size=64,
                    min_adx=25.0,
                    min_ema_slope_abs=0.002,
                    require_gate_open=False,
                    description="Only clear directional trends (bull_strong/bear_strong)",
                ),
                StageConfig(
                    name="add_weak_trends",
                    epochs=30,
                    learning_rate=5e-5,
                    batch_size=64,
                    min_adx=20.0,
                    min_ema_slope_abs=0.0005,
                    require_gate_open=False,
                    description="Include weak trends for generalization",
                ),
                StageConfig(
                    name="full_dataset",
                    epochs=30,
                    learning_rate=1e-5,
                    batch_size=64,
                    min_adx=0.0,
                    min_ema_slope_abs=0.0,
                    require_gate_open=True,
                    description="All bars passing entry gate — learn when to stay out",
                ),
            ]

    def filter_bars(self, stage_idx: int, df_labels, adx=None, ema_slope=None,
                    gate_open=None):
        """Return boolean mask of bars to include for this stage."""
        if stage_idx >= len(self.stages):
            stage_idx = len(self.stages) - 1
        stage = self.stages[stage_idx]

        mask = np.ones(len(df_labels), dtype=bool)

        # Only include labeled (non-flat) bars for training
        mask &= (df_labels["label"] != 0)

        # Filter by regime if param specified
        if "regime_filter_pass" in df_labels.columns and stage.min_adx > 0:
            mask &= df_labels["regime_filter_pass"]

        # ADX filter
        if adx is not None and stage.min_adx > 0:
            mask &= (adx >= stage.min_adx)

        # EMA slope filter
        if ema_slope is not None and stage.min_ema_slope_abs > 0:
            mask &= (np.abs(ema_slope) >= stage.min_ema_slope_abs)

        # Gate filter
        if stage.require_gate_open and gate_open is not None:
            mask &= gate_open

        return mask

    def get_stage_info(self, stage_idx: int) -> dict:
        """Get human-readable stage info."""
        if stage_idx >= len(self.stages):
            return {"name": "unknown", "description": "Stage index out of range"}
        s = self.stages[stage_idx]
        return {
            "name": s.name,
            "description": s.description,
            "epochs": s.epochs,
            "lr": s.learning_rate,
            "batch_size": s.batch_size,
            "min_adx": s.min_adx,
            "min_ema_slope": s.min_ema_slope_abs,
            "gate_required": s.require_gate_open,
        }
