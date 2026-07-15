"""Configurable slippage on entries, exits, SL fills, and TP fills."""
from dataclasses import dataclass


@dataclass
class SlippageConfig:
    entry_slippage_pct: float = 0.02   # 0.02% for entries
    exit_slippage_pct: float = 0.02    # 0.02% for exits
    sl_slippage_pct: float = 0.01      # 0.01% for SL fills (adverse)
    tp_slippage_pct: float = 0.01      # 0.01% for TP fills (adverse)
    mode: str = "percentage"           # "percentage" or "fixed"
    fixed_points: float = 5.0          # price points for fixed mode


class SlippageModel:
    """Applies adverse slippage — all fills slightly worse for the trader."""

    def __init__(self, config: SlippageConfig = None):
        self.cfg = config or SlippageConfig()

    def entry_price(self, bar_close: float, direction: int) -> float:
        """Long buy at ask (worse), short sell at bid (worse)."""
        mult = 1 if direction == 1 else -1
        return bar_close * (1.0 + mult * self.cfg.entry_slippage_pct / 100.0)

    def exit_price(self, bar_close: float, direction: int) -> float:
        """Long exit sell at bid (worse), short exit buy at ask (worse)."""
        mult = -1 if direction == 1 else 1
        return bar_close * (1.0 + mult * self.cfg.exit_slippage_pct / 100.0)

    def sl_fill_price(self, sl_price: float, direction: int) -> float:
        """SL fill — always adverse: below SL for longs, above for shorts."""
        mult = -1 if direction == 1 else 1
        return sl_price * (1.0 + mult * self.cfg.sl_slippage_pct / 100.0)

    def tp_fill_price(self, tp_price: float, direction: int) -> float:
        """TP fill — slightly adverse: below TP for longs, above for shorts."""
        mult = -1 if direction == 1 else 1
        return tp_price * (1.0 + mult * self.cfg.tp_slippage_pct / 100.0)

    @staticmethod
    def none() -> "SlippageModel":
        """Zero slippage — fills at exact prices."""
        cfg = SlippageConfig(
            entry_slippage_pct=0.0, exit_slippage_pct=0.0,
            sl_slippage_pct=0.0, tp_slippage_pct=0.0)
        return SlippageModel(cfg)
