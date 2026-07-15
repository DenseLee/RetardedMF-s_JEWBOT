"""
Entry Gate per architecture spec:
  Selects Trend model or Range model based on regime.
  Outputs direction + signal strength.
  Gates transitions (blocks during regime transitions).

  TREND_UP/TREND_DOWN → use TrendModel (directional entries)
  RANGE → use RangeModel (mean-reversion entries)
  TRANSITION → gate closed (wait for resolution)
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GateDecision:
    entry_signal: bool = False
    direction: int = 0           # 1=long, -1=short, 0=no signal
    confidence: float = 0.0
    signal_strength: float = 0.0  # 0-1 scale, how strong the signal is
    model_used: str = ""         # "trend" or "range" or "none"
    reason: str = ""
    components: dict = field(default_factory=dict)


class TrendModel:
    """Directional entries for trending markets."""

    def __init__(self, min_confidence=0.6):
        self.min_confidence = min_confidence

    def evaluate(self, regime: str, confidence: float,
                 h1_embedding: np.ndarray = None) -> GateDecision:
        direction = 1 if regime == "TREND_UP" else -1
        ok = confidence >= self.min_confidence
        return GateDecision(
            entry_signal=ok, direction=direction,
            confidence=confidence, signal_strength=confidence if ok else 0.0,
            model_used="trend",
            reason="trend signal" if ok else f"trend confidence {confidence:.2f} < {self.min_confidence}",
            components={"regime": regime, "confidence": confidence})


class RangeModel:
    """Mean-reversion entries for ranging markets: fade extremes."""

    def __init__(self, bb_position_threshold=0.7, min_confidence=0.4):
        self.bb_threshold = bb_position_threshold
        self.min_confidence = min_confidence

    def evaluate(self, regime: str, confidence: float,
                 bb_position: float = 0.0) -> GateDecision:
        """
        In range: short when bb_position > +threshold (overbought),
                  long when bb_position < -threshold (oversold).
        """
        if abs(bb_position) < self.bb_threshold:
            return GateDecision(entry_signal=False, direction=0, confidence=confidence,
                                model_used="range",
                                reason=f"bb_position {bb_position:+.2f} within neutral zone")

        direction = -1 if bb_position > 0 else 1  # fade the extreme
        signal_strength = min(1.0, abs(bb_position) / self.bb_threshold)
        return GateDecision(
            entry_signal=True, direction=direction,
            confidence=confidence * signal_strength,
            signal_strength=signal_strength,
            model_used="range",
            reason=f"range fade bb_pos={bb_position:+.2f}")


class EntryGate:
    """
    Combined entry gate: selects TrendModel or RangeModel based on regime.
    Blocks TRANSITION regimes entirely.
    """

    def __init__(self, min_confidence=0.6, min_atr_pct=0.3, max_atr_pct=0.9):
        self.trend_model = TrendModel(min_confidence=min_confidence)
        self.range_model = RangeModel()
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    def evaluate(self, regime: str, regime_confidence: float,
                 atr_percentile: float, bb_position: float = 0.0,
                 h1_embedding: np.ndarray = None) -> GateDecision:
        # Volatility gate (shared)
        if atr_percentile < self.min_atr_pct:
            return GateDecision(entry_signal=False, model_used="none",
                                reason=f"vol too low (atr_pct={atr_percentile:.2f})")
        if atr_percentile > self.max_atr_pct:
            return GateDecision(entry_signal=False, model_used="none",
                                reason=f"vol too high (atr_pct={atr_percentile:.2f})")

        # Route to appropriate model based on regime
        if regime == "TREND_UP" or regime == "TREND_DOWN":
            return self.trend_model.evaluate(regime, regime_confidence, h1_embedding)
        elif regime == "RANGE":
            return self.range_model.evaluate(regime, regime_confidence, bb_position)
        else:  # TRANSITION
            return GateDecision(entry_signal=False, model_used="none",
                                reason="gate closed: TRANSITION regime")
