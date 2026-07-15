"""
M15 pullback confirmation for entry timing.

Three checks combined:
  1. Model check: CNN-GRU entry_confidence > threshold AND direction_bias agrees with H1
  2. Pullback check: price pulled back to EMA21 on M15 and is now reversing
  3. Pattern check: engulfing or pin bar at EMA zone

Returns Confirmation with confirmed(bool), entry_price, confidence.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Confirmation:
    confirmed: bool
    entry_price: float
    confidence: float
    reason: str = ""
    details: dict = None


class M15Confirmation:
    def __init__(self, model_confidence_threshold=0.6, ema_span=21,
                 pullback_min_bars=2, recovery_min_bars=2):
        self.model_threshold = model_confidence_threshold
        self.ema_span = ema_span
        self.pullback_min_bars = pullback_min_bars
        self.recovery_min_bars = recovery_min_bars

    def confirm(self, m15_model_output: dict, m15_bars: np.ndarray,
                m15_features: np.ndarray, h1_direction: int,
                current_price: float) -> Confirmation:
        """
        Args:
            m15_model_output: {'entry_confidence': tensor, 'direction_bias': tensor}
            m15_bars: raw OHLCV (n_bars, 5) — [open, high, low, close, volume]
            m15_features: computed features (n_bars, 17)
            h1_direction: 1 for long, -1 for short (from H1 regime)
            current_price: latest price

        Returns:
            Confirmation
        """
        # ── Check 1: Model ──
        entry_conf = m15_model_output["entry_confidence"]
        direction_bias = m15_model_output["direction_bias"]
        if hasattr(entry_conf, 'item'):
            entry_conf = entry_conf.item()
        if hasattr(direction_bias, 'item'):
            direction_bias = direction_bias.item()
        model_ok = entry_conf >= self.model_threshold
        direction_agrees = (direction_bias > 0 and h1_direction == 1) or \
                           (direction_bias < 0 and h1_direction == -1)

        if not model_ok or not direction_agrees:
            reasons = []
            if not model_ok:
                reasons.append(f"confidence {entry_conf:.2f} < {self.model_threshold}")
            if not direction_agrees:
                reasons.append(f"direction bias {direction_bias:+.2f} vs H1 {h1_direction}")
            return Confirmation(False, current_price, entry_conf,
                                reason="; ".join(reasons))

        # ── Check 2: Pullback ──
        pullback_ok = self._check_pullback(m15_bars, h1_direction)

        # ── Check 3: Pattern ──
        pattern_ok, pattern_name = self._check_pattern(m15_bars, h1_direction)

        # Combine
        checks = [model_ok, pullback_ok, pattern_ok]
        n_passed = sum(checks)
        confirmed = n_passed >= 2  # need 2 of 3

        # Confidence: model_conf weighted by checks passed
        if n_passed == 3:
            final_conf = entry_conf * 1.2
        elif n_passed == 2:
            final_conf = entry_conf
        else:
            final_conf = entry_conf * 0.5

        details = {
            "model_ok": model_ok,
            "pullback_ok": pullback_ok,
            "pattern_ok": pattern_ok,
            "pattern_name": pattern_name,
            "entry_conf": entry_conf,
            "direction_bias": direction_bias,
        }

        reason = f"{n_passed}/3 checks passed"
        if confirmed:
            reason += f" (model{' +pullback' if pullback_ok else ''}{' +' + pattern_name if pattern_ok else ''})"

        return Confirmation(
            confirmed=confirmed,
            entry_price=current_price,
            confidence=min(1.0, final_conf),
            reason=reason,
            details=details,
        )

    def _check_pullback(self, bars: np.ndarray, direction: int) -> bool:
        """Check if price pulled back to EMA and is reversing."""
        if len(bars) < self.ema_span + 4:
            return False

        closes = bars[:, 3]  # close
        ema = self._ema(closes, self.ema_span)

        # Last few bars
        recent_close = closes[-self.recovery_min_bars - self.pullback_min_bars:]
        recent_ema = ema[-self.recovery_min_bars - self.pullback_min_bars:]

        if direction == 1:  # Long — price dipped below/at EMA, now recovering
            # Were any recent bars at or below EMA?
            near_ema = np.any(recent_close[:-self.recovery_min_bars] <= recent_ema[:-self.recovery_min_bars] * 1.005)
            # Are last 2 bars moving up?
            recovering = closes[-1] > closes[-2] > closes[-3]
            return near_ema and recovering
        else:  # Short — price rallied to/above EMA, now dropping
            near_ema = np.any(recent_close[:-self.recovery_min_bars] >= recent_ema[:-self.recovery_min_bars] * 0.995)
            recovering = closes[-1] < closes[-2] < closes[-3]
            return near_ema and recovering

    def _check_pattern(self, bars: np.ndarray, direction: int) -> tuple:
        """Check for engulfing or pin bar patterns. Returns (ok, pattern_name)."""
        if len(bars) < 3:
            return False, "none"

        # Last completed bar + current
        o1, h1, l1, c1 = bars[-2, 0:4]
        o2, h2, l2, c2 = bars[-1, 0:4]

        # Engulfing
        if direction == 1:  # Bullish engulfing
            if c2 > o2 and c1 < o1 and c2 > o1 and o2 < c1:
                return True, "bullish_engulfing"
        else:  # Bearish engulfing
            if c2 < o2 and c1 > o1 and c2 < o1 and o2 > c1:
                return True, "bearish_engulfing"

        # Pin bar (long wick, small body at opposite end)
        body = abs(c2 - o2)
        total_range = h2 - l2
        if total_range < 1e-9:
            return False, "none"

        lower_wick = min(o2, c2) - l2
        upper_wick = h2 - max(o2, c2)
        wick_ratio = max(lower_wick, upper_wick) / total_range if total_range > 0 else 0

        if wick_ratio < 0.67 or body / total_range > 0.3:
            return False, "none"

        if direction == 1 and lower_wick > upper_wick * 2:
            return True, "bullish_pinbar"
        elif direction == -1 and upper_wick > lower_wick * 2:
            return True, "bearish_pinbar"

        return False, "none"

    @staticmethod
    def _ema(x, span):
        """Exponential moving average."""
        alpha = 2.0 / (span + 1)
        ema = np.zeros_like(x)
        ema[0] = x[0]
        for i in range(1, len(x)):
            ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
        return ema
