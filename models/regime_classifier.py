"""
Regime classifier — 4 classes per architecture spec.
  TREND_UP=0, TREND_DOWN=1, RANGE=2, TRANSITION=3

Rule-based fallback from EMA slope + ATR percentile.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REGIME_NAMES = ["TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION"]
REGIME_DIRECTION = {"TREND_UP": 1, "TREND_DOWN": -1, "RANGE": 0, "TRANSITION": 0}


class RegimeClassifier(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, n_classes=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, embedding):
        return F.log_softmax(self.net(embedding), dim=1)

    def raw_logits(self, embedding):
        """Return raw logits (no softmax) for temperature scaling."""
        return self.net(embedding)


class RuleBasedRegimeDetector:
    """EMA slope + ATR percentile → TREND_UP/DOWN/RANGE/TRANSITION."""

    def __init__(self, ema_fast=9, ema_slow=21, atr_window=14,
                 slope_window=4, strong_trend=0.0015, weak_trend=0.0003):
        self.ema_fast = ema_fast; self.ema_slow = ema_slow
        self.atr_window = atr_window; self.slope_window = slope_window
        self.strong_trend = strong_trend; self.weak_trend = weak_trend
        self._atr_history = []
        self._ema_fast_val = None; self._ema_slow_val = None
        self._prev_regime = "RANGE"
        self._regime_bars = 0

    def update(self, high, low, close):
        tr = high - low
        if self._atr_history:
            tr = max(tr, abs(high - self._atr_history[-1][1]), abs(low - self._atr_history[-1][1]))
        self._atr_history.append((close, tr))
        if len(self._atr_history) > self.atr_window * 5:
            self._atr_history = self._atr_history[-self.atr_window * 5:]

        if self._ema_fast_val is None:
            self._ema_fast_val = close; self._ema_slow_val = close
        else:
            af = 2.0 / (self.ema_fast + 1); a_s = 2.0 / (self.ema_slow + 1)
            self._ema_fast_val = af * close + (1 - af) * self._ema_fast_val
            self._ema_slow_val = a_s * close + (1 - a_s) * self._ema_slow_val

        return self._classify()

    def _classify(self):
        if self._ema_slow_val is None or self._ema_slow_val == 0:
            return {"regime": "RANGE", "direction": 0, "confidence": 0.0, "atr_percentile": 0.5}

        slope = (self._ema_fast_val - self._ema_slow_val) / abs(self._ema_slow_val)

        if slope > self.strong_trend:
            new_regime = "TREND_UP"; conf = min(1.0, abs(slope) / (self.strong_trend * 3))
        elif slope > self.weak_trend:
            new_regime = "TREND_UP"; conf = min(1.0, abs(slope) / (self.strong_trend * 2))
        elif slope < -self.strong_trend:
            new_regime = "TREND_DOWN"; conf = min(1.0, abs(slope) / (self.strong_trend * 3))
        elif slope < -self.weak_trend:
            new_regime = "TREND_DOWN"; conf = min(1.0, abs(slope) / (self.strong_trend * 2))
        else:
            conf = max(0.0, 1.0 - abs(slope) / self.weak_trend)
            new_regime = "RANGE"

        # Only mark TRANSITION when crossing TREND ↔ RANGE boundary, for 2 bars max
        major_change = ((self._prev_regime in ("TREND_UP", "TREND_DOWN") and new_regime == "RANGE") or
                        (self._prev_regime == "RANGE" and new_regime in ("TREND_UP", "TREND_DOWN")))
        if major_change:
            self._regime_bars = 1
            new_regime = "TRANSITION"
        elif self._prev_regime == "TRANSITION":
            self._regime_bars += 1
            if self._regime_bars <= 2:
                new_regime = "TRANSITION"

        self._prev_regime = new_regime

        # ATR percentile
        if len(self._atr_history) >= self.atr_window:
            atr_vals = [x[1] for x in self._atr_history[-self.atr_window:]]
            curr_atr = sum(atr_vals) / len(atr_vals)
            all_atr = sorted([x[1] for x in self._atr_history])
            atr_pct = sum(1 for v in all_atr if v <= curr_atr) / max(len(all_atr), 1)
        else:
            atr_pct = 0.5

        return {"regime": new_regime, "direction": REGIME_DIRECTION.get(new_regime, 0),
                "confidence": conf, "atr_percentile": atr_pct, "ema_slope": slope}


def classify_regime(encoder, classifier, features_tensor, rule_detector,
                    model_confidence_threshold=0.6, temperature=4.0):
    """
    Classify market regime using encoder + classifier with temperature scaling.

    Temperature > 1 softens the softmax, producing calibrated probabilities
    instead of saturated 0.000/1.000 outputs. T=4.0 is tuned for this model's
    typical logit spread of 10-28, reducing it to ~2.5-7 for meaningful
    probability distributions.
    """
    with torch.no_grad():
        enc_out = encoder(features_tensor)
        raw_logits = classifier.raw_logits(enc_out["embedding"])
        probs = torch.softmax(raw_logits / temperature, dim=1)
        max_prob, pred_class = probs.max(dim=1)
        max_prob = max_prob.item(); pred_class = pred_class.item()

    if max_prob >= model_confidence_threshold:
        regime = REGIME_NAMES[pred_class]
        return {"regime": regime, "direction": REGIME_DIRECTION[regime],
                "confidence": max_prob, "source": "model",
                "class_probs": probs.squeeze(0).tolist()}
    else:
        out = rule_detector._classify() if rule_detector._ema_slow_val else \
              {"regime": "RANGE", "direction": 0, "confidence": 0.0}
        return {**out, "source": "rule"}
