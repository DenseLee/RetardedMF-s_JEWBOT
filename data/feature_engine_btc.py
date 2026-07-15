"""
BTC-specific feature engine — 17 features per architecture spec.

12 base features (same as V12++) + 5 BTC-specific:
  Price structure:   hl_range_norm, oc_range_norm, gap_norm, rsi_norm
  Volatility:        atr_pct, bb_width, bb_position, volatility_ratio
  Trend:             adx_norm, adx_direction, momentum_norm
  Volume:            volume_ratio
  BTC-specific:      funding_rate_norm, volume_delta_norm, btc_dominance_norm,
                     sentiment_score_norm, liquidation_proxy

Note: funding_rate, volume_delta, btc_dominance, and sentiment require
external data. When unavailable, features default to 0 (neutral).
"""
import numpy as np
import pandas as pd

BTC_FEATURE_NAMES = [
    # Price structure (6)
    "hl_range_norm", "oc_range_norm", "gap_norm",
    "rsi_norm", "bb_position", "momentum_norm",
    # Volatility (4)
    "atr_pct", "bb_width", "volatility_ratio", "volume_ratio",
    # Trend (2)
    "adx_norm", "adx_direction",
    # BTC-specific (5)
    "funding_rate_norm", "volume_delta_norm",
    "btc_dominance_norm", "sentiment_score_norm",
    "liquidation_proxy",
]
N_FEATURES = len(BTC_FEATURE_NAMES)


class BTCFeatureEngine:
    FEATURE_NAMES = BTC_FEATURE_NAMES
    N_FEATURES = N_FEATURES

    def __init__(self, zscore_window=1000):
        self.zscore_window = zscore_window

    def compute(self, df: pd.DataFrame,
                external: dict = None) -> np.ndarray:
        """
        Compute all 17 features.

        Args:
            df: [timestamp, open, high, low, close, volume]
            external: optional dict with keys funding_rate, volume_delta,
                      btc_dominance, sentiment_score (arrays same length as df)

        Returns:
            (n, 17) float32 array
        """
        external = external or {}
        n = len(df)
        out = np.zeros((n, N_FEATURES), dtype=np.float32)

        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        v = df["volume"].values.astype(np.float64)

        out[:, 0] = self._hl_range_norm(h, l, c)
        out[:, 1] = self._oc_range_norm(o, c)
        out[:, 2] = self._gap_norm(o, c)
        out[:, 3] = self._rsi_norm(c)
        out[:, 4] = self._bb_position(c)
        out[:, 5] = self._momentum_norm(c)
        out[:, 6] = self._atr_pct(h, l, c)
        out[:, 7] = self._bb_width(c)
        out[:, 8] = self._volatility_ratio(h, l, c)
        out[:, 9] = self._volume_ratio(v)
        out[:, 10] = self._adx_norm(h, l, c)
        out[:, 11] = self._adx_direction(h, l, c)

        # BTC-specific (use external data if provided, else 0)
        out[:, 12] = self._zscore(external.get("funding_rate", np.zeros(n)))
        out[:, 13] = self._zscore(external.get("volume_delta", np.zeros(n)))
        out[:, 14] = self._zscore(external.get("btc_dominance", np.zeros(n)))
        out[:, 15] = self._zscore(external.get("sentiment_score", np.zeros(n)))
        out[:, 16] = self._liquidation_proxy(h, l, c)

        np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return out

    def compute_sequence(self, features: np.ndarray, idx: int, seq_len: int) -> np.ndarray:
        start = max(0, idx - seq_len + 1)
        window = features[start:idx + 1]
        if len(window) < seq_len:
            pad = np.zeros((seq_len - len(window), N_FEATURES), dtype=np.float32)
            window = np.vstack([pad, window])
        return window

    # ── helpers ──
    def _zscore(self, x, window=None):
        w = window or self.zscore_window
        if len(x) < w:
            w = max(2, len(x))
        r = pd.Series(x).rolling(w, min_periods=2)
        mean = r.mean().values
        std = r.std().values.copy()
        std[std < 1e-12] = 1.0
        return (x - mean) / std

    @staticmethod
    def _ema(x, span):
        return pd.Series(x).ewm(span=span, adjust=False).mean().values

    @staticmethod
    def _sma(x, period):
        return pd.Series(x).rolling(period, min_periods=1).mean().values

    @staticmethod
    def _tr(h, l, c):
        prev_c = np.roll(c, 1); prev_c[0] = c[0]
        return np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    def _hl_range_norm(self, h, l, c):
        return self._zscore((h - l) / np.maximum(c, 1e-9))

    def _oc_range_norm(self, o, c):
        return self._zscore(np.abs(c - o) / np.maximum(c, 1e-9))

    def _gap_norm(self, o, c):
        gap = np.abs(o - np.roll(c, 1)) / np.maximum(np.roll(c, 1), 1e-9)
        gap[0] = 0
        return self._zscore(gap)

    @staticmethod
    def _rsi_norm(c, period=14):
        d = np.diff(c, prepend=c[0])
        gain = np.where(d > 0, d, 0.0)
        loss = np.where(d < 0, -d, 0.0)
        avg_g = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_l = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = avg_g / np.maximum(avg_l, 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 100.0

    def _atr_pct(self, h, l, c):
        tr = self._tr(h, l, c)
        atr = self._ema(tr, 14)
        return atr / np.maximum(c, 1e-9)

    def _bb_width(self, c, period=20):
        sma = self._sma(c, period)
        std = pd.Series(c).rolling(period, min_periods=2).std().values
        return (2.0 * std) / np.maximum(sma, 1e-9)

    def _bb_position(self, c, period=20):
        sma = self._sma(c, period)
        std = pd.Series(c).rolling(period, min_periods=2).std().values
        pos = (c - sma) / np.maximum(2.0 * std, 1e-12)
        return np.clip(pos, -1.0, 1.0)

    def _volatility_ratio(self, h, l, c, short=14, long=100):
        tr = self._tr(h, l, c)
        atr_s = self._ema(tr, short)
        atr_l = self._ema(tr, long)
        return atr_s / np.maximum(atr_l, 1e-12)

    def _volume_ratio(self, v):
        sma20 = self._sma(v, 20)
        return v / np.maximum(sma20, 1e-12)

    @staticmethod
    def _adx_norm(h, l, c, period=14):
        tr = pd.Series(np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))).ewm(span=period, adjust=False).mean().values.copy()
        tr[0] = h[0] - l[0]
        up = np.maximum(h - np.roll(h, 1), 0.0); up[0] = 0
        down = np.maximum(np.roll(l, 1) - l, 0.0); down[0] = 0
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        ndm = np.where((down > up) & (down > 0), down, 0.0)
        pdi = 100 * pd.Series(pdm).ewm(span=period, adjust=False).mean().values / np.maximum(tr, 1e-12)
        ndi = 100 * pd.Series(ndm).ewm(span=period, adjust=False).mean().values / np.maximum(tr, 1e-12)
        dx = 100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx / 100.0

    @staticmethod
    def _adx_direction(h, l, c, period=14):
        tr = pd.Series(np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))).ewm(span=period, adjust=False).mean().values.copy()
        tr[0] = h[0] - l[0]
        up = np.maximum(h - np.roll(h, 1), 0.0); up[0] = 0
        down = np.maximum(np.roll(l, 1) - l, 0.0); down[0] = 0
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        ndm = np.where((down > up) & (down > 0), down, 0.0)
        pdi = pd.Series(pdm).ewm(span=period, adjust=False).mean().values / np.maximum(tr, 1e-12)
        ndi = pd.Series(ndm).ewm(span=period, adjust=False).mean().values / np.maximum(tr, 1e-12)
        return np.sign(pdi - ndi)

    def _momentum_norm(self, c, period=10):
        roc = (c - np.roll(c, period)) / np.maximum(np.roll(c, period), 1e-9)
        roc[:period] = 0
        return self._zscore(roc)

    def _liquidation_proxy(self, h, l, c):
        """Unusually large range relative to ATR, rolling max - 1.0."""
        tr = self._tr(h, l, c)
        atr = self._ema(tr, 14)
        range_atr_ratio = (h - l) / np.maximum(atr, 1e-12)
        proxy = pd.Series(range_atr_ratio).rolling(3, min_periods=1).max().values - 1.0
        return self._zscore(proxy)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_btc import BTCConfig

    config = BTCConfig()
    data_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    since = pd.Timestamp("2026-01-01", tz="UTC")
    df = df[df["timestamp"] >= since].reset_index(drop=True)
    print(f"Bars from 2026-01-01: {len(df)}")

    engine = BTCFeatureEngine()
    feats = engine.compute(df)
    print(f"Shape: {feats.shape}  NaN: {np.isnan(feats).sum()}  Inf: {np.isinf(feats).sum()}")
    for i, name in enumerate(BTC_FEATURE_NAMES):
        col = feats[:, i]
        print(f"  {name:>24s}: mean={col.mean():+.4f}  std={col.std():.4f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")
