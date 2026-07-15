"""
Barrier-based label generation per architecture spec.

For each bar, simulates a 1:3 R:R trade and checks which barrier
(SL or TP) is hit first within max_hold bars.

  label = 1 if TP hit before SL  (win)
  label = 0 if SL hit before TP  (loss)
  label = neutral if neither hit within max hold

This directly trains the model to find 1:3 setups, not just
directional accuracy.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class LabelStats:
    n_total: int; n_long: int; n_short: int; n_neutral: int
    n_win_long: int; n_win_short: int; n_loss_long: int; n_loss_short: int
    win_rate_long: float; win_rate_short: float
    avg_win_r: float; avg_loss_r: float


class AsymmetricLabelGenerator:
    def __init__(self, sl_atr_mult=1.0, tp_atr_mult=3.0, max_hold=12,
                 min_atr_pct=0.003):
        self.sl_atr_mult = sl_atr_mult; self.tp_atr_mult = tp_atr_mult
        self.max_hold = max_hold; self.min_atr_pct = min_atr_pct

    def create_labels(self, df: pd.DataFrame, atr: np.ndarray = None,
                      verbose=True) -> pd.DataFrame:
        """
        Barrier-based labels: TP hit before SL = win, SL first = loss.

        Returns DataFrame with columns:
          bar_idx, direction, label (1=win, 0=loss), outcome, entry_price, sl, tp
        """
        n = len(df)
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)

        if atr is None:
            tr = np.maximum(highs - lows,
                            np.maximum(np.abs(highs - np.roll(closes, 1)),
                                       np.abs(lows - np.roll(closes, 1))))
            tr[0] = highs[0] - lows[0]
            atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

        rows = []
        total = n - 50
        for i in range(total):
            a = atr[i]; entry = closes[i]
            if a / max(entry, 1e-9) < self.min_atr_pct:
                continue

            for direction in [1, -1]:
                sl = entry - direction * self.sl_atr_mult * a
                tp = entry + direction * self.tp_atr_mult * a
                outcome = "neutral"

                for j in range(i + 1, min(i + self.max_hold + 1, n)):
                    hi, lo = highs[j], lows[j]
                    if direction == 1:
                        if lo <= sl: outcome = "loss"; break
                        if hi >= tp: outcome = "win"; break
                    else:
                        if hi >= sl: outcome = "loss"; break
                        if lo <= tp: outcome = "win"; break

                rows.append({"bar_idx": i, "direction": direction,
                             "label": 1 if outcome == "win" else 0,
                             "outcome": outcome, "entry_price": entry,
                             "sl": sl, "tp": tp})
            if verbose and i % 5000 == 0 and i > 0:
                print(f"  Label generation: {i}/{total} bars ({i/total*100:.0f}%)")

        return pd.DataFrame(rows)

    def generate_m15_labels(self, df: pd.DataFrame, atr: np.ndarray = None,
                            lookahead=8, verbose=True) -> np.ndarray:
        """
        Generate binary M15 entry labels: 1 if entering here is profitable
        within lookahead bars, 0 otherwise. Uses same barrier logic as H1.
        """
        return self.create_labels(df, atr, verbose=verbose)

    def get_stats(self, labels_df: pd.DataFrame) -> LabelStats:
        df = labels_df
        n = len(df)
        if n == 0:
            return LabelStats(0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        long = df[df["direction"] == 1]; short = df[df["direction"] == -1]
        n_long = len(long); n_short = len(short)
        n_neutral = int((df["outcome"] == "neutral").sum())

        win_l = int((long["outcome"] == "win").sum())
        win_s = int((short["outcome"] == "win").sum())
        loss_l = int((long["outcome"] == "loss").sum())
        loss_s = int((short["outcome"] == "loss").sum())

        wr_l = win_l / max(win_l + loss_l, 1)
        wr_s = win_s / max(win_s + loss_s, 1)

        return LabelStats(n_total=n, n_long=n_long, n_short=n_short,
                          n_neutral=n_neutral, n_win_long=win_l, n_win_short=win_s,
                          n_loss_long=loss_l, n_loss_short=loss_s,
                          win_rate_long=wr_l, win_rate_short=wr_s,
                          avg_win_r=self.tp_atr_mult, avg_loss_r=-self.sl_atr_mult)


# Legacy wrapper for backward compatibility
class LabelGenerator(AsymmetricLabelGenerator):
    """Backward-compatible wrapper — delegates to AsymmetricLabelGenerator."""
    pass


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_btc import BTCConfig
    from data.feature_engine_btc import BTCFeatureEngine

    config = BTCConfig()
    data_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Test on 2024
    mask = (df["timestamp"] >= "2024-01-01") & (df["timestamp"] < "2025-01-01")
    df = df[mask].reset_index(drop=True)
    print(f"Bars: {len(df)}")

    gen = AsymmetricLabelGenerator()
    labels = gen.create_labels(df)
    stats = gen.get_stats(labels)
    print(f"Total labels: {stats.n_total}")
    print(f"  Long:  {stats.n_long}  (win={stats.n_win_long}, loss={stats.n_loss_long})  WR={stats.win_rate_long*100:.1f}%")
    print(f"  Short: {stats.n_short}  (win={stats.n_win_short}, loss={stats.n_loss_short})  WR={stats.win_rate_short*100:.1f}%")
    print(f"  Neutral: {stats.n_neutral}")
    total_win = stats.n_win_long + stats.n_win_short
    total_loss = stats.n_loss_long + stats.n_loss_short
    total_wr = total_win / max(total_win + total_loss, 1) * 100
    print(f"  Combined WR: {total_wr:.1f}%")
    print(f"  Expected at 1:3 → PF = {total_wr/100 * 3 / max((1-total_wr/100) * 1, 0.01):.2f}")
