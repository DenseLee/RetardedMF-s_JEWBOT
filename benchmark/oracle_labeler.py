"""
Oracle Labeler — perfect-foresight benchmark for every H1 bar.

For each H1 bar, looks ahead up to max_hold bars and simulates the optimal
LONG and SHORT trade using M1 intrabar data. Outputs a label per bar:
  LONG_ONLY  — only a long trade would have been profitable
  SHORT_ONLY — only a short trade would have been profitable
  BOTH_GOOD  — both directions profitable
  CHOP       — neither direction works (stops out before profit)
  LONG_GOOD  — long is profitable, short is break-even
  SHORT_GOOD — short is profitable, long is break-even

Respects the same constraints as the live bot: SL, TP, max hold, M1 intrabar.
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.trade_manager_btc import TradeManager


@dataclass
class OracleLabel:
    timestamp: str
    close: float
    atr_dollar: float
    label: str               # LONG_ONLY, SHORT_ONLY, BOTH_GOOD, CHOP
    best_long_r: float       # best possible LONG outcome in R
    best_short_r: float      # best possible SHORT outcome in R
    optimal_dir: int         # +1 long, -1 short, 0 neither
    optimal_r: float         # best R achievable
    long_tp_bar: int = 0     # how many bars to TP (0 = didn't hit)
    short_tp_bar: int = 0
    long_sl_bar: int = 0     # how many bars to SL (0 = survived)
    short_sl_bar: int = 0


class OracleLabeler:
    """Pre-compute optimal trade outcomes for every H1 bar."""

    def __init__(self, config: BTCConfig = None,
                 sl_atr: float = 0.6, tp_atr: float = 1.4,
                 max_hold: int = 18):
        self.config = config or BTCConfig()
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.max_hold = max_hold
        self.engine = BTCFeatureEngine()

    def label(self, start: str, end: str,
              use_m1: bool = True) -> list:
        """Run oracle labeling over a date range. Returns list of OracleLabel."""
        ds = self._fetch_data(start, end, use_m1)
        labels = []

        n = len(ds['h1'])
        for i in range(self.config.seq_len_h1, n):
            entry_price = float(ds['h1']['close'].iloc[i])
            atr_pct = float(ds['h1_atr'][i])
            atr_dollar = atr_pct * entry_price
            sl_dist = self.sl_atr * atr_dollar
            tp_dist = self.tp_atr * atr_dollar
            ts = str(ds['h1']['timestamp'].iloc[i])[:19]

            # Simulate LONG and SHORT from this bar
            long_r, long_tp_bar, long_sl_bar = self._simulate_trade(
                ds, i, direction=1, entry_price=entry_price,
                sl_dist=sl_dist, tp_dist=tp_dist)
            short_r, short_tp_bar, short_sl_bar = self._simulate_trade(
                ds, i, direction=-1, entry_price=entry_price,
                sl_dist=sl_dist, tp_dist=tp_dist)

            # Classify
            label, optimal_dir, optimal_r = self._classify(long_r, short_r)

            labels.append(OracleLabel(
                timestamp=ts, close=entry_price, atr_dollar=atr_dollar,
                label=label, best_long_r=long_r, best_short_r=short_r,
                optimal_dir=optimal_dir, optimal_r=optimal_r,
                long_tp_bar=long_tp_bar, short_tp_bar=short_tp_bar,
                long_sl_bar=long_sl_bar, short_sl_bar=short_sl_bar,
            ))

        return labels

    def _simulate_trade(self, ds, start_h1_idx, direction, entry_price,
                        sl_dist, tp_dist):
        """Simulate a trade from start_h1_idx forward. Returns (best_r, tp_bar, sl_bar)."""
        n = len(ds['h1'])
        end_idx = min(start_h1_idx + self.max_hold, n)
        tp_price = entry_price + direction * tp_dist
        sl_price = entry_price - direction * sl_dist

        best_r = 0.0  # we can always "not enter" for 0R
        tp_bar = 0
        sl_bar = 0
        exit_price = entry_price
        exit_reason = 'time'

        # Check each subsequent bar for SL/TP hits
        for j in range(start_h1_idx + 1, end_idx):
            if ds['has_m1'] and ds['m1'] is not None:
                # Use M1 intrabar data for precise fills
                m1_indices = self._get_m1_bars_for_h1(ds, j)
                for m1_idx in m1_indices:
                    m1_hi = float(ds['m1']['high'].iloc[m1_idx])
                    m1_lo = float(ds['m1']['low'].iloc[m1_idx])

                    if direction == 1:  # LONG
                        if m1_lo <= sl_price:
                            exit_price = sl_price
                            exit_reason = 'sl'
                            sl_bar = j - start_h1_idx
                            break
                        if m1_hi >= tp_price:
                            exit_price = tp_price
                            exit_reason = 'tp'
                            tp_bar = j - start_h1_idx
                            break
                    else:  # SHORT
                        if m1_hi >= sl_price:
                            exit_price = sl_price
                            exit_reason = 'sl'
                            sl_bar = j - start_h1_idx
                            break
                        if m1_lo <= tp_price:
                            exit_price = tp_price
                            exit_reason = 'tp'
                            tp_bar = j - start_h1_idx
                            break
                if exit_reason != 'time':
                    break
            else:
                # M15/H1 bar-level check only (less precise)
                bar_hi = float(ds['h1']['high'].iloc[j])
                bar_lo = float(ds['h1']['low'].iloc[j])
                if direction == 1:
                    if bar_lo <= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; sl_bar = j - start_h1_idx; break
                    if bar_hi >= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; tp_bar = j - start_h1_idx; break
                else:
                    if bar_hi >= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; sl_bar = j - start_h1_idx; break
                    if bar_lo <= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; tp_bar = j - start_h1_idx; break

        # If no SL/TP hit, exit at last bar's close
        if exit_reason == 'time' and end_idx > start_h1_idx + 1:
            exit_price = float(ds['h1']['close'].iloc[end_idx - 1])

        # Compute R
        if direction == 1:
            pnl_dollar = (exit_price - entry_price)
        else:
            pnl_dollar = (entry_price - exit_price)
        best_r = pnl_dollar / max(sl_dist, 1e-9)

        return round(best_r, 4), tp_bar, sl_bar

    def _get_m1_bars_for_h1(self, ds, h1_idx):
        """Get M1 bar indices for a given H1 bar."""
        h1_ts = ds['h1']['timestamp'].iloc[h1_idx]
        h1_end = h1_ts + pd.Timedelta(hours=1)
        m1_ts = ds['m1']['timestamp']
        mask = (m1_ts > h1_ts) & (m1_ts <= h1_end)
        return np.where(mask)[0]

    @staticmethod
    def _classify(long_r, short_r):
        """Classify the bar based on optimal outcomes."""
        if long_r <= 0 and short_r <= 0:
            return ('CHOP', 0, 0.0)
        if long_r > 0.5 and short_r <= 0:
            return ('LONG_ONLY', 1, long_r)
        if short_r > 0.5 and long_r <= 0:
            return ('SHORT_ONLY', -1, short_r)
        if long_r > 0.5 and short_r > 0.5:
            return ('BOTH_GOOD', 1 if long_r > short_r else -1,
                    max(long_r, short_r))
        if long_r > 0 and long_r <= 0.5 and short_r <= 0:
            return ('LONG_GOOD', 1, long_r)
        if short_r > 0 and short_r <= 0.5 and long_r <= 0:
            return ('SHORT_GOOD', -1, short_r)
        # Both weakly positive
        return ('BOTH_GOOD', 1 if long_r > short_r else -1,
                max(long_r, short_r))

    def _fetch_data(self, start, end, use_m1):
        """Fetch H1 (+ M1) data, compute features and ATR."""
        import MetaTrader5 as mt5
        mt5.initialize()

        h1_rates = mt5.copy_rates_range(
            'BTCUSD', mt5.TIMEFRAME_H1,
            datetime.fromisoformat(start) - timedelta(days=7),
            datetime.fromisoformat(end) + timedelta(days=1))
        h1 = pd.DataFrame(h1_rates).rename(
            columns={'time': 'timestamp', 'tick_volume': 'volume'})
        h1['timestamp'] = pd.to_datetime(h1['timestamp'], unit='s', utc=True)
        h1 = h1.sort_values('timestamp').reset_index(drop=True)

        # Compute features and ATR
        feats = self.engine.compute(h1)
        h1_atr_pct = feats[:, 6]  # ATR as fraction of price

        m1 = None
        has_m1 = False
        if use_m1:
            try:
                m1_rates = mt5.copy_rates_range(
                    'BTCUSD', mt5.TIMEFRAME_M1,
                    datetime.fromisoformat(start) - timedelta(days=1),
                    datetime.fromisoformat(end) + timedelta(days=1))
                m1 = pd.DataFrame(m1_rates).rename(
                    columns={'time': 'timestamp', 'tick_volume': 'volume'})
                m1['timestamp'] = pd.to_datetime(m1['timestamp'], unit='s', utc=True)
                m1 = m1.sort_values('timestamp').reset_index(drop=True)
                has_m1 = len(m1) > 0
            except Exception:
                pass

        mt5.shutdown()

        return {
            'h1': h1, 'm1': m1, 'has_m1': has_m1,
            'h1_atr': h1_atr_pct,
        }

    def to_dataframe(self, labels: list) -> pd.DataFrame:
        """Convert labels to DataFrame."""
        return pd.DataFrame([{
            'timestamp': l.timestamp, 'close': l.close,
            'atr_dollar': l.atr_dollar, 'label': l.label,
            'best_long_r': l.best_long_r, 'best_short_r': l.best_short_r,
            'optimal_dir': l.optimal_dir, 'optimal_r': l.optimal_r,
        } for l in labels])

    def save(self, labels: list, path: str):
        """Save labels to CSV and pickle."""
        df = self.to_dataframe(labels)
        df.to_csv(path.replace('.pkl', '.csv'), index=False)
        with open(path, 'wb') as f:
            pickle.dump(labels, f)
        print('Saved {} labels to {}'.format(len(labels), path))

    @staticmethod
    def load(path: str) -> list:
        """Load cached labels."""
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Oracle Labeler — perfect foresight benchmark')
    p.add_argument('--from', dest='start', default='2026-01-01')
    p.add_argument('--to', dest='end', default='2026-05-25')
    p.add_argument('--sl', type=float, default=0.6)
    p.add_argument('--tp', type=float, default=1.4)
    p.add_argument('--max-hold', type=int, default=18)
    p.add_argument('--no-m1', action='store_true')
    p.add_argument('--save', default=None)
    args = p.parse_args()

    lab = OracleLabeler(sl_atr=args.sl, tp_atr=args.tp, max_hold=args.max_hold)
    labels = lab.label(args.start, args.end, use_m1=not args.no_m1)

    # Summary
    df = lab.to_dataframe(labels)
    print()
    print('=== ORACLE LABEL DISTRIBUTION ===')
    for lbl in ['LONG_ONLY', 'SHORT_ONLY', 'BOTH_GOOD', 'CHOP', 'LONG_GOOD', 'SHORT_GOOD']:
        count = (df['label'] == lbl).sum()
        if count > 0:
            print('  {}: {} bars ({:.1f}%)'.format(lbl, count, count/len(df)*100))

    print()
    print('Optimal direction split:')
    long_pct = (df['optimal_dir'] == 1).mean() * 100
    short_pct = (df['optimal_dir'] == -1).mean() * 100
    flat_pct = (df['optimal_dir'] == 0).mean() * 100
    print('  LONG:  {:.1f}%'.format(long_pct))
    print('  SHORT: {:.1f}%'.format(short_pct))
    print('  FLAT:  {:.1f}%'.format(flat_pct))

    print()
    print('Optimal R distribution:')
    opt_r = df[df['optimal_r'] > 0]['optimal_r']
    if len(opt_r) > 0:
        for p in [25, 50, 75, 90]:
            print('  P{}: {:+.3f}R'.format(p, np.percentile(opt_r, p)))

    if args.save:
        lab.save(labels, args.save)
