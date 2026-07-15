"""
M15 Oracle — pure price benchmark (no SL/TP).

For every M15 bar, looks ahead up to max_hold bars and measures the maximum
favorable price excursion for both LONG and SHORT directions.

R is measured in ATR units: long_r = max(high - entry) / atr within window.
No stop-loss, no take-profit — just "how far did price move?"

The oracle answers: "if you entered at this M15 bar, what was the best possible
price you could have exited at within the max_hold window?"
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine


@dataclass
class M15Oracle:
    timestamp: str
    close: float
    atr_dollar: float
    long_r: float      # max favorable excursion LONG  (ATR units)
    short_r: float     # max favorable excursion SHORT (ATR units)
    label: str         # LONG_WIN, SHORT_WIN, BOTH_WIN, CHOP


class M15OracleLabeler:
    """Label every M15 bar with pure price excursion (no SL/TP simulation)."""

    def __init__(self, max_hold_m15=72):
        self.max_hold_m15 = max_hold_m15  # 72 M15 bars = 18 hours
        self.engine = BTCFeatureEngine()

    def label(self, start, end, use_m1=True):
        """Run oracle on M15 bars."""
        ds = self._fetch(start, end, use_m1)
        m15 = ds['m15']
        h1 = ds['h1']
        n_m15 = len(m15)

        # Compute ATR from H1 and interpolate to M15
        h1_atr_pct = ds['h1_atr']
        m15_atr = np.zeros(n_m15)
        for i in range(len(m15)):
            ts = m15['timestamp'].iloc[i]
            h1_idx = int((h1['timestamp'] <= ts).sum() - 1)
            if h1_idx >= 0 and h1_idx < len(h1_atr_pct):
                m15_atr[i] = h1_atr_pct[h1_idx] * float(m15['close'].iloc[i])

        labels = []
        print('Labeling {} M15 bars...'.format(n_m15))
        for i in range(n_m15):
            entry = float(m15['close'].iloc[i])
            atr = m15_atr[i]
            if atr < 1: atr = entry * 0.005  # fallback

            long_r, short_r = self._sim(ds, i, entry, atr)

            lbl = self._classify(long_r, short_r)
            labels.append(M15Oracle(
                timestamp=str(m15['timestamp'].iloc[i])[:19],
                close=entry, atr_dollar=atr,
                long_r=long_r, short_r=short_r, label=lbl,
            ))

            if (i + 1) % 2000 == 0:
                print('  {} / {} ({:.0f}%)'.format(i + 1, n_m15, (i + 1) / n_m15 * 100))

        return labels

    def _sim(self, ds, start_m15, entry, atr):
        """Measure max favorable excursion in both directions within max_hold window.

        Returns (long_r, short_r) in ATR units.
        long_r = max(high - entry) / atr  for all M1 bars in the lookahead
        short_r = max(entry - low) / atr  for all M1 bars in the lookahead
        """
        n = len(ds['m15'])
        end = min(start_m15 + self.max_hold_m15, n)
        long_r = 0.0
        short_r = 0.0

        if ds['has_m1'] and ds['m1'] is not None:
            m1_ts = ds['m1']['timestamp']
            m1_hi = ds['m1']['high'].values
            m1_lo = ds['m1']['low'].values

            for j in range(start_m15, end):
                m15_ts = ds['m15']['timestamp'].iloc[j]
                m15_start = m15_ts - pd.Timedelta(minutes=15)
                m1_mask = (m1_ts > m15_start) & (m1_ts <= m15_ts)
                m1_idx = np.where(m1_mask)[0]

                for mi in m1_idx:
                    hi = float(m1_hi[mi])
                    lo = float(m1_lo[mi])
                    long_r = max(long_r, (hi - entry) / atr)
                    short_r = max(short_r, (entry - lo) / atr)
        else:
            # H1 bar-level fallback (less precise)
            for j in range(start_m15, end):
                h1_idx = int((ds['h1']['timestamp'] <= ds['m15']['timestamp'].iloc[j]).sum() - 1)
                if h1_idx < 0:
                    continue
                hi = float(ds['h1']['high'].iloc[h1_idx])
                lo = float(ds['h1']['low'].iloc[h1_idx])
                long_r = max(long_r, (hi - entry) / atr)
                short_r = max(short_r, (entry - lo) / atr)

        return round(long_r, 4), round(short_r, 4)

    @staticmethod
    def _classify(long_r, short_r):
        """Classify based on max price excursion (ATR units).

        Thresholds calibrated for BTC H1/M15:
          min_move = 1.0 ATR  — minimum excursion worth trading
          ratio    = 1.5x    — one direction must dominate to be directional
        """
        min_move = 1.0
        ratio = 1.5

        if long_r < min_move and short_r < min_move:
            return 'CHOP'
        if long_r > short_r * ratio and long_r >= min_move:
            return 'LONG_WIN'
        if short_r > long_r * ratio and short_r >= min_move:
            return 'SHORT_WIN'
        return 'BOTH_WIN'

    def _fetch(self, start, end, use_m1):
        import MetaTrader5 as mt5
        mt5.initialize()
        sd = datetime.fromisoformat(start) - timedelta(days=7)
        ed = datetime.fromisoformat(end) + timedelta(days=1)

        h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, sd, ed)
        m15r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15, sd, ed)
        h1 = pd.DataFrame(h1r).rename(columns={'time':'timestamp','tick_volume':'volume'})
        m15 = pd.DataFrame(m15r).rename(columns={'time':'timestamp','tick_volume':'volume'})
        h1['timestamp'] = pd.to_datetime(h1['timestamp'], unit='s', utc=True)
        m15['timestamp'] = pd.to_datetime(m15['timestamp'], unit='s', utc=True)
        h1 = h1.sort_values('timestamp').reset_index(drop=True)
        m15 = m15.sort_values('timestamp').reset_index(drop=True)

        feats = self.engine.compute(h1)
        h1_atr = feats[:, 6]

        m1 = None; has_m1 = False
        if use_m1:
            try:
                m1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M1, sd, ed)
                m1 = pd.DataFrame(m1r).rename(columns={'time':'timestamp','tick_volume':'volume'})
                m1['timestamp'] = pd.to_datetime(m1['timestamp'], unit='s', utc=True)
                m1 = m1.sort_values('timestamp').reset_index(drop=True)
                has_m1 = len(m1) > 0
            except Exception: pass

        mt5.shutdown()
        return {'h1': h1, 'm15': m15, 'm1': m1, 'has_m1': has_m1, 'h1_atr': h1_atr}

    def save(self, labels, path):
        with open(path, 'wb') as f:
            pickle.dump(labels, f)
        print('Saved {} M15 oracle labels to {}'.format(len(labels), path))

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='start', default='2026-05-01')
    p.add_argument('--to', dest='end', default='2026-05-25')
    p.add_argument('--save', default=None)
    args = p.parse_args()

    lab = M15OracleLabeler()
    labels = lab.label(args.start, args.end, use_m1=True)

    # Summary
    print()
    counts = {}
    for l in labels:
        counts[l.label] = counts.get(l.label, 0) + 1
    total = len(labels)
    print('M15 Oracle distribution:')
    for lbl in ['LONG_WIN', 'SHORT_WIN', 'BOTH_WIN', 'CHOP']:
        c = counts.get(lbl, 0)
        print('  {}: {} bars ({:.1f}%)'.format(lbl, c, c/total*100 if total else 0))
