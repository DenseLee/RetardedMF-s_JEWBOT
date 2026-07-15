"""Generate YTD oracle labels using backtest cached data (avoids MT5 conflict)."""
import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from backtest.data_manager import BacktestDataManager
from benchmark.oracle_m15 import M15Oracle, M15OracleLabeler
from config_btc import BTCConfig
from collections import Counter
import __main__; __main__.M15Oracle = M15Oracle

cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-05-25", use_cache=True)
print(f"Using cached data: {ds.n_h1} H1, {ds.n_m15} M15, {ds.n_m1} M1")

lab = M15OracleLabeler()
m15 = ds.m15_df; h1 = ds.h1_df; n_m15 = len(m15)

# ATR from H1 features
h1_atr_pct = ds.h1_features[:, 6]
m15_atr = np.zeros(n_m15)
for i in range(n_m15):
    ts = m15['timestamp'].iloc[i]
    h1_idx = int((h1['timestamp'] <= ts).sum() - 1)
    if h1_idx >= 0 and h1_idx < len(h1_atr_pct):
        m15_atr[i] = h1_atr_pct[h1_idx] * float(m15['close'].iloc[i])

labels = []
print(f'Labeling {n_m15} M15 bars...')
for i in range(n_m15):
    entry = float(m15['close'].iloc[i])
    atr = m15_atr[i]
    if atr < 1: atr = entry * 0.005

    data = {'m15': m15, 'h1': h1, 'm1': ds.m1_df, 'has_m1': ds.has_m1}
    long_r, short_r = lab._sim(data, i, entry, atr)

    lbl = lab._classify(long_r, short_r)
    labels.append(M15Oracle(
        timestamp=str(m15['timestamp'].iloc[i])[:19],
        close=entry, atr_dollar=atr,
        long_r=long_r, short_r=short_r, label=lbl,
    ))

    if (i + 1) % 2000 == 0:
        print(f'  {i+1} / {n_m15} ({(i+1)/n_m15*100:.0f}%)')

lab.save(labels, "D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl")

counts = Counter(l.label for l in labels)
total = len(labels)
print(f'\nYTD Oracle ({total} M15 bars, pure price benchmark):')
for lbl in ['LONG_WIN', 'SHORT_WIN', 'BOTH_WIN', 'CHOP']:
    c = counts.get(lbl, 0)
    print(f'  {lbl}: {c} ({c/total*100:.1f}%)')

# Show R distribution
long_rs = [l.long_r for l in labels]
short_rs = [l.short_r for l in labels]
print(f'\nR distribution (ATR units):')
for p in [50, 75, 90, 95, 99]:
    print(f'  LONG  P{p}: {np.percentile(long_rs, p):.2f}')
    print(f'  SHORT P{p}: {np.percentile(short_rs, p):.2f}')
