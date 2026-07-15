"""Investigate why speed 3.0+ trades are negative."""
import sys, pickle, pandas as pd, numpy as np
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from benchmark.oracle_m15 import M15Oracle
import __main__; __main__.M15Oracle = M15Oracle

with open('D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl', 'rb') as f:
    oracle = pickle.load(f)

df = pd.read_csv('D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv')
df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)

from backtest.data_manager import BacktestDataManager
from config_btc import BTCConfig
cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare('2026-01-01', '2026-05-25', use_cache=True)
m15 = ds.m15_df; h1 = ds.h1_df
m15_hi = m15['high'].values; m15_lo = m15['low'].values; m15_cl = m15['close'].values
h1_atr_pct = ds.h1_features[:, 6]

def get_atr(idx):
    ts = pd.Timestamp(m15['timestamp'].iloc[idx])
    h1_ts_naive = h1['timestamp'].dt.tz_convert(None)
    h1_idx = int((h1_ts_naive <= ts.tz_convert(None)).sum() - 1)
    if h1_idx >= 0 and h1_idx < len(h1_atr_pct):
        return h1_atr_pct[h1_idx] * m15_cl[idx]
    return m15_cl[idx] * 0.005

# Match all trades and compute speed
all_trades = []
for _, t in df.iterrows():
    ts = pd.Timestamp(t['entry_ts'])
    if ts.tz is not None: ts = ts.tz_convert(None)
    minute = (ts.minute // 15) * 15
    key = ts.replace(minute=minute, second=0, microsecond=0)

    m15_ts_naive = m15['timestamp'].dt.tz_convert(None)
    matches = m15[m15_ts_naive == key]
    if len(matches) == 0:
        for dm in [15, -15, 30, -30]:
            adj_key = key + pd.Timedelta(minutes=dm)
            matches = m15[m15_ts_naive == adj_key]
            if len(matches) > 0: break
    if len(matches) == 0: continue

    idx = matches.index[0]
    entry = m15_cl[idx]; atr = get_atr(idx)
    if atr < 1: atr = entry * 0.005

    mdir = 1 if t['direction'] == 'LONG' else -1
    end = min(idx + 72, len(m15))
    best_exc = 0.0; bars_to_1atr = 72
    # Also track MAE (max adverse)
    mae = 0.0  # positive = adverse

    for j in range(idx, end):
        hi = m15_hi[j]; lo = m15_lo[j]
        if mdir == 1:
            exc = (hi - entry) / atr
            adv = (entry - lo) / atr  # adverse
        else:
            exc = (entry - lo) / atr
            adv = (hi - entry) / atr
        best_exc = max(best_exc, exc)
        mae = max(mae, adv)
        if bars_to_1atr == 72 and exc >= 1.0:
            bars_to_1atr = j - idx

    speed = best_exc / (1 + bars_to_1atr) if best_exc > 0 else 0

    # Check per-bar data for drawdown timing
    import json
    per_bar_raw = t.get('per_bar', '[]')
    if isinstance(per_bar_raw, str):
        try: per_bar = json.loads(per_bar_raw.replace("'", '"'))
        except: per_bar = []
    else:
        per_bar = per_bar_raw if isinstance(per_bar_raw, list) else []

    mfe_at_bar_0 = per_bar[0].get('mfe', 0) if len(per_bar) > 0 else 0
    mae_at_bar_0 = per_bar[0].get('mae', 0) if len(per_bar) > 0 else 0

    all_trades.append({
        'speed': speed, 'pnl_d': t['pnl_dollar'], 'pnl_r': t['pnl_r'],
        'exit': t['exit_reason'], 'direction': t['direction'],
        'best_exc': best_exc, 'mae': mae, 'bars_to_1atr': bars_to_1atr,
        'bars_held': t['bars_held'],
        'mfe_bar0': mfe_at_bar_0, 'mae_bar0': mae_at_bar_0,
    })

atdf = pd.DataFrame(all_trades)

# Show speed buckets with mae context
bins = [0, 0.1, 0.2, 0.5, 1.0, 3.0, 100]
labels = ['0-0.1', '0.1-0.2', '0.2-0.5', '0.5-1.0', '1.0-3.0', '3.0+']
atdf['bucket'] = pd.cut(atdf['speed'], bins=bins, labels=labels)

print("Speed Bucket Analysis (with MAE):")
print(f"{'Speed':>10s} {'N':>5s} {'PnL':>10s} {'WR':>6s} {'BestExc':>8s} {'MAE':>8s} {'MAE/Best':>8s} {'Mfe@0':>8s} {'Mae@0':>8s}")
print("-" * 90)
for bucket in labels:
    grp = atdf[atdf['bucket'] == bucket]
    if len(grp) == 0: continue
    n = len(grp)
    wr = (grp['pnl_r'] > 0).mean() * 100
    pnl = grp['pnl_d'].sum()
    avg_best = grp['best_exc'].mean()
    avg_mae = grp['mae'].mean()
    mae_ratio = avg_mae / max(avg_best, 0.01)
    mfe0 = grp['mfe_bar0'].mean()
    mae0 = grp['mae_bar0'].mean()
    print(f"{bucket:>10s} {n:>5d} ${pnl:>+9.1f} {wr:>5.1f}% {avg_best:>+8.2f} {avg_mae:>+8.2f} {mae_ratio:>8.2f} {mfe0:>+8.3f} {mae0:>+8.3f}")

# Drill into 3.0+ trades
fast = atdf[atdf['speed'] > 3.0]
print(f"\n--- 3.0+ Speed Trades ({len(fast)}) ---")
for _, t in fast.iterrows():
    print(f"  {t['direction']:5s} speed={t['speed']:.1f} best={t['best_exc']:+.2f} mae={t['mae']:+.2f} "
          f"bars_to_1atr={int(t['bars_to_1atr'])} pnl=${t['pnl_d']:+.1f} ({t['pnl_r']:+.3f}R) "
          f"exit={t['exit']} held={int(t['bars_held'])}h mae@0={t['mae_bar0']:+.3f} mfe@0={t['mfe_bar0']:+.3f}")

# Key insight: what's the MAE at bar 0 for fast trades?
print(f"\nKey stat: MAE at bar 0 for speed 3.0+ trades: {fast['mae_bar0'].mean():+.3f}")
print(f"          MAE at bar 0 for speed 0-0.1 trades: {atdf[atdf['speed']<0.1]['mae_bar0'].mean():+.3f}")
print(f"          MAE overall for speed 3.0+ trades: {fast['mae'].mean():+.3f}")
print(f"          MAE overall for all other trades: {atdf[atdf['speed']<=3.0]['mae'].mean():+.3f}")
