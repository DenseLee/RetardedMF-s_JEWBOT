"""Timing-aware oracle: measures HOW QUICKLY the move develops, not just how far."""
import sys, pickle, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from benchmark.oracle_m15 import M15Oracle
import __main__; __main__.M15Oracle = M15Oracle

with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle = pickle.load(f)
oracle_by_ts = {}
for ol in oracle: oracle_by_ts[ol.timestamp] = ol

# Load trades
df = pd.read_csv("D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

# ===== Build M15 OHLC arrays from the oracle labels =====
timestamps = []
closes = []
long_rs = []
short_rs = []

for ol in oracle:
    timestamps.append(pd.Timestamp(ol.timestamp, tz="UTC"))
    closes.append(ol.close)
    long_rs.append(ol.long_r)
    short_rs.append(ol.short_r)

# Sort by timestamp
idx = np.argsort(timestamps)
timestamps = [timestamps[i] for i in idx]
closes_arr = np.array([closes[i] for i in idx])
long_rs_arr = np.array([long_rs[i] for i in idx])
short_rs_arr = np.array([short_rs[i] for i in idx])

n = len(timestamps)
print(f"Oracle bars: {n}")

# ===== For each bar, compute time-to-target excursion =====
# "How many M15 bars until price moves +1.0 ATR (long) or -1.0 ATR (short)?"
# We need OHLC to check this. Oracle only has close prices.
# Re-derive from the backtest cached data.

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from backtest.data_manager import BacktestDataManager
from config_btc import BTCConfig

cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-05-25", use_cache=True)

m15 = ds.m15_df
m15_hi = m15["high"].values
m15_lo = m15["low"].values
m15_cl = m15["close"].values
m15_ts = m15["timestamp"].values

# Build oracle lookup at M15 resolution
oracle_map = {}
for ol in oracle:
    oracle_map[str(ol.timestamp)[:19]] = ol

print(f"M15 bars: {len(m15)}")

# For speed: precompute H1 ATR
h1_atr_pct = ds.h1_features[:, 6]
h1 = ds.h1_df
def get_atr(m15_idx):
    ts = pd.Timestamp(m15_ts[m15_idx])
    if ts.tz is None: ts = ts.tz_localize("UTC")
    h1_idx = int((h1["timestamp"] <= ts).sum() - 1)
    if h1_idx >= 0 and h1_idx < len(h1_atr_pct):
        return h1_atr_pct[h1_idx] * m15_cl[m15_idx]
    return m15_cl[m15_idx] * 0.005

# ===== Compute timing metrics for every M15 bar =====
max_hold = 72  # M15 bars = 18h
targets = [0.5, 1.0, 1.5, 2.0]  # ATR targets

print("Computing timing metrics for all M15 bars...")
timing_data = []  # list of dicts per bar

sample_every = max(1, len(m15) // 1000)

for i in range(len(m15)):
    entry = m15_cl[i]
    atr = get_atr(i)
    if atr < 1: atr = entry * 0.005

    end = min(i + max_hold, len(m15))
    best_long = 0.0
    best_short = 0.0

    # Time to reach each target ATR excursion
    time_to_target_long = {t: max_hold for t in targets}
    time_to_target_short = {t: max_hold for t in targets}

    for j in range(i, end):
        hi = m15_hi[j]; lo = m15_lo[j]
        long_exc = (hi - entry) / atr
        short_exc = (entry - lo) / atr
        best_long = max(best_long, long_exc)
        best_short = max(best_short, short_exc)

        for t in targets:
            if time_to_target_long[t] == max_hold and long_exc >= t:
                time_to_target_long[t] = j - i
            if time_to_target_short[t] == max_hold and short_exc >= t:
                time_to_target_short[t] = j - i

    # Speed scores: higher = better timing
    # Score = max_excursion / (1 + time_to_1atr)
    # Fast mover: 5.0 / (1+2) = 1.67
    # Slow mover: 5.0 / (1+50) = 0.10
    long_speed = best_long / (1 + time_to_target_long[1.0]) if best_long > 0 else 0
    short_speed = best_short / (1 + time_to_target_short[1.0]) if best_short > 0 else 0

    timing_data.append({
        "i": i,
        "ts": pd.Timestamp(m15_ts[i]),  # store as Timestamp, not string
        "entry": entry, "atr": atr,
        "best_long": best_long, "best_short": best_short,
        "bars_to_1atr_long": time_to_target_long[1.0],
        "bars_to_1atr_short": time_to_target_short[1.0],
        "long_speed": round(long_speed, 4),
        "short_speed": round(short_speed, 4),
    })

    if i % sample_every == 0:
        pct = i / len(m15) * 100
        print(f"  {i}/{len(m15)} ({pct:.0f}%)")

# Convert to DataFrame
# Convert to DataFrame and dict keyed by Timestamp (rounded to M15)
tdf = pd.DataFrame(timing_data)
timing_map = {}
for _, row in tdf.iterrows():
    ts = row["ts"]
    if ts.tz is not None: ts = ts.tz_convert(None)  # strip tz for matching
    minute = (ts.minute // 15) * 15
    key = ts.replace(minute=minute, second=0, microsecond=0)
    timing_map[key] = row

print(f"\nTiming data: {len(tdf)} bars")

# ===== Classify entry quality =====
def classify_entry_timing(bars_to_1atr, best_excursion):
    """Classify entry timing quality."""
    if best_excursion < 1.0:
        return "no_move"  # never reaches 1 ATR
    if bars_to_1atr <= 2:
        return "impulse"  # move starts within 30 min
    elif bars_to_1atr <= 6:
        return "early"    # move starts within 1.5h
    elif bars_to_1atr <= 18:
        return "delayed"  # move starts within 4.5h
    else:
        return "late"     # move starts after 4.5h+

# ===== Compare bot entries vs all bars =====
print("\n" + "="*90)
print("TIMING QUALITY: Bot entries vs All bars")
print("="*90)

bot_timings = []
for _, t in df.iterrows():
    ts = pd.Timestamp(t["entry_ts"])
    if ts.tz is not None: ts = ts.tz_convert(None)  # strip tz for matching
    minute = (ts.minute // 15) * 15
    key = ts.replace(minute=minute, second=0, microsecond=0)
    row = timing_map.get(key)
    if row is None:
        for dm in [15, -15, 30, -30]:
            adj = ts + pd.Timedelta(minutes=dm)
            adj_min = (adj.minute // 15) * 15
            adj_key = adj.replace(minute=adj_min, second=0, microsecond=0)
            row = timing_map.get(adj_key)
            if row is not None: break
    if row is None: continue

    mdir = 1 if t["direction"] == "LONG" else -1
    bars_to_1atr = row["bars_to_1atr_long"] if mdir == 1 else row["bars_to_1atr_short"]
    best_exc = row["best_long"] if mdir == 1 else row["best_short"]
    speed = row["long_speed"] if mdir == 1 else row["short_speed"]
    timing_label = classify_entry_timing(bars_to_1atr, best_exc)

    bot_timings.append({
        "ts": str(key),
        "direction": t["direction"],
        "bars_to_1atr": bars_to_1atr,
        "best_exc": best_exc,
        "speed": speed,
        "timing": timing_label,
        "pnl_r": t["pnl_r"],
        "pnl_d": t["pnl_dollar"],
        "exit": t["exit_reason"],
        "confirmation": t["confirmation_method"],
    })

bdf = pd.DataFrame(bot_timings)
print(f"Bot entries matched: {len(bdf)}")
if len(bdf) == 0:
    print("No matches found — exiting")
    exit()

# Distribution
print(f"\n{'Timing':>12s} {'Bot_Trades':>10s} {'Bot%':>7s} {'Avg_Exc':>8s} {'Avg_Speed':>10s} {'PnL':>10s}")
print("-" * 65)
for label in ["impulse", "early", "delayed", "late", "no_move"]:
    grp = bdf[bdf["timing"] == label]
    n = len(grp)
    if n == 0: continue
    avg_exc = grp["best_exc"].mean()
    avg_speed = grp["speed"].mean()
    pnl = grp["pnl_d"].sum()
    pnl_r = grp["pnl_r"].mean()
    print(f"{label:>12s} {n:>10d} {n/len(bdf)*100:>6.1f}% {avg_exc:>+8.2f} {avg_speed:>10.4f} ${pnl:>+9.1f}")

# All bars distribution for comparison
print(f"\nAll M15 bars timing distribution:")
all_timings = []
for _, row in tdf.iterrows():
    l_timing = classify_entry_timing(row["bars_to_1atr_long"], row["best_long"])
    s_timing = classify_entry_timing(row["bars_to_1atr_short"], row["best_short"])
    all_timings.append(l_timing)
    all_timings.append(s_timing)

from collections import Counter
all_counts = Counter(all_timings)
total_all = len(all_timings)
for label in ["impulse", "early", "delayed", "late", "no_move"]:
    c = all_counts.get(label, 0)
    print(f"  {label:>12s}: {c:>6d} ({c/total_all*100:.1f}%)")

# ===== Speed score vs PnL =====
print(f"\n--- Speed Score vs Trade PnL ---")
bdf["speed_bucket"] = pd.cut(bdf["speed"], bins=[0, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0],
                              labels=["0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1.0", "1.0-3.0", "3.0+"])
for bucket, grp in bdf.groupby("speed_bucket", observed=False):
    if len(grp) == 0: continue
    n = len(grp)
    wr = (grp["pnl_r"] > 0).mean() * 100
    pnl = grp["pnl_d"].sum()
    avg_r = grp["pnl_r"].mean()
    print(f"  Speed {bucket:>8s}: n={n:>4d}  WR={wr:.0f}%  PnL=${pnl:>+9.1f}  avgR={avg_r:>+.3f}")

# ===== Key question: is the bot entering at optimal timing? =====
print("\n" + "="*90)
print("IS THE BOT ENTERING AT OPTIMAL TIMING?")
print("="*90)

# Compare bot speed vs random bar speed
bot_long_speed = bdf[bdf["direction"] == "LONG"]["speed"].mean()
bot_short_speed = bdf[bdf["direction"] == "SHORT"]["speed"].mean()
all_long_speed = tdf["long_speed"].mean()
all_short_speed = tdf["short_speed"].mean()

print(f"  Bot LONG avg speed:  {bot_long_speed:.4f}")
print(f"  All LONG avg speed:  {all_long_speed:.4f}")
print(f"  Bot SHORT avg speed: {bot_short_speed:.4f}")
print(f"  All SHORT avg speed: {all_short_speed:.4f}")

# Impulse rate
bot_impulse_pct = (bdf["timing"] == "impulse").mean() * 100
all_impulse_pct_long = (tdf["bars_to_1atr_long"] <= 2).mean() * 100
all_impulse_pct_short = (tdf["bars_to_1atr_short"] <= 2).mean() * 100

print(f"\n  Bot entries at impulse bars (0-2 bars before move): {bot_impulse_pct:.1f}%")
print(f"  All bars that are impulse (long):  {all_impulse_pct_long:.1f}%")
print(f"  All bars that are impulse (short): {all_impulse_pct_short:.1f}%")

# ===== Oracle with timing: score = max_excursion * (1 - bars_to_move/max_hold) =====
print("\n" + "="*90)
print("TIMING-WEIGHTED ORACLE SCORE")
print("="*90)

# Score = best_excursion * (1 - bars_to_1atr / max_hold)
# Impulse bar (2 bars, 5 ATR): 5 * (1 - 2/72) = 4.86
# Late bar (50 bars, 5 ATR):   5 * (1 - 50/72) = 1.53
bdf["timing_score"] = bdf.apply(
    lambda r: r["best_exc"] * (1 - min(r["bars_to_1atr"], 72) / 72), axis=1)

print("\nBot trades by timing score:")
bdf["score_bucket"] = pd.cut(bdf["timing_score"], bins=[0, 0.5, 1.0, 2.0, 3.0, 5.0, 20.0],
                              labels=["0-0.5", "0.5-1", "1-2", "2-3", "3-5", "5+"])
for bucket, grp in bdf.groupby("score_bucket", observed=False):
    if len(grp) == 0: continue
    n = len(grp)
    wr = (grp["pnl_r"] > 0).mean() * 100
    pnl = grp["pnl_d"].sum()
    print(f"  Score {bucket:>8s}: n={n:>4d}  WR={wr:.0f}%  PnL=${pnl:>+9.1f}")

# Compare: bot's avg timing score vs all bars
all_long_scores = tdf["best_long"] * (1 - tdf["bars_to_1atr_long"].clip(0, 72) / 72)
all_short_scores = tdf["best_short"] * (1 - tdf["bars_to_1atr_short"].clip(0, 72) / 72)
print(f"\n  Bot avg timing score (long):  {bdf[bdf['direction']=='LONG']['timing_score'].mean():.3f}")
print(f"  All avg timing score (long):  {all_long_scores.mean():.3f}")
print(f"  Bot avg timing score (short): {bdf[bdf['direction']=='SHORT']['timing_score'].mean():.3f}")
print(f"  All avg timing score (short): {all_short_scores.mean():.3f}")
