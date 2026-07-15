"""Find entry signals that separate noise from winners, and micro-wins from death-zone losses."""
import pandas as pd, numpy as np, json
from collections import defaultdict

df = pd.read_csv("logs/btc_all_trades.csv")

# ── Define groups ──
noise = df[(df["pnl_r"] <= 0) & (df["mfe_peak"] <= 0.25)]           # losses that barely moved
good_wins = df[(df["pnl_r"] > 0.50)]                                  # decent wins
micro_wins = df[(df["pnl_r"] > 0) & (df["pnl_r"] <= 0.25)]           # cut short
death_zone = df[(df["pnl_r"] <= 0) & (df["mfe_peak"] >= 0.25) & (df["mfe_peak"] <= 0.50)]  # had MFE, died
full_losses = df[(df["pnl_r"] <= 0) & (df["mfe_peak"] < 0.25)]       # all noise losses

print(f"Noise entries (MFE<0.25R, loss):    {len(noise)}")
print(f"Good wins (PnL>0.50R):              {len(good_wins)}")
print(f"Micro-wins (PnL 0-0.25R):           {len(micro_wins)}")
print(f"Death-zone (MFE 0.25-0.50R, loss):  {len(death_zone)}")
print(f"Full losses (MFE<0.25R):             {len(full_losses)}")

# ═══════════════════════════════════════════════════
# 1. NOISE vs GOOD WINS — Entry confirmation signals
# ═══════════════════════════════════════════════════
ENTRY_FEATURES = [
    "m15_ema_distance_r", "m15_3bar_momentum_r", "m15_atr_ratio",
    "m15_5min_realized_vol", "m15_entry_confidence", "m15_direction_bias",
    "bars_listened", "entry_hour_utc", "h1_regime_confidence", "h1_regime",
    "confirmation_method", "with_h1_trend", "direction"
]

print("\n" + "=" * 70)
print("1. NOISE vs GOOD WINS — Entry Signal Comparison")
print("=" * 70)

for col in ENTRY_FEATURES:
    if col in ["h1_regime", "confirmation_method", "direction", "with_h1_trend"]:
        # Categorical
        print(f"\n  {col}:")
        for group_name, group_df in [("NOISE", noise), ("GOOD_WINS", good_wins)]:
            counts = group_df[col].value_counts()
            total = len(group_df)
            parts = [f"{v}: {c} ({c/total*100:.0f}%)" for v, c in counts.items()]
            print(f"    {group_name:<12s} {', '.join(parts)}")
    else:
        n_vals = noise[col].dropna().values.astype(float)
        g_vals = good_wins[col].dropna().values.astype(float)
        if len(n_vals) < 5 or len(g_vals) < 5:
            continue
        print(f"\n  {col}:")
        print(f"    {'NOISE':<12s} mean={np.mean(n_vals):+.4f}  median={np.median(n_vals):+.4f}  "
              f"std={np.std(n_vals):.4f}  [P25={np.percentile(n_vals,25):+.4f}, P75={np.percentile(n_vals,75):+.4f}]")
        print(f"    {'GOOD_WINS':<12s} mean={np.mean(g_vals):+.4f}  median={np.median(g_vals):+.4f}  "
              f"std={np.std(g_vals):.4f}  [P25={np.percentile(g_vals,25):+.4f}, P75={np.percentile(g_vals,75):+.4f}]")
        # Effect size
        diff = np.mean(g_vals) - np.mean(n_vals)
        pooled_std = np.sqrt((np.var(n_vals) + np.var(g_vals)) / 2)
        cohens_d = diff / max(pooled_std, 0.001)
        print(f"    {'Diff:':<12s} {diff:+.4f}  Cohen's d = {cohens_d:.2f} {'***' if abs(cohens_d) > 0.5 else '*' if abs(cohens_d) > 0.2 else ''}")

# ═══════════════════════════════════════════════════
# 2. Entry hour breakdown (noise vs wins)
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("2. ENTRY HOUR — Noise rate by hour")
print("=" * 70)
print(f"  {'Hour':>5s} {'Total':>6s} {'Noise':>6s} {'Noise%':>7s} {'GoodWin%':>9s}")
for hr in sorted(df["entry_hour_utc"].dropna().unique()):
    hr_df = df[df["entry_hour_utc"] == hr]
    hr_noise = len(hr_df[(hr_df["pnl_r"] <= 0) & (hr_df["mfe_peak"] <= 0.25)])
    hr_good = len(hr_df[hr_df["pnl_r"] > 0.50])
    print(f"  {int(hr):5d} {len(hr_df):6d} {hr_noise:6d} {hr_noise/len(hr_df)*100:6.1f}% {hr_good/len(hr_df)*100:8.1f}%")

# ═══════════════════════════════════════════════════
# 3. NOISE vs GOOD WINS — Per-bar MFE trajectory
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("3. PER-BAR MFE TRAJECTORY — First 8 bars")
print("=" * 70)

def avg_mfe_at_bar(group_df, bar_idx):
    """Compute average MFE at a specific bar index across trades."""
    mfes = []
    for _, row in group_df.iterrows():
        try:
            bars = json.loads(row["per_bar"])
            if bar_idx < len(bars):
                mfes.append(bars[bar_idx]["mfe"])
        except (json.JSONDecodeError, KeyError):
            pass
    return np.mean(mfes) if mfes else 0, len(mfes)

for group_name, group_df in [("NOISE", noise), ("GOOD_WINS", good_wins),
                              ("MICRO_WINS", micro_wins), ("DEATH_ZONE", death_zone)]:
    traj = []
    for bar in range(9):
        avg, n = avg_mfe_at_bar(group_df, bar)
        traj.append(f"bar{bar}:{avg:+.3f}R")
    print(f"  {group_name:<15s} ({len(group_df):3d} trades): {'  '.join(traj)}")

# ═══════════════════════════════════════════════════
# 4. MICRO-WINS vs DEATH-ZONE — What separates them?
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("4. MICRO-WINS vs DEATH-ZONE — Entry signal differences")
print("=" * 70)
print(f"  Micro-wins: {len(micro_wins)} trades, avg PnL={np.mean(micro_wins['pnl_r']):+.3f}R, avg MFE={np.mean(micro_wins['mfe_peak']):+.3f}R")
print(f"  Death-zone: {len(death_zone)} trades, avg PnL={np.mean(death_zone['pnl_r']):+.3f}R, avg MFE={np.mean(death_zone['mfe_peak']):+.3f}R")

for col in ENTRY_FEATURES:
    if col in ["h1_regime", "confirmation_method", "direction", "with_h1_trend"]:
        print(f"\n  {col}:")
        for group_name, group_df in [("MICRO_WINS", micro_wins), ("DEATH_ZONE", death_zone)]:
            counts = group_df[col].value_counts()
            total = len(group_df)
            parts = [f"{v}: {c} ({c/total*100:.0f}%)" for v, c in counts.items()]
            print(f"    {group_name:<12s} {', '.join(parts)}")
    else:
        m_vals = micro_wins[col].dropna().values.astype(float)
        d_vals = death_zone[col].dropna().values.astype(float)
        if len(m_vals) < 5 or len(d_vals) < 5:
            continue
        diff = np.mean(m_vals) - np.mean(d_vals)
        pooled_std = np.sqrt((np.var(m_vals) + np.var(d_vals)) / 2)
        cohens_d = diff / max(pooled_std, 0.001)
        print(f"\n  {col}:")
        print(f"    {'MICRO_WINS':<12s} mean={np.mean(m_vals):+.4f}  median={np.median(m_vals):+.4f}  std={np.std(m_vals):.4f}")
        print(f"    {'DEATH_ZONE':<12s} mean={np.mean(d_vals):+.4f}  median={np.median(d_vals):+.4f}  std={np.std(d_vals):.4f}")
        print(f"    {'Diff:':<12s} {diff:+.4f}  Cohen's d={cohens_d:.2f} {'***' if abs(cohens_d)>0.5 else '*' if abs(cohens_d)>0.2 else ''}")

# ═══════════════════════════════════════════════════
# 5. Per-bar MFE distribution by outcome (critical for early exit signal)
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("5. BAR 3 MFE vs FINAL OUTCOME — Does bar-3 MFE predict recovery?")
print("=" * 70)

def get_bar_mfe(row, bar_idx):
    try:
        bars = json.loads(row["per_bar"])
        if bar_idx < len(bars):
            return bars[bar_idx]["mfe"]
    except:
        pass
    return None

# For trades that had bar-3 MFE in certain ranges, what was the final outcome?
df["bar3_mfe"] = df.apply(lambda r: get_bar_mfe(r, 3), axis=1)
df["bar3_mae"] = df.apply(lambda r: get_bar_mfe(r, 3), axis=1)

# Actually let me do this properly
for bar in [1, 2, 3, 5]:
    col_name = f"bar{bar}_mfe"
    vals = []
    for _, row in df.iterrows():
        v = get_bar_mfe(row, bar)
        vals.append(v)
    df[col_name] = vals

print(f"\n  Bar-3 MFE buckets → final outcome:")
print(f"  {'Bar3 MFE':>12s} {'Count':>6s} {'Wins':>5s} {'Losses':>6s} {'WR':>7s} {'AvgPnL':>8s} {'AvgMFE':>8s}")
for lo, hi in [(-9, -0.5), (-0.5, -0.2), (-0.2, -0.05), (-0.05, 0.05),
                (0.05, 0.2), (0.2, 0.5), (0.5, 9)]:
    valid = df[df["bar3_mfe"].notna()]
    bucket = valid[(valid["bar3_mfe"] >= lo) & (valid["bar3_mfe"] < hi)]
    if len(bucket) < 3:
        continue
    w = bucket[bucket["pnl_r"] > 0]
    wr = len(w) / len(bucket) * 100
    avg_pnl = np.mean(bucket["pnl_r"])
    avg_mfe = np.mean(bucket["mfe_peak"])
    print(f"  {f'{lo:+.0f} to {hi:+.0f}R':>12s} {len(bucket):6d} {len(w):5d} {len(bucket)-len(w):6d} {wr:6.1f}% {avg_pnl:+7.3f}R {avg_mfe:+7.3f}R")

# ═══════════════════════════════════════════════════
# 6. Bars_listened distribution — does waiting longer help?
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("6. BARS LISTENED vs WIN RATE")
print("=" * 70)
print(f"  {'Listened':>9s} {'Count':>6s} {'WR':>7s} {'AvgPnL':>8s} {'Noise%':>7s} {'GoodWin%':>9s}")
for bl in sorted(df["bars_listened"].dropna().unique()):
    bl_df = df[df["bars_listened"] == bl]
    if len(bl_df) < 5: continue
    w = bl_df[bl_df["pnl_r"] > 0]
    wr = len(w) / len(bl_df) * 100
    n = len(bl_df[(bl_df["pnl_r"] <= 0) & (bl_df["mfe_peak"] <= 0.25)])
    g = len(bl_df[bl_df["pnl_r"] > 0.50])
    print(f"  {int(bl):9d} {len(bl_df):6d} {wr:6.1f}% {np.mean(bl_df['pnl_r']):+7.3f}R {n/len(bl_df)*100:6.1f}% {g/len(bl_df)*100:8.1f}%")

# ═══════════════════════════════════════════════════
# 7. REGIME — noise rate by regime
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("7. REGIME vs OUTCOME")
print("=" * 70)
for regime in sorted(df["h1_regime"].dropna().unique()):
    r_df = df[df["h1_regime"] == regime]
    if len(r_df) < 5: continue
    n = len(r_df[(r_df["pnl_r"] <= 0) & (r_df["mfe_peak"] <= 0.25)])
    g = len(r_df[r_df["pnl_r"] > 0.50])
    m = len(r_df[(r_df["pnl_r"] > 0) & (r_df["pnl_r"] <= 0.25)])
    d = len(r_df[(r_df["pnl_r"] <= 0) & (r_df["mfe_peak"] >= 0.25) & (r_df["mfe_peak"] <= 0.50)])
    print(f"  {regime:>15s}: {len(r_df):4d} trades, WR={len(r_df[r_df['pnl_r']>0])/len(r_df)*100:.0f}%, "
          f"noise={n} ({n/len(r_df)*100:.0f}%), micro={m} ({m/len(r_df)*100:.0f}%), "
          f"death={d} ({d/len(r_df)*100:.0f}%), good={g} ({g/len(r_df)*100:.0f}%)")

# ═══════════════════════════════════════════════════
# 8. Combined filter simulation
# ═══════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("8. COMBINED FILTERS — What combo removes most noise while keeping wins?")
print("=" * 70)

# Test various filter combinations
filters = {
    "BASELINE": None,
    "no_hour_21_23": lambda d: ~d["entry_hour_utc"].isin([21, 22, 23]),
    "no_hour_2_11_18_19_21_22_23": lambda d: ~d["entry_hour_utc"].isin([2, 11, 18, 19, 21, 22, 23]),
    "ema_dist_positive_long": lambda d: ((d["direction"] == "LONG") & (d["m15_ema_distance_r"] > -0.2)) |
                                         ((d["direction"] == "SHORT") & (d["m15_ema_distance_r"] < 0.2)),
    "bars_listened_ge_3": lambda d: d["bars_listened"] >= 3,
    "bars_listened_ge_4": lambda d: d["bars_listened"] >= 4,
    "atr_ratio_lt_1.2": lambda d: d["m15_atr_ratio"] < 1.2,
    "realized_vol_lt_0.005": lambda d: d["m15_5min_realized_vol"] < 0.005,
    "with_h1_trend": lambda d: d["with_h1_trend"] == True,
    "regime_not_transition": lambda d: d["h1_regime"] != "TRANSITION",
}

# Test single filters
for name, fn in filters.items():
    if fn is None:
        filtered = df
    else:
        filtered = df[fn(df)]
    if len(filtered) < 10: continue
    w = filtered[filtered["pnl_r"] > 0]
    l = filtered[filtered["pnl_r"] <= 0]
    wr = len(w) / len(filtered) * 100
    tg = sum(w["pnl_r"]); tl = abs(sum(l["pnl_r"]))
    pf = tg / max(tl, 0.001)
    pnl = sum(filtered["pnl_dollar"])
    noise_removed = len(noise) - len(filtered[(filtered["pnl_r"] <= 0) & (filtered["mfe_peak"] <= 0.25)])
    good_kept = len(filtered[filtered["pnl_r"] > 0.50])
    print(f"  {name:<35s} N={len(filtered):4d} WR={wr:.1f}% PF={pf:.2f} PnL=${pnl:,.0f}  "
          f"noise-removed={noise_removed}/{len(noise)}  good-kept={good_kept}/{len(good_wins)}")
