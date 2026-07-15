"""Analyze available R distribution to answer: wide SL or doom?"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from benchmark.oracle_m15 import M15Oracle
import __main__; __main__.M15Oracle = M15Oracle

with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle = pickle.load(f)
oracle_by_ts = {}
for ol in oracle: oracle_by_ts[ol.timestamp] = ol

df = pd.read_csv("D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
trades = df.to_dict("records")

matched = []
for t in trades:
    entry_str = str(t["entry_ts"])[:19]
    ol = oracle_by_ts.get(entry_str)
    if ol is None:
        for dm in [15, -15, 30, -30]:
            adj = t["entry_ts"] + pd.Timedelta(minutes=dm)
            ol = oracle_by_ts.get(str(adj)[:19])
            if ol: break
    if ol is None: continue

    mdir = 1 if t["direction"] == "LONG" else -1
    entry_r = ol.long_r if mdir == 1 else ol.short_r   # available in bot dir
    enemy_r = ol.short_r if mdir == 1 else ol.long_r    # available opposite dir

    matched.append({
        "pnl_r": t["pnl_r"], "pnl_d": t["pnl_dollar"],
        "exit": t["exit_reason"], "entry_r": entry_r,
        "enemy_r": enemy_r, "direction": t["direction"],
    })

df_m = pd.DataFrame(matched)

# Buckets
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"]
df_m["bucket"] = pd.cut(df_m["entry_r"], bins=bins, labels=labels)

print("AVAILABLE R DISTRIBUTION (in bot direction, within 18h)")
print("=" * 70)
hdr = f"{'Available R':>12s} {'Trades':>7s} {'Wins%':>7s} {'Bot PnL':>10s} {'Perfect':>10s} {'Enemy R':>8s}"
print(hdr)
print("-" * 65)

for bucket in labels:
    grp = df_m[df_m["bucket"] == bucket]
    if len(grp) == 0: continue
    n = len(grp)
    wr = (grp["pnl_r"] > 0).mean() * 100
    pnl_sum = grp["pnl_d"].sum()
    perfect = grp["entry_r"].sum() * grp["pnl_d"].std() / 100  # not useful
    print(f"{bucket:>12s} {n:>7d} {wr:>6.1f}% ${pnl_sum:>+9.1f}  ---      {grp['enemy_r'].mean():>+8.2f}")

print("-" * 65)
n_total = len(df_m)
wr_total = (df_m["pnl_r"] > 0).mean() * 100
print(f"{'TOTAL':>12s} {n_total:>7d} {wr_total:>6.1f}% ${df_m['pnl_d'].sum():>+9.1f}  ---      {df_m['enemy_r'].mean():>+8.2f}")

# Directional classification
print()
print("DIRECTIONAL BIAS (oracle label by price excursion)")
df_m["label"] = "neutral"
for i, row in df_m.iterrows():
    if row["entry_r"] > row["enemy_r"] * 1.5 and row["entry_r"] > 1.0:
        df_m.at[i, "label"] = "strong_right"
    elif row["enemy_r"] > row["entry_r"] * 1.5 and row["enemy_r"] > 1.0:
        df_m.at[i, "label"] = "strong_wrong"
    elif row["entry_r"] > 1.0 and row["enemy_r"] > 1.0:
        df_m.at[i, "label"] = "both_ways"
    elif row["entry_r"] < 0.5 and row["enemy_r"] < 0.5:
        df_m.at[i, "label"] = "dead"
    else:
        df_m.at[i, "label"] = "marginal"

for label, grp in df_m.groupby("label"):
    n = len(grp)
    pnl = grp["pnl_d"].sum()
    avg_entry_r = grp["entry_r"].mean()
    avg_enemy_r = grp["enemy_r"].mean()
    print(f"  {label:15s}: {n:>4d} trades  PnL=${pnl:>+9.1f}  entry={avg_entry_r:.2f} ATR  enemy={avg_enemy_r:.2f} ATR")

# The key question
print()
print("=" * 70)
print("WHAT HAPPENS WITH NO SL?")
print("=" * 70)

# Strong right direction: no SL captures more
strong = df_m[df_m["label"] == "strong_right"]
print(f"\nStrong right direction ({len(strong)} trades):")
print(f"  These move {strong['entry_r'].mean():.1f} ATR in our direction")
print(f"  Enemy only moves {strong['enemy_r'].mean():.1f} ATR")
print(f"  Current PnL: ${strong['pnl_d'].sum():+.1f}")
print(f"  With no SL: capture most of the {strong['entry_r'].sum():.0f} ATR available")
print(f"  Risk: enemy movement would cause {-strong['enemy_r'].sum():.0f} ATR drawdown total")

# Strong wrong direction: no SL = account death
wrong = df_m[df_m["label"] == "strong_wrong"]
print(f"\nStrong WRONG direction ({len(wrong)} trades):")
print(f"  These only move {wrong['entry_r'].mean():.1f} ATR our way")
print(f"  Enemy moves {wrong['enemy_r'].mean():.1f} ATR against us!")
print(f"  Current PnL: ${wrong['pnl_d'].sum():+.1f} (SL protected us)")
print(f"  With no SL: these would LOSE {-wrong['enemy_r'].sum():.0f} ATR total")
print(f"  ACCOUNT KILLER: enemy_r > 5 on {len(wrong[wrong['enemy_r']>5])} of these trades")

# Both ways: need wide SL but can profit
both = df_m[df_m["label"] == "both_ways"]
print(f"\nBoth ways ({len(both)} trades):")
print(f"  Move {both['entry_r'].mean():.1f} ATR our way AND {both['enemy_r'].mean():.1f} ATR against")
print(f"  Current PnL: ${both['pnl_d'].sum():+.1f}")
print(f"  Wide SL (>2 ATR) survives drawdown → captures upside")
print(f"  Tight SL (0.6 ATR) gets stopped at drawdown → misses upside")

# Net effect
print()
print("=" * 70)
print("NET EFFECT: Wide SL vs Current")
print("=" * 70)
# Approximate: if SL = 2.0 ATR
# strong_right: captures maybe 60% of entry_r (time stop exit)
# strong_wrong: loses entry_r (time stop, price never recovers)
# both_ways: captures 50% of entry_r (survives drawdown, time stop exit)
# marginal/dead: about breakeven

strong_capture = strong["entry_r"].sum() * 0.5
wrong_loss = -wrong["entry_r"].sum()  # lose what little we can get
both_capture = both["entry_r"].sum() * 0.4
estimated = strong_capture + wrong_loss + both_capture

print(f"  Strong right (50% capture):  +{strong_capture:.0f} R")
print(f"  Wrong dir (lose entry_r):     {wrong_loss:.0f} R")
print(f"  Both ways (40% capture):      +{both_capture:.0f} R")
print(f"  Estimated net:                {estimated:+.0f} R")
print(f"  Current net:                  {df_m['pnl_r'].sum():+.1f} R")

# But the real issue: without directional filter, wrong_dir kills you
print()
print("CONCLUSION:")
print(f"  Without SL: strong_right +{strong_capture:.0f}R  vs  strong_wrong {wrong_loss:.0f}R")
if abs(wrong_loss) > strong_capture:
    print(f"  WRONG DIRECTION LOSSES ({abs(wrong_loss):.0f}R) EXCEED RIGHT DIRECTION GAINS ({strong_capture:.0f}R)")
    print(f"  Removing SL alone would BLOW UP the account.")
    print(f"  Must pair wide SL with directional filter that blocks strong_wrong entries.")
else:
    print(f"  Wide SL could work IF wrong entries are caught early.")
