"""Compare live bot trades against M15 oracle benchmark."""
import sys, os, pickle, numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from benchmark.oracle_m15 import M15Oracle, M15OracleLabeler
import __main__; __main__.M15Oracle = M15Oracle

# Load oracle
print("Loading oracle...")
with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle_labels = pickle.load(f)
print(f"  {len(oracle_labels)} M15 oracle labels loaded")

# Index oracle by timestamp (M15 resolution)
oracle_by_ts = {}
for ol in oracle_labels:
    oracle_by_ts[ol.timestamp] = ol

# Load live bot trades
df = pd.read_csv("D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
trades = df.to_dict("records")
print(f"  {len(trades)} live bot trades loaded")

# Compare each trade
results = []
unmatched = 0
for t in trades:
    entry_str = str(t["entry_ts"])[:19]
    ol = oracle_by_ts.get(entry_str)

    # Try ±1 M15 bar
    if ol is None:
        for dm in [15, -15, 30, -30]:
            adj = t["entry_ts"] + timedelta(minutes=dm)
            ol = oracle_by_ts.get(str(adj)[:19])
            if ol: break

    if ol is None:
        unmatched += 1
        continue

    mdir = 1 if t["direction"] == "LONG" else -1
    mr = t["pnl_r"]
    mpnl = t["pnl_dollar"]

    # What oracle says about this direction
    oracle_r = ol.long_r if mdir == 1 else ol.short_r
    oracle_label = ol.label
    oracle_enemy_r = ol.short_r if mdir == 1 else ol.long_r

    # Classify trade quality
    if oracle_label == "CHOP":
        verdict = "CHOP_ENTRY"
        lost_r = abs(mr) if mr < 0 else 0
    elif (mdir == 1 and ol.long_r <= 0) or (mdir == -1 and ol.short_r <= 0):
        verdict = "WRONG_DIR"
        lost_r = abs(oracle_enemy_r) if oracle_enemy_r > 0 else abs(mr)
    else:
        # Correct direction
        if mr < 0.3 and oracle_r > 1.0:
            verdict = "EXITED_EARLY"
            lost_r = oracle_r - max(mr, 0)
        elif mr < 0.5 and oracle_r > 0.5:
            verdict = "LEFT_MONEY"
            lost_r = oracle_r - max(mr, 0)
        elif mr >= 0.5:
            verdict = "GOOD"
            lost_r = oracle_r - mr
        else:
            verdict = "SL_HIT_CORRECT_DIR"
            lost_r = oracle_r - max(mr, 0)

    results.append({
        "ts": entry_str,
        "dir": t["direction"],
        "model_r": mr,
        "model_pnl": mpnl,
        "oracle_label": oracle_label,
        "oracle_r": round(oracle_r, 3),
        "oracle_enemy_r": round(oracle_enemy_r, 3),
        "verdict": verdict,
        "lost_r": round(lost_r, 3),
        "exit": t["exit_reason"],
        "regime": t.get("h1_regime", ""),
        "confirmation": t.get("confirmation_method", ""),
        "entry_hour": t.get("entry_hour_utc", 0),
        "m15_conf": t.get("m15_entry_confidence", 0),
    })

print(f"\nMatched: {len(results)} / {len(trades)} (unmatched: {unmatched})")

# Summary
df_r = pd.DataFrame(results)
n = len(df_r)

print(f"\n{'='*70}")
print(f"BOT vs ORACLE BENCHMARK")
print(f"{'='*70}")

# Verdict distribution
print(f"\nTrade Quality Distribution:")
verdicts = df_r.groupby("verdict").agg(
    count=("verdict", "count"),
    pct=("verdict", lambda x: len(x)/n*100),
    total_pnl=("model_pnl", "sum"),
    avg_r=("model_r", "mean"),
    total_lost_r=("lost_r", "sum"),
).sort_values("count", ascending=False)
print(verdicts.to_string())

# Overall stats
total_profit = df_r[df_r["model_r"] > 0]["model_r"].sum()
total_loss = abs(df_r[df_r["model_r"] <= 0]["model_r"].sum())
total_oracle_r = df_r["oracle_r"].sum()
total_lost = df_r["lost_r"].sum()
captured_pct = (total_profit / total_oracle_r * 100) if total_oracle_r > 0 else 0

print(f"\nOverall:")
print(f"  Oracle total available R: {total_oracle_r:+.1f}")
print(f"  Bot captured R:           {total_profit - total_loss:+.1f}")
print(f"  R left on table:          {total_lost:+.1f}")
print(f"  Capture rate:             {captured_pct:.1f}%")

# By confirmation method
print(f"\nBy Confirmation Method:")
conf = df_r.groupby("confirmation").agg(
    trades=("verdict", "count"),
    pnl=("model_pnl", "sum"),
    avg_r=("model_r", "mean"),
    oracle_avg_r=("oracle_r", "mean"),
    lost_r=("lost_r", "sum"),
    early_exits=("verdict", lambda x: (x.isin(["EXITED_EARLY","LEFT_MONEY"])).sum()),
    wrong_dir=("verdict", lambda x: (x == "WRONG_DIR").sum()),
    chop=("verdict", lambda x: (x == "CHOP_ENTRY").sum()),
).round(2)
print(conf.to_string())

# By oracle label (what did we enter on?)
print(f"\nBy Oracle Label at Entry:")
olabel = df_r.groupby("oracle_label").agg(
    trades=("verdict", "count"),
    pnl=("model_pnl", "sum"),
    avg_r=("model_r", "mean"),
    oracle_avg_r=("oracle_r", "mean"),
    lost_r=("lost_r", "sum"),
).round(2)
print(olabel.to_string())

# By regime
print(f"\nBy H1 Regime:")
by_regime = df_r.groupby("regime").agg(
    trades=("verdict", "count"),
    pnl=("model_pnl", "sum"),
    avg_r=("model_r", "mean"),
    oracle_avg_r=("oracle_r", "mean"),
    lost_r=("lost_r", "sum"),
    early_pct=("verdict", lambda x: (x.isin(["EXITED_EARLY","LEFT_MONEY"])).mean()*100),
).round(2)
print(by_regime.to_string())

# By entry hour (UTC)
print(f"\nBy Entry Hour (UTC):")
df_r["hour"] = df_r["entry_hour"].astype(int)
by_hour = df_r.groupby("hour").agg(
    trades=("verdict", "count"),
    pnl=("model_pnl", "sum"),
    avg_r=("model_r", "mean"),
    oracle_avg_r=("oracle_r", "mean"),
    early_pct=("verdict", lambda x: (x.isin(["EXITED_EARLY","LEFT_MONEY"])).mean()*100),
).round(2)
print(by_hour.to_string())

# Common conditions for losses
print(f"\n{'='*70}")
print(f"COMMON CONDITIONS IN LOST/MISSED PROFIT")
print(f"{'='*70}")

bad_trades = df_r[df_r["verdict"].isin(["WRONG_DIR", "CHOP_ENTRY", "EXITED_EARLY", "LEFT_MONEY", "SL_HIT_CORRECT_DIR"])]
good_trades = df_r[df_r["verdict"] == "GOOD"]

print(f"\nBad trades ({len(bad_trades)}):")
print(f"  Avg M15 confidence: {bad_trades['m15_conf'].mean():.4f}")
print(f"  Avg oracle R:       {bad_trades['oracle_r'].mean():.3f}")
print(f"  Avg model R:        {bad_trades['model_r'].mean():.3f}")
print(f"  Top exit reasons:")
for reason, count in bad_trades.groupby("exit")["verdict"].count().sort_values(ascending=False).head(5).items():
    print(f"    {reason}: {count}")
print(f"  Top regimes:")
for regime, count in bad_trades.groupby("regime")["verdict"].count().sort_values(ascending=False).head(5).items():
    print(f"    {regime}: {count}")

print(f"\nGood trades ({len(good_trades)}):")
print(f"  Avg M15 confidence: {good_trades['m15_conf'].mean():.4f}")
print(f"  Avg oracle R:       {good_trades['oracle_r'].mean():.3f}")
print(f"  Avg model R:        {good_trades['model_r'].mean():.3f}")

# Specific examples
print(f"\nTop 10 Most R Left on Table:")
top_lost = df_r.nlargest(10, "lost_r")
for _, t in top_lost.iterrows():
    print(f"  {t['ts'][:19]} {t['dir']:5s} bot={t['model_r']:+.3f}R oracle={t['oracle_r']:+.3f}R "
          f"lost={t['lost_r']:+.3f}R {t['verdict']:20s} exit={t['exit']} regime={t['regime']}")
