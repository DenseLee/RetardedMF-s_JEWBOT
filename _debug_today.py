"""Debug today's BTC bot performance: oracle labels + bot trades + missed opportunities."""
import sys, os, json, pickle
import numpy as np, pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from benchmark.oracle_m15 import M15OracleLabeler, M15Oracle
from data.feature_engine_btc import BTCFeatureEngine

cfg = BTCConfig()

# ── 1. Generate oracle labels for today ──
print("=" * 80)
print("1. GENERATING ORACLE LABELS FOR TODAY (2026-05-26)")
print("=" * 80)

lab = M15OracleLabeler(max_hold_m15=72)
labels = lab.label("2026-05-26", "2026-05-27", use_m1=True)

# Index by M15 timestamp
oracle_by_ts = {}
for ol in labels:
    oracle_by_ts[ol.timestamp] = ol

# Distribution
counts = {}
for l in labels:
    counts[l.label] = counts.get(l.label, 0) + 1
total = len(labels)
print(f"\nOracle distribution for today ({total} M15 bars):")
for lbl in ['LONG_WIN', 'SHORT_WIN', 'BOTH_WIN', 'CHOP']:
    c = counts.get(lbl, 0)
    print(f"  {lbl}: {c} bars ({c/total*100:.1f}%)")

# ── 2. Load live bot trades ──
print(f"\n{'='*80}")
print("2. BOT TRADES TODAY")
print("=" * 80)

log_path = os.path.join(cfg.log_dir, "h1_eval_BTCBot.jsonl")
def load_live_trades(log_path):
    """Extract trades from the JSONL eval log."""
    trades = []
    current_entry = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("event")
            if ev == "enter":
                current_entry = {
                    "entry_ts": rec["ts"],
                    "direction": "LONG" if rec["direction"] == 1 else "SHORT",
                    "entry_price": rec["price"],
                    "lots": rec.get("lots", 0),
                    "sl": rec.get("sl", 0),
                    "tp": rec.get("tp", 0),
                    "confidence": rec.get("confidence", 0),
                }
            elif ev == "exit" and current_entry is not None:
                current_entry["exit_ts"] = rec["ts"]
                current_entry["exit_price"] = rec["price"]
                current_entry["pnl_dollar"] = rec.get("pnl_dollar", 0)
                current_entry["pnl_r"] = rec.get("pnl_r", 0)
                current_entry["mfe_r"] = rec.get("mfe_r", 0)
                current_entry["mae_r"] = rec.get("mae_r", 0)
                current_entry["bars_held"] = rec.get("bars_held", 0)
                current_entry["exit_reason"] = rec.get("reason", "")
                trades.append(current_entry)
                current_entry = None
    return trades

live_trades = load_live_trades(log_path)
print(f"Loaded {len(live_trades)} trades from eval log")

for i, t in enumerate(live_trades):
    print(f"\n  Trade {i+1}: {t['direction']} @ {t['entry_price']:.1f}")
    print(f"    Entry: {t['entry_ts'][:19]}")
    print(f"    Exit:  {t.get('exit_ts', '?')[:19]} ({t.get('bars_held', '?')} bars) — {t.get('exit_reason', '?')}")
    print(f"    PnL: ${t.get('pnl_dollar', 0):+.2f} ({t.get('pnl_r', 0):+.3f}R) | MFE: {t.get('mfe_r', 0):+.3f}R | MAE: {t.get('mae_r', 0):+.3f}R")

# ── 3. Cross-reference with oracle ──
print(f"\n{'='*80}")
print("3. ORACLE vs BOT COMPARISON")
print("=" * 80)

for i, t in enumerate(live_trades):
    entry_str = t["entry_ts"]
    # Parse the entry timestamp
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%d %H:%M:%S.%f+00:00"]:
        try:
            entry_dt = datetime.strptime(entry_str, fmt).replace(tzinfo=None)
            break
        except ValueError:
            continue
    else:
        print(f"  Trade {i+1}: CANNOT PARSE {entry_str}")
        continue

    # Find matching oracle bar (closest M15)
    entry_m15 = entry_dt.replace(minute=(entry_dt.minute // 15) * 15, second=0, microsecond=0)
    entry_key = entry_m15.strftime("%Y-%m-%d %H:%M:%S")
    ol = oracle_by_ts.get(entry_key)

    if ol is None:
        # Try adjacent 15-min windows
        for offset in [15, -15, 30, -30]:
            adj = entry_m15 + timedelta(minutes=offset)
            ol = oracle_by_ts.get(adj.strftime("%Y-%m-%d %H:%M:%S"))
            if ol: break

    if ol is None:
        print(f"  Trade {i+1}: NO ORACLE MATCH for {entry_key}")
        continue

    mdir = t["direction"]
    oracle_r = ol.long_r if mdir == "LONG" else ol.short_r
    enemy_r = ol.short_r if mdir == "LONG" else ol.long_r
    model_r = t["pnl_r"]
    mfe_r = t["mfe_r"]

    # Verdict
    if ol.label == "CHOP":
        verdict = "CHOP — no clear direction, shouldn't have entered"
    elif oracle_r < 0.5:
        verdict = f"WRONG DIR — oracle says {ol.label}, {mdir} only had {oracle_r:.2f}R"
    elif model_r < 0.3 and oracle_r > 1.0:
        verdict = f"EXITED TOO EARLY — oracle had {oracle_r:.2f}R, bot got {model_r:+.2f}R"
    elif model_r < 0:
        verdict = f"LOSS but correct dir — oracle had {oracle_r:.2f}R, lost {model_r:+.2f}R"
    else:
        verdict = f"GOOD — captured {model_r:+.2f}R of {oracle_r:.2f}R available"

    print(f"\n  Trade {i+1}: {mdir} @ {t['entry_price']:.1f}")
    print(f"    Oracle bar:  {ol.timestamp} | Label: {ol.label}")
    print(f"    Long avail:  {ol.long_r:.3f}R | Short avail: {ol.short_r:.3f}R")
    print(f"    Bot result:  {model_r:+.3f}R (MFE: {mfe_r:+.3f}R)")
    print(f"    VERDICT:     {verdict}")

# ── 4. Missed opportunities (best entry bars today) ──
print(f"\n{'='*80}")
print("4. TOP MISSED OPPORTUNITIES TODAY (best oracle bars with no bot trade)")
print("=" * 80)

# Find H1 bars where bot DID enter
bot_entry_m15_keys = set()
for t in live_trades:
    entry_str = t["entry_ts"]
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%d %H:%M:%S.%f+00:00"]:
        try:
            entry_dt = datetime.strptime(entry_str, fmt).replace(tzinfo=None)
            break
        except ValueError:
            continue
    for offset in [0, 15, -15, 30, -30]:
        adj = entry_dt.replace(minute=(entry_dt.minute // 15) * 15, second=0, microsecond=0) + timedelta(minutes=offset)
        bot_entry_m15_keys.add(adj.strftime("%Y-%m-%d %H:%M:%S"))

# Find best untraded bars
untraded = []
for ol in labels:
    if ol.timestamp not in bot_entry_m15_keys and ol.label != "CHOP":
        best_r = max(ol.long_r, ol.short_r)
        best_dir = "LONG" if ol.long_r >= ol.short_r else "SHORT"
        untraded.append((ol.timestamp, best_dir, best_r, ol.label, ol.long_r, ol.short_r, ol.close))

untraded.sort(key=lambda x: x[2], reverse=True)

print(f"\nTop 15 untraded oracle bars today:")
print(f"{'Time':22s} {'Dir':6s} {'Best R':>7s} {'Label':12s} {'Long R':>7s} {'Short R':>7s} {'Price':>10s}")
print("-" * 90)
for ts, d, r, lbl, lr, sr, price in untraded[:15]:
    print(f"{ts:22s} {d:6s} {r:+.3f}R  {lbl:12s} {lr:+.3f}R  {sr:+.3f}R  {price:>10.1f}")

# ── 5. Regime timeline vs oracle ──
print(f"\n{'='*80}")
print("5. TIMELINE: Oracle signal vs Gate decision")
print("=" * 80)

# Load H1 eval log for today
h1_evals = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "atr" in rec and "gate" in rec:  # H1 eval entry
            h1_evals.append(rec)

# Align H1 evals with oracle
print(f"\n{'H1 Time':22s} {'Price':>8s} {'Model':>14s} {'Gate':>10s} {'Oracle':>12s} {'Best Long':>9s} {'Best Short':>9s}")
print("-" * 95)

for ev in h1_evals:
    ts = ev["ts"]
    # Parse timestamp
    for fmt in ["%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%d %H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S.%f+00:00"]:
        try:
            bar_dt = datetime.strptime(ts, fmt)
            break
        except ValueError:
            continue
    else:
        continue

    bar_m15 = bar_dt.strftime("%Y-%m-%d %H:%M:%S")
    ol = oracle_by_ts.get(bar_m15)

    final_regime = ev["final"]["regime"]
    final_conf = ev["final"]["confidence"]
    gate_signal = ev["gate"]["signal"]
    gate_dir = ev["gate"]["direction"]
    gate_reason = ev["gate"]["reason"]
    price = ev["price"]

    if ol:
        oracle_lbl = ol.label
        best_l = ol.long_r
        best_s = ol.short_r
    else:
        oracle_lbl = "?"
        best_l = 0.0
        best_s = 0.0

    dir_str = f"{'LONG' if gate_dir == 1 else 'SHORT' if gate_dir == -1 else '-'}"
    signal_str = f"{'SIGNAL ' + dir_str if gate_signal else 'BLOCKED'}"
    print(f"{str(bar_dt)[:19]:22s} {price:>8.1f} {final_regime + '(' + str(round(final_conf,2)) + ')':>14s} {signal_str:>10s} {oracle_lbl:>12s} {best_l:+.3f}R  {best_s:+.3f}R")

print("\nDone.")
