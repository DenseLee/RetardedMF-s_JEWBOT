"""Find optimal bar to detect wrong-direction trades."""
import sys, pickle, json, numpy as np, pandas as pd
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from benchmark.oracle_m15 import M15Oracle
import __main__; __main__.M15Oracle = M15Oracle

# Load oracle
with open("D:/FiananceBot/BTC_BOT/benchmark/ytd_oracle.pkl", "rb") as f:
    oracle = pickle.load(f)
oracle_by_ts = {}
for ol in oracle: oracle_by_ts[ol.timestamp] = ol

# Load trades
df = pd.read_csv("D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

# Match and classify
trades = []
for _, t in df.iterrows():
    entry_str = str(t["entry_ts"])[:19]
    ol = oracle_by_ts.get(entry_str)
    if ol is None:
        for dm in [15, -15, 30, -30]:
            adj = t["entry_ts"] + pd.Timedelta(minutes=dm)
            ol = oracle_by_ts.get(str(adj)[:19])
            if ol: break
    if ol is None: continue

    mdir = 1 if t["direction"] == "LONG" else -1
    entry_r = ol.long_r if mdir == 1 else ol.short_r
    enemy_r = ol.short_r if mdir == 1 else ol.long_r

    if entry_r > enemy_r * 1.5 and entry_r > 1.0:
        label = "strong_right"
    elif enemy_r > entry_r * 1.5 and enemy_r > 1.0:
        label = "strong_wrong"
    elif entry_r > 1.0 and enemy_r > 1.0:
        label = "both_ways"
    else:
        continue

    # Parse per-bar data
    per_bar_raw = t.get("per_bar", "[]")
    if isinstance(per_bar_raw, str):
        try:
            per_bar = json.loads(per_bar_raw.replace("'", '"'))
        except:
            per_bar = []
    else:
        per_bar = per_bar_raw if isinstance(per_bar_raw, list) else []

    trades.append({
        "label": label, "entry_r": entry_r, "enemy_r": enemy_r,
        "pnl_r": t["pnl_r"], "pnl_d": t["pnl_dollar"],
        "exit": t["exit_reason"], "bars_held": t["bars_held"],
        "per_bar": per_bar,
    })

print(f"Trades with per-bar data: {len(trades)}")
right = [t for t in trades if t["label"] == "strong_right"]
wrong = [t for t in trades if t["label"] == "strong_wrong"]
both = [t for t in trades if t["label"] == "both_ways"]
print(f"strong_right: {len(right)}  strong_wrong: {len(wrong)}  both_ways: {len(both)}")

# ===== Analyze per-bar MFE and MAE by label =====
max_bars = 18  # max hold

# Collect per-bar MFE/MAE for each group
def collect_per_bar(trade_list, max_bars):
    """Collect MFE and MAE at each bar offset for a group of trades."""
    mfe_by_bar = {b: [] for b in range(max_bars + 1)}
    mae_by_bar = {b: [] for b in range(max_bars + 1)}
    unrealized_by_bar = {b: [] for b in range(max_bars + 1)}

    for t in trade_list:
        pb = t["per_bar"]
        if not pb: continue

        for bar_data in pb:
            b = min(bar_data.get("bar", 0), max_bars)
            mfe = bar_data.get("mfe", 0)
            mae = bar_data.get("mae", 0)
            price = bar_data.get("price", 0)

            mfe_by_bar[b].append(mfe)
            mae_by_bar[b].append(mae)

    return mfe_by_bar, mae_by_bar

right_mfe, right_mae = collect_per_bar(right, max_bars)
wrong_mfe, wrong_mae = collect_per_bar(wrong, max_bars)
both_mfe, both_mae = collect_per_bar(both, max_bars)

# ===== Print MFE/MAE by bar =====
print("\n" + "="*95)
print("PER-BAR MFE COMPARISON (how much profit was seen at each bar)")
print("="*95)
print(f"{'Bar':>4s} {'Right_MFE':>10s} {'Wrong_MFE':>10s} {'Diff':>10s} {'Right_MAE':>10s} {'Wrong_MAE':>10s} {'Diff_MAE':>10s} {'Separation':>10s}")
print("-"*85)

separation_by_bar = []
for b in range(max_bars + 1):
    r_mfe = np.mean(right_mfe[b]) if right_mfe[b] else 0
    w_mfe = np.mean(wrong_mfe[b]) if wrong_mfe[b] else 0
    r_mae = np.mean(right_mae[b]) if right_mae[b] else 0
    w_mae = np.mean(wrong_mae[b]) if wrong_mae[b] else 0

    mfe_diff = r_mfe - w_mfe
    mae_diff = r_mae - w_mae

    # Pooled std for separation metric
    r_std = np.std(right_mfe[b]) if len(right_mfe[b]) > 1 else 0.001
    w_std = np.std(wrong_mfe[b]) if len(wrong_mfe[b]) > 1 else 0.001
    pooled = np.sqrt((r_std**2 + w_std**2) / 2)
    separation = mfe_diff / max(pooled, 0.001)

    separation_by_bar.append((b, separation, mfe_diff, r_mfe, w_mfe, r_mae, w_mae, len(right_mfe[b]), len(wrong_mfe[b])))

    if b <= 12:
        print(f"{b:>4d} {r_mfe:>+10.4f} {w_mfe:>+10.4f} {mfe_diff:>+10.4f} {r_mae:>+10.4f} {w_mae:>+10.4f} {mae_diff:>+10.4f} {separation:>10.3f}")

# ===== Find optimal detection bar =====
print("\n" + "="*95)
print("OPTIMAL WRONG-DETECTION BAR (highest MFE separation)")
print("="*95)

separation_by_bar.sort(key=lambda x: x[1], reverse=True)
print(f"{'Bar':>4s} {'Separation':>12s} {'MFE_Diff':>10s} {'Right_MFE':>10s} {'Wrong_MFE':>10s} {'N_right':>8s} {'N_wrong':>8s}")
print("-"*75)
for b, sep, diff, rm, wm, rmae, wmae, nr, nw in separation_by_bar[:10]:
    print(f"{b:>4d} {sep:>12.4f} {diff:>+10.4f} {rm:>+10.4f} {wm:>+10.4f} {nr:>8d} {nw:>8d}")

# ===== Practical detection rules =====
print("\n" + "="*95)
print("PRACTICAL DETECTION RULES: Can we detect wrong at bar N?")
print("="*95)

for detect_bar in [1, 2, 3, 4, 5, 6, 8, 10]:
    # At bar N, look at MFE and MAE
    # Rule 1: If MFE < threshold at bar N, it's likely wrong
    # Rule 2: If MAE < threshold at bar N, it's likely wrong

    all_mfe = []
    all_labels = []
    for t in trades:
        pb = t["per_bar"]
        if not pb or len(pb) <= detect_bar: continue
        bar_data = pb[detect_bar]
        all_mfe.append(bar_data.get("mfe", 0))
        all_labels.append(1 if t["label"] == "strong_right" else (0 if t["label"] == "strong_wrong" else -1))

    if len(all_mfe) < 50: continue

    # Try different MFE thresholds
    print(f"\n--- Bar {detect_bar} ---")

    # What if we close if MFE < 0?
    for threshold in [0, 0.1, 0.2, 0.3, -0.1, -0.2]:
        close_if_below = threshold
        kept = [(m, l) for m, l in zip(all_mfe, all_labels) if l >= 0 and m >= close_if_below]
        closed = [(m, l) for m, l in zip(all_mfe, all_labels) if l >= 0 and m < close_if_below]

        if len(kept) < 20 or len(closed) < 5: continue

        kept_right = sum(1 for _, l in kept if l == 1)
        kept_wrong = sum(1 for _, l in kept if l == 0)
        closed_right = sum(1 for _, l in closed if l == 1)
        closed_wrong = sum(1 for _, l in closed if l == 0)

        kept_right_pct = kept_right / len(kept) * 100
        closed_right_pct = closed_right / len(closed) * 100 if closed else 0

        wrong_caught = closed_wrong / (closed_wrong + kept_wrong) * 100 if (closed_wrong + kept_wrong) > 0 else 0
        right_sacrificed = closed_right / (closed_right + kept_right) * 100 if (closed_right + kept_right) > 0 else 0

        if wrong_caught > 20:  # Only show if it catches at least 20% of wrong
            print(f"  Close if MFE < {threshold:+.1f}:  keep {len(kept)} ({kept_right_pct:.0f}% right)  "
                  f"close {len(closed)} ({closed_right_pct:.0f}% right sacrificed)  "
                  f"wrong_caught={wrong_caught:.0f}%  right_sacrificed={right_sacrificed:.0f}%")

# ===== MAE-based detection =====
print(f"\n--- MAE-based detection ---")
for detect_bar in [1, 2, 3, 4, 5, 6]:
    all_mae = []
    all_labels_mae = []
    for t in trades:
        pb = t["per_bar"]
        if not pb or len(pb) <= detect_bar: continue
        bar_data = pb[detect_bar]
        all_mae.append(bar_data.get("mae", 0))
        all_labels_mae.append(1 if t["label"] == "strong_right" else (0 if t["label"] == "strong_wrong" else -1))

    for threshold in [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8]:
        close_if_below = threshold  # MAE is negative, so "below" means more negative
        kept = [(m, l) for m, l in zip(all_mae, all_labels_mae) if l >= 0 and m >= close_if_below]
        closed = [(m, l) for m, l in zip(all_mae, all_labels_mae) if l >= 0 and m < close_if_below]

        if len(kept) < 20 or len(closed) < 5: continue

        kept_right = sum(1 for _, l in kept if l == 1)
        kept_wrong = sum(1 for _, l in kept if l == 0)
        closed_right = sum(1 for _, l in closed if l == 1)
        closed_wrong = sum(1 for _, l in closed if l == 0)

        wrong_caught = closed_wrong / (closed_wrong + kept_wrong) * 100 if (closed_wrong + kept_wrong) > 0 else 0
        right_sacrificed = closed_right / (closed_right + kept_right) * 100 if (closed_right + kept_right) > 0 else 0

        if wrong_caught > 10:
            print(f"  Bar {detect_bar}: Close if MAE < {threshold:+.1f} → "
                  f"catch {wrong_caught:.0f}% wrong, sacrifice {right_sacrificed:.0f}% right  "
                  f"keep {len(kept)} trades ({kept_right/len(kept)*100:.0f}% right)")

# ===== Time-to-first-profit analysis =====
print("\n" + "="*95)
print("TIME TO FIRST PROFIT (MFE > 0)")
print("="*95)

for label, trade_list in [("strong_right", right), ("strong_wrong", wrong), ("both_ways", both)]:
    first_profit_bars = []
    for t in trade_list:
        pb = t["per_bar"]
        if not pb: continue
        for bar_data in pb:
            if bar_data.get("mfe", 0) > 0:
                first_profit_bars.append(bar_data.get("bar", 0))
                break
    if first_profit_bars:
        print(f"  {label:15s}: median={np.median(first_profit_bars):.0f} bars  "
              f"mean={np.mean(first_profit_bars):.1f} bars  "
              f"never_profitable={(1-len(first_profit_bars)/len(trade_list))*100:.0f}%")

# ===== Time-to-first-loss analysis =====
print("\nTIME TO FIRST LOSS (MAE < 0)")
for label, trade_list in [("strong_right", right), ("strong_wrong", wrong), ("both_ways", both)]:
    first_loss_bars = []
    for t in trade_list:
        pb = t["per_bar"]
        if not pb: continue
        for bar_data in pb:
            if bar_data.get("mae", 0) < 0:
                first_loss_bars.append(bar_data.get("bar", 0))
                break
    if first_loss_bars:
        print(f"  {label:15s}: median={np.median(first_loss_bars):.0f} bars  "
              f"mean={np.mean(first_loss_bars):.1f} bars  "
              f"never_losing={(1-len(first_loss_bars)/len(trade_list))*100:.0f}%")
