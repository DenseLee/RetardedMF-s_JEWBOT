"""Can we predict strong_right vs strong_wrong at entry time?"""
import sys, pickle, numpy as np, pandas as pd
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
rows = []
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

    # Classify
    if entry_r > enemy_r * 1.5 and entry_r > 1.0:
        oracle_label = "strong_right"
    elif enemy_r > entry_r * 1.5 and enemy_r > 1.0:
        oracle_label = "strong_wrong"
    elif entry_r > 1.0 and enemy_r > 1.0:
        oracle_label = "both_ways"
    elif entry_r < 0.5 and enemy_r < 0.5:
        oracle_label = "dead"
    else:
        oracle_label = "marginal"

    rows.append({
        "oracle_label": oracle_label,
        "entry_r": entry_r, "enemy_r": enemy_r,
        "entry_hour": t["entry_hour_utc"],
        "h1_regime": t["h1_regime"],
        "h1_regime_confidence": t["h1_regime_confidence"],
        "m15_ema_distance_r": t["m15_ema_distance_r"],
        "m15_3bar_momentum_r": t["m15_3bar_momentum_r"],
        "m15_atr_ratio": t["m15_atr_ratio"],
        "m15_5min_realized_vol": t["m15_5min_realized_vol"],
        "m15_entry_confidence": t["m15_entry_confidence"],
        "m15_direction_bias": t["m15_direction_bias"],
        "confirmation_method": t["confirmation_method"],
        "bars_listened": t["bars_listened"],
        "with_h1_trend": t["with_h1_trend"],
        "direction": t["direction"],
        "pnl_r": t["pnl_r"], "pnl_d": t["pnl_dollar"],
    })

dm = pd.DataFrame(rows)
print(f"Matched: {len(dm)} trades")
print(f"strong_right: {(dm.oracle_label=='strong_right').sum()}  strong_wrong: {(dm.oracle_label=='strong_wrong').sum()}  both_ways: {(dm.oracle_label=='both_ways').sum()}  dead: {(dm.oracle_label=='dead').sum()}  marginal: {(dm.oracle_label=='marginal').sum()}")

# ========= 1. FEATURE DISTRIBUTION =========
print("\n" + "="*90)
print("1. NUMERIC FEATURE DISTRIBUTION: strong_right vs strong_wrong")
print("="*90)

numeric_cols = ["h1_regime_confidence", "m15_ema_distance_r", "m15_3bar_momentum_r",
                "m15_atr_ratio", "m15_5min_realized_vol", "m15_entry_confidence",
                "m15_direction_bias", "bars_listened"]

right = dm[dm.oracle_label == "strong_right"]
wrong = dm[dm.oracle_label == "strong_wrong"]

print(f"{'Feature':30s} {'Right_mean':>10s} {'Wrong_mean':>10s} {'Diff':>10s} {'Right_med':>10s} {'Wrong_med':>10s} {'Separation':>10s}")
print("-" * 85)
for col in numeric_cols:
    rm = right[col].mean(); wm = wrong[col].mean()
    rmed = right[col].median(); wmed = wrong[col].median()
    diff = rm - wm
    # Simple separation metric: |diff| / pooled_std
    pooled_std = np.sqrt((right[col].std()**2 + wrong[col].std()**2) / 2)
    sep = abs(diff) / max(pooled_std, 0.001)
    print(f"{col:30s} {rm:>10.4f} {wm:>10.4f} {diff:>+10.4f} {rmed:>10.4f} {wmed:>10.4f} {sep:>10.3f}")

# ========= 2. CATEGORICAL BREAKDOWNS =========
print("\n" + "="*90)
print("2. CATEGORICAL FEATURE BREAKDOWN")
print("="*90)

for col in ["h1_regime", "confirmation_method", "direction", "with_h1_trend"]:
    print(f"\n--- {col} ---")
    ct = pd.crosstab(dm[col], dm["oracle_label"])
    # Add percentages
    pct = ct.div(ct.sum(axis=0), axis=1) * 100
    print(ct.to_string())
    print("\nRow % (within each oracle label):")
    print(pct.round(1).to_string())

# ========= 3. HOUR OF DAY =========
print("\n" + "="*90)
print("3. HOUR OF DAY ANALYSIS")
print("="*90)
dm["hour"] = dm["entry_hour"].astype(int)
hour_stats = dm.groupby("hour").agg(
    total=("oracle_label", "count"),
    right=("oracle_label", lambda x: (x == "strong_right").sum()),
    wrong=("oracle_label", lambda x: (x == "strong_wrong").sum()),
    pnl=("pnl_d", "sum"),
).assign(ratio=lambda x: (x.right / x.wrong.replace(0, 1)))
print(f"{'Hour':>5s} {'Trades':>7s} {'Right':>7s} {'Wrong':>7s} {'Ratio':>8s} {'PnL':>10s} {'Verdict':>10s}")
print("-" * 60)
for h, row in hour_stats.iterrows():
    ratio = row["right"]/max(row["wrong"],1)
    verdict = "GOOD" if ratio > 1.5 else ("BAD" if ratio < 0.67 else "NEUTRAL")
    print(f"{int(h):>5d} {int(row['total']):>7d} {int(row['right']):>7d} {int(row['wrong']):>7d} {ratio:>8.2f} ${row['pnl']:>+9.1f} {verdict:>10s}")

# ========= 4. RULE COMBINATIONS =========
print("\n" + "="*90)
print("4. COMBINED RULE MINING")
print("="*90)

# Try simple rule combos
rules = [
    ("h1_regime == 'TREND_DOWN'", dm["h1_regime"] == "TREND_DOWN"),
    ("h1_regime == 'TREND_UP'", dm["h1_regime"] == "TREND_UP"),
    ("confirmation == 'nn_model'", dm["confirmation_method"] == "nn_model"),
    ("confirmation == 'ema_rule'", dm["confirmation_method"] == "ema_rule"),
    ("with_h1_trend == True", dm["with_h1_trend"] == True),
    ("with_h1_trend == False", dm["with_h1_trend"] == False),
    ("direction == 'LONG'", dm["direction"] == "LONG"),
    ("direction == 'SHORT'", dm["direction"] == "SHORT"),
    ("m15_confidence >= 0.5", dm["m15_entry_confidence"] >= 0.5),
    ("m15_confidence < 0.5", dm["m15_entry_confidence"] < 0.5),
    ("m15_confidence >= 0.7", dm["m15_entry_confidence"] >= 0.7),
    ("hour in [0,1,10,14,15,16,20]", dm["hour"].isin([0,1,10,14,15,16,20])),
    ("hour in [2,4,11,18,19,21,22,23]", dm["hour"].isin([2,4,11,18,19,21,22,23])),
    ("bars_listened <= 2", dm["bars_listened"] <= 2),
    ("m15_ema_distance < -0.5", dm["m15_ema_distance_r"] < -0.5),
    ("m15_ema_distance > 0.5", dm["m15_ema_distance_r"] > 0.5),
]

for rule_name, mask in rules:
    subset = dm[mask]
    if len(subset) < 10: continue
    n = len(subset)
    n_right = (subset.oracle_label == "strong_right").sum()
    n_wrong = (subset.oracle_label == "strong_wrong").sum()
    right_pct = n_right / n * 100
    wrong_pct = n_wrong / n * 100
    ratio = n_right / max(n_wrong, 1)
    pnl = subset["pnl_d"].sum()
    print(f"  {rule_name:40s} n={n:>4d}  right={right_pct:.0f}%  wrong={wrong_pct:.0f}%  R/W={ratio:.1f}  PnL=${pnl:+.0f}")

# Combined rules: regime + confidence + hour
print("\n--- Combined Rules ---")
for regime in ["TREND_UP", "TREND_DOWN"]:
    for conf_level in [0.3, 0.5, 0.7]:
        for hour_set_name, hour_mask in [("good_hours", dm["hour"].isin([0,1,10,14,15,16,20])),
                                          ("bad_hours", dm["hour"].isin([2,4,11,18,19,21,22,23]))]:
            mask = (dm["h1_regime"] == regime) & (dm["m15_entry_confidence"] >= conf_level) & hour_mask
            subset = dm[mask]
            if len(subset) < 10: continue
            n = len(subset)
            n_right = (subset.oracle_label == "strong_right").sum()
            n_wrong = (subset.oracle_label == "strong_wrong").sum()
            ratio = n_right / max(n_wrong, 1)
            pnl = subset["pnl_d"].sum()
            label = f"{regime} + M15conf>={conf_level} + {hour_set_name}"
            print(f"  {label:50s} n={n:>4d}  right={n_right}  wrong={n_wrong}  R/W={ratio:.1f}  PnL=${pnl:+.0f}")

# ========= 5. FEATURE IMPORTANCE (Random Forest) =========
print("\n" + "="*90)
print("5. RANDOM FOREST FEATURE IMPORTANCE")
print("="*90)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare binary classification: strong_right (1) vs strong_wrong (0)
binary = dm[dm.oracle_label.isin(["strong_right", "strong_wrong"])].copy()
binary["target"] = (binary.oracle_label == "strong_right").astype(int)

feature_cols = ["entry_hour", "h1_regime_confidence", "m15_ema_distance_r",
                "m15_3bar_momentum_r", "m15_atr_ratio", "m15_5min_realized_vol",
                "m15_entry_confidence", "m15_direction_bias", "bars_listened",
                "with_h1_trend"]

X = binary[feature_cols].copy()
X["with_h1_trend"] = X["with_h1_trend"].astype(int)
# One-hot encode regime
X = pd.get_dummies(X, columns=[], drop_first=False)
# Add regime one-hot
regime_dummies = pd.get_dummies(binary["h1_regime"], prefix="regime")
X = pd.concat([X, regime_dummies], axis=1)
# Add direction
dir_dummies = pd.get_dummies(binary["direction"], prefix="dir")
X = pd.concat([X, dir_dummies], axis=1)
# Add confirmation method
conf_dummies = pd.get_dummies(binary["confirmation_method"], prefix="conf")
X = pd.concat([X, conf_dummies], axis=1)

X = X.fillna(0)
y = binary["target"].values

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
importances = importances.sort_values("importance", ascending=False)
print(importances.head(15).to_string(index=False))

# ========= 6. DECISION TREE (interpretable rules) =========
print("\n" + "="*90)
print("6. DECISION TREE — Interpretable Split Points")
print("="*90)

from sklearn.tree import DecisionTreeClassifier, export_text

# Use only numeric features for a clean tree
num_features = ["h1_regime_confidence", "m15_ema_distance_r", "m15_3bar_momentum_r",
                "m15_atr_ratio", "m15_5min_realized_vol", "m15_entry_confidence",
                "m15_direction_bias", "bars_listened", "entry_hour"]
X_num = binary[num_features].fillna(0)

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
dt.fit(X_num, y)

tree_rules = export_text(dt, feature_names=num_features)
print(tree_rules)
print(f"Train accuracy: {dt.score(X_num, y):.3f}")

# ========= 7. CORRELATION MATRIX =========
print("\n" + "="*90)
print("7. FEATURE CORRELATION WITH 'IS_RIGHT' (point-biserial)")
print("="*90)

for col in numeric_cols + ["entry_hour"]:
    corr = binary[col].corr(binary["target"])
    print(f"  {col:30s} r={corr:+.4f}")

# Categorical: right% per category
for col in ["h1_regime", "direction", "confirmation_method", "with_h1_trend"]:
    print(f"\n  {col}:")
    for val, grp in binary.groupby(col):
        right_pct = (grp["target"] == 1).mean() * 100
        print(f"    {val}: {right_pct:.1f}% right  (n={len(grp)})")

# ========= 8. SUMMARY =========
print("\n" + "="*90)
print("8. CAN WE PREDICT strong_right vs strong_wrong?")
print("="*90)

# Best single rule
best_rules = []
for col in numeric_cols:
    # Try thresholds
    for pct in [25, 50, 75]:
        thresh = binary[col].quantile(pct/100)
        low = binary[binary[col] <= thresh]
        high = binary[binary[col] > thresh]
        if len(low) < 20 or len(high) < 20: continue
        low_right = (low["target"] == 1).mean()
        high_right = (high["target"] == 1).mean()
        sep = abs(low_right - high_right)
        best_rules.append((sep, f"{col} split at {thresh:.3f}: low={low_right:.2f} high={high_right:.2f}"))

best_rules.sort(reverse=True)
print("\nBest single-feature splits:")
for sep, rule in best_rules[:10]:
    print(f"  sep={sep:.3f}  {rule}")

# Baseline
baseline = binary["target"].mean()
print(f"\nBaseline (always predict right): {baseline:.3f}")
print(f"Best model accuracy: {dt.score(X_num, y):.3f} (vs baseline {max(baseline, 1-baseline):.3f})")

improvement = dt.score(X_num, y) - max(baseline, 1-baseline)
if improvement < 0.05:
    print(f"\nCONCLUSION: Features provide {improvement*100:.1f}% improvement over baseline.")
    print("Direction cannot be reliably predicted from entry-time features alone.")
    print("The H1 regime + M15 confirmation system does NOT distinguish right from wrong.")
    print("Recommendation: a different approach is needed (e.g., wider SL + faster wrong-detection exit).")
else:
    print(f"\nCONCLUSION: Features provide {improvement*100:.1f}% improvement over baseline.")
    print("Direction CAN be partially predicted from entry-time features.")
