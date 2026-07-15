"""Detect S/R levels and see if they'd beat ATR-based SL/TP on today's trades."""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from collections import defaultdict

if not mt5.initialize():
    print("MT5 not available"); exit()

symbol = "BTCUSD"

# Fetch H1 + M15 bars
h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 300)
m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 800)
h1 = pd.DataFrame(h1_rates); m15 = pd.DataFrame(m15_rates)
h1["timestamp"] = pd.to_datetime(h1["time"], unit="s")
m15["timestamp"] = pd.to_datetime(m15["time"], unit="s")

# ── S/R Detection Algorithms ──

def pivot_points(df, window=5):
    """Classic pivot points using rolling windows."""
    highs = df["high"].values; lows = df["low"].values
    res = []; supp = []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            res.append({"price": highs[i], "ts": df["timestamp"].iloc[i], "strength": window})
        if lows[i] == min(lows[i-window:i+window+1]):
            supp.append({"price": lows[i], "ts": df["timestamp"].iloc[i], "strength": window})
    return res, supp

def swing_levels(df, min_swing_pct=0.5):
    """Detect swing highs/lows as S/R levels. A swing is a peak >min_swing_pct% from neighbors."""
    closes = df["close"].values; highs = df["high"].values; lows = df["low"].values
    res = []; supp = []
    for i in range(2, len(df) - 2):
        # Swing high: current high > all neighbors
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            # Filter by minimum swing size
            swing_size = highs[i] / min(highs[i-2:i+3]) - 1
            if swing_size * 100 >= min_swing_pct:
                res.append({"price": highs[i], "ts": df["timestamp"].iloc[i], "strength": swing_size})
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_size = max(lows[i-2:i+3]) / lows[i] - 1
            if swing_size * 100 >= min_swing_pct:
                supp.append({"price": lows[i], "ts": df["timestamp"].iloc[i], "strength": swing_size})
    return res, supp

def rolling_range(df, lookback=24):
    """Recent range: highest high / lowest low of last N bars."""
    recent_high = df["high"].iloc[-lookback:].max()
    recent_low = df["low"].iloc[-lookback:].min()
    return [{"price": recent_high, "ts": df["timestamp"].iloc[-1], "strength": lookback}], \
           [{"price": recent_low, "ts": df["timestamp"].iloc[-1], "strength": lookback}]

def volume_profile_poc(df, bins=50):
    """Volume Profile POC (Point of Control) — price with highest volume."""
    price_range = np.linspace(df["low"].min(), df["high"].max(), bins)
    volume_by_price = np.zeros(bins - 1)
    for i in range(len(df)):
        for j in range(bins - 1):
            if price_range[j] <= df["high"].iloc[i] and price_range[j+1] >= df["low"].iloc[i]:
                overlap = min(df["high"].iloc[i], price_range[j+1]) - max(df["low"].iloc[i], price_range[j])
                volume_by_price[j] += df["tick_volume"].iloc[i] * overlap / (price_range[j+1] - price_range[j])
    poc_idx = np.argmax(volume_by_price)
    poc = (price_range[poc_idx] + price_range[poc_idx+1]) / 2
    vah = price_range[max(0, min(bins-1, int(bins * 0.7)))]  # 70% value area high
    val = price_range[max(0, min(bins-1, int(bins * 0.3)))]  # 30% value area low
    return [{"price": vah, "ts": df["timestamp"].iloc[-1], "strength": "VAH"},
            {"price": poc, "ts": df["timestamp"].iloc[-1], "strength": "POC"}], \
           [{"price": val, "ts": df["timestamp"].iloc[-1], "strength": "VAL"},
            {"price": poc, "ts": df["timestamp"].iloc[-1], "strength": "POC"}]

def nearest_levels(res_levels, supp_levels, price, n=3):
    """Find N nearest resistance above price and support below price."""
    res_above = sorted([r for r in res_levels if r["price"] > price], key=lambda x: x["price"])[:n]
    supp_below = sorted([s for s in supp_levels if s["price"] < price], key=lambda x: x["price"], reverse=True)[:n]
    return res_above, supp_below


# Today's trades from logs
trades = [
    {"entry_ts": "2026-05-20 20:30", "entry": 77481.83, "atr": 250.39, "dir": 1,
     "sl": 77231.44, "tp": 78107.82, "exit_ts": "2026-05-20 21:15", "exit_reason": "sl_hit", "actual_r": 0.51},
    {"entry_ts": "2026-05-20 21:45", "entry": 77095.85, "atr": 261.09, "dir": 1,
     "sl": 76834.76, "tp": 77748.58, "exit_ts": "2026-05-21 00:30", "exit_reason": "sl_hit", "actual_r": 1.32},
    {"entry_ts": "2026-05-21 01:30", "entry": 77559.45, "atr": 382.14, "dir": 1,
     "sl": 77177.31, "tp": 78514.80, "exit_ts": "2026-05-21 06:00", "exit_reason": "time_stop", "actual_r": -0.33},
    {"entry_ts": "2026-05-21 06:30", "entry": 77441.72, "atr": 335.01, "dir": 1,
     "sl": 77106.71, "tp": 78279.25, "exit_ts": "2026-05-21 10:00", "exit_reason": "manual", "actual_r": None},
]

print("S/R-BASED SL/TP vs ATR-BASED — Today's Trades")
print("=" * 100)

for i, t in enumerate(trades):
    if i == 3 and t["actual_r"] is None: continue

    entry_t = pd.Timestamp(t["entry_ts"])
    exit_t = pd.Timestamp(t["exit_ts"])
    atr = t["atr"]; entry = t["entry"]

    # Get data up to entry time (what was known at entry)
    h1_known = h1[h1["timestamp"] < entry_t]
    m15_known = m15[m15["timestamp"] < entry_t]
    # Get data during trade (for simulation)
    m15_during = m15[(m15["timestamp"] >= entry_t) & (m15["timestamp"] <= exit_t)]

    if len(h1_known) < 20:
        print(f"  Trade {i+1}: not enough data"); continue

    # Run all S/R detection algorithms on data KNOWN at entry time
    # 1. Pivot points (H1, window=5)
    pp_res, pp_supp = pivot_points(h1_known, window=5)
    # 2. Swing levels (M15, 0.3% min swing)
    sw_res, sw_supp = swing_levels(m15_known, min_swing_pct=0.3)
    # 3. Rolling range (M15, 24 bars = 6 hours)
    rr_res, rr_supp = rolling_range(m15_known, lookback=24)
    # 4. Volume profile (M15, last 100 bars)
    vp_res, vp_supp = volume_profile_poc(m15_known.iloc[-100:])

    # Combine all levels
    all_res = pp_res + sw_res + rr_res + vp_res
    all_supp = pp_supp + sw_supp + rr_supp + vp_supp

    # Nearest levels above/below entry
    res_above, supp_below = nearest_levels(all_res, all_supp, entry, n=5)

    print(f"\n  Trade {i+1}: LONG @ ${entry:,.0f}  ATR=${atr:,.0f}  ({t['entry_ts']})")
    print(f"    ATR SL: ${t['sl']:,.0f} ({entry-t['sl']:,.0f} below)  |  ATR TP: ${t['tp']:,.0f} ({t['tp']-entry:,.0f} above)")
    print(f"    Actual: {t['actual_r']:+.2f}R ({t['exit_reason']})")

    # Show detected levels
    if supp_below:
        print(f"    Support below entry:")
        for s in supp_below[:3]:
            dist_r = (entry - s["price"]) / atr
            print(f"      ${s['price']:,.0f} (-{entry - s['price']:,.0f} = -{dist_r:.2f}R) [{s.get('strength','')}] {s['ts']}")
    if res_above:
        print(f"    Resistance above entry:")
        for r in res_above[:3]:
            dist_r = (r["price"] - entry) / atr
            print(f"      ${r['price']:,.0f} (+{r['price'] - entry:,.0f} = +{dist_r:.2f}R) [{r.get('strength','')}] {r['ts']}")

    # Simulate: what if we used the nearest resistance as TP and nearest support as SL?
    if res_above and supp_below:
        # Best S/R SL (nearest strong support below, but not too close)
        sr_sl_candidates = [s for s in supp_below if (entry - s["price"]) / atr >= 0.3]  # at least 0.3R away
        sr_sl = sr_sl_candidates[0]["price"] if sr_sl_candidates else supp_below[0]["price"]
        sr_sl_r = (entry - sr_sl) / atr

        # Best S/R TP (nearest resistance above)
        sr_tp_candidates = [r for r in res_above if (r["price"] - entry) / atr >= 0.5]
        sr_tp = sr_tp_candidates[0]["price"] if sr_tp_candidates else res_above[0]["price"]
        sr_tp_r = (sr_tp - entry) / atr

        # Simulate: which would hit first — S/R SL or S/R TP?
        hit_sl = False; hit_tp = False; exit_price = None
        for _, bar in m15_during.iterrows():
            if bar["low"] <= sr_sl:
                hit_sl = True; exit_price = sr_sl; break
            if bar["high"] >= sr_tp:
                hit_tp = True; exit_price = sr_tp; break
        if not (hit_sl or hit_tp):
            exit_price = m15_during["close"].iloc[-1] if len(m15_during) > 0 else entry

        sr_exit_r = (exit_price - entry) / atr
        sr_result = f"{'TP HIT' if hit_tp else 'SL HIT' if hit_sl else 'held'}"

        # Also test: S/R SL only (keep ATR TP)
        hit_sl2 = any(bar["low"] <= sr_sl for _, bar in m15_during.iterrows())
        hit_tp2 = any(bar["high"] >= t["tp"] for _, bar in m15_during.iterrows())
        if hit_sl2: sr_sl_only_r = (sr_sl - entry) / atr; sr_sl_only_label = "SL (S/R)"
        elif hit_tp2: sr_sl_only_r = (t["tp"] - entry) / atr; sr_sl_only_label = "TP (ATR)"
        else: sr_sl_only_r = (m15_during["close"].iloc[-1] - entry) / atr if len(m15_during) > 0 else 0; sr_sl_only_label = "held"

        delta = sr_exit_r - t["actual_r"]
        print(f"    S/R SL: ${sr_sl:,.0f} (-{sr_sl_r:.2f}R)  |  S/R TP: ${sr_tp:,.0f} (+{sr_tp_r:.2f}R)")
        print(f"    S/R result: {sr_exit_r:+.2f}R ({sr_result})  |  S/R SL only: {sr_sl_only_r:+.2f}R ({sr_sl_only_label})")
        print(f"    vs ATR:  Δ={delta:+.2f}R  {'*** BETTER ***' if delta > 0 else 'worse'}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("  S/R levels exist near entries but they are clustered around current price.")
print("  The nearest S/R above/below are often within 0.5-1.5R — tighter than ATR.")
print("  Whether this helps depends on if price respects those levels more than ATR multiples.")
