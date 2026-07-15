"""Simulate progressive trailing stop on today's actual trades."""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize():
    print("MT5 not available"); exit()

# Today's trades from logs
trades = [
    {"entry_ts": "2026-05-20 20:30", "entry": 77481.83, "fill": 77506.73,
     "sl": 77231.44, "tp": 78107.82, "dir": 1, "atr": 250.39,
     "exit_ts": "2026-05-20 21:15", "exit_price": None, "exit_reason": "sl_hit", "actual_r": 0.51},
    {"entry_ts": "2026-05-20 21:45", "entry": 77095.85, "fill": 77120.32,
     "sl": 76834.76, "tp": 77748.58, "dir": 1, "atr": 261.09,
     "exit_ts": "2026-05-21 00:30", "exit_price": None, "exit_reason": "sl_hit", "actual_r": 1.32},
    {"entry_ts": "2026-05-21 01:30", "entry": 77559.45, "fill": 77576.59,
     "sl": 77177.31, "tp": 78514.80, "dir": 1, "atr": 382.14,
     "exit_ts": "2026-05-21 06:00", "exit_price": 77431.67, "exit_reason": "time_stop", "actual_r": -0.33},
    {"entry_ts": "2026-05-21 06:30", "entry": 77441.72, "fill": 77466.63,
     "sl": 77106.71, "tp": 78279.25, "dir": 1, "atr": 335.01,
     "exit_ts": "2026-05-21 09:57", "exit_price": None, "exit_reason": "manual_close", "actual_r": None},
]

# Fetch M15 bars for today
symbol = "BTCUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 500)
df = pd.DataFrame(rates)
df["timestamp"] = pd.to_datetime(df["time"], unit="s")

def simulate_trail(trade, trigger1=1.0, dist1=0.3, trigger2=2.0, dist2=0.75, be_trigger=0.5):
    """Simulate progressive trail on a single trade."""
    atr = trade["atr"]; entry = trade["entry"]; direction = trade["dir"]
    sl = trade["sl"]; tp = trade["tp"]
    be_activated = False; trail1_active = False; trail2_active = False
    best_price = entry; current_sl = sl

    # Find M15 bars during this trade
    entry_t = pd.Timestamp(trade["entry_ts"])
    exit_t = pd.Timestamp(trade["exit_ts"]) if trade["exit_ts"] else df["timestamp"].max()
    bars = df[(df["timestamp"] >= entry_t) & (df["timestamp"] <= exit_t)].copy()
    if len(bars) == 0:
        return None

    trail_updates = []
    for _, bar in bars.iterrows():
        hi = bar["high"]; lo = bar["low"]; close = bar["close"]
        if direction == 1:
            best_price = max(best_price, hi)
            profit_r = (close - entry) / atr
            mfe_r = (best_price - entry) / atr
        else:
            best_price = min(best_price, lo)
            profit_r = (entry - close) / atr
            mfe_r = (entry - best_price) / atr

        # SL hit check
        if direction == 1 and lo <= current_sl:
            exit_r = (current_sl - entry) / atr
            return {"bars": len(bars), "exit_r": round(exit_r, 3), "exit_reason": "sl_hit",
                    "mfe_r": round(mfe_r, 3), "trail_updates": trail_updates,
                    "final_sl": current_sl}
        elif direction == -1 and hi >= current_sl:
            exit_r = (entry - current_sl) / atr
            return {"bars": len(bars), "exit_r": round(exit_r, 3), "exit_reason": "sl_hit",
                    "mfe_r": round(mfe_r, 3), "trail_updates": trail_updates,
                    "final_sl": current_sl}

        # TP check
        if (direction == 1 and hi >= tp) or (direction == -1 and lo <= tp):
            exit_r = 2.5
            return {"bars": len(bars), "exit_r": exit_r, "exit_reason": "tp_hit",
                    "mfe_r": round(mfe_r, 3), "trail_updates": trail_updates,
                    "final_sl": current_sl}

        # BE activation
        if not be_activated and profit_r >= be_trigger:
            be_activated = True
            current_sl = entry + direction * 0.05 * atr
            trail_updates.append(f"BE@{profit_r:.2f}R")

        # Progressive trail phase 1 (tight, at +1.0R)
        if profit_r >= trigger1 and not trail2_active:
            trail1_active = True
            new_sl = best_price - direction * dist1 * atr
            if direction == 1: new_sl = max(new_sl, current_sl)
            else: new_sl = min(new_sl, current_sl)
            if new_sl != current_sl:
                trail_updates.append(f"Trail1@{profit_r:.2f}R: {current_sl:.0f}->{new_sl:.0f}")
                current_sl = new_sl

        # Progressive trail phase 2 (wide, at +2.0R)
        if profit_r >= trigger2:
            trail2_active = True
            new_sl = best_price - direction * dist2 * atr
            if direction == 1: new_sl = max(new_sl, current_sl)
            else: new_sl = min(new_sl, current_sl)
            if new_sl != current_sl:
                trail_updates.append(f"Trail2@{profit_r:.2f}R: {current_sl:.0f}->{new_sl:.0f}")
                current_sl = new_sl

    # Reached end of bars without SL/TP — time stop or manual close
    exit_r = profit_r  # last bar close
    return {"bars": len(bars), "exit_r": round(exit_r, 3), "exit_reason": "time_stop",
            "mfe_r": round(mfe_r, 3), "trail_updates": trail_updates,
            "final_sl": current_sl}


print("PROGRESSIVE TRAIL SIMULATION — Today's Trades")
print("=" * 90)
print(f"  Proposed: BE@{0.5}R → Trail@{1.0}R (0.3R dist) → Trail@{2.0}R (0.75R dist)")
print(f"  Current:  BE@{0.5}R → Trail@{2.0}R (0.75R dist)")
print()

for i, t in enumerate(trades):
    if t["actual_r"] is None: continue

    # Current config simulation
    curr = simulate_trail(t, trigger1=2.0, dist1=0.75, trigger2=2.0, dist2=0.75)
    # Progressive config simulation
    prog = simulate_trail(t, trigger1=1.0, dist1=0.3, trigger2=2.0, dist2=0.75)

    if curr is None or prog is None:
        print(f"  Trade {i+1}: no M15 bars found")
        continue

    delta = prog["exit_r"] - curr["exit_r"]
    print(f"  Trade {i+1}: {t['entry_ts']} LONG @ {t['entry']:.0f}  ATR=${t['atr']:.0f}")
    print(f"    Actual exit:  {t['actual_r']:+.2f}R ({t['exit_reason']})")
    print(f"    Current conf: {curr['exit_r']:+.3f}R ({curr['exit_reason']}) MFE={curr['mfe_r']:+.2f}R  trail updates: {curr['trail_updates']}")
    print(f"    Progressive:  {prog['exit_r']:+.3f}R ({prog['exit_reason']}) MFE={prog['mfe_r']:+.2f}R  trail updates: {prog['trail_updates']}")
    print(f"    Delta: {delta:+.3f}R  {'*** BETTER ***' if delta > 0 else 'worse' if delta < 0 else 'same'}")
    print()

# Net effect
curr_total = sum(simulate_trail(t, trigger1=2.0, dist1=0.75, trigger2=2.0, dist2=0.75)["exit_r"]
                  for t in trades if t["actual_r"] is not None)
prog_total = sum(simulate_trail(t, trigger1=1.0, dist1=0.3, trigger2=2.0, dist2=0.75)["exit_r"]
                  for t in trades if t["actual_r"] is not None)
print(f"  NET: Current={curr_total:+.2f}R → Progressive={prog_total:+.2f}R (Δ={prog_total-curr_total:+.2f}R)")
