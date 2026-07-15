"""Check what the encoder saw before today's rally."""
import sys, os, numpy as np, pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine

cfg = BTCConfig()
mt5.initialize()

# Fetch 10 days of H1 data
ed = pd.Timestamp("2026-05-26 18:00", tz="UTC")
sd = ed - pd.Timedelta(days=10)
h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, sd.to_pydatetime(), ed.to_pydatetime())
h1 = pd.DataFrame(h1r).rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
h1['timestamp'] = pd.to_datetime(h1['timestamp'], unit='s', utc=True)
h1 = h1.sort_values('timestamp').reset_index(drop=True)

# Show price action
print("Last 5 days of H1 price action (what the encoder sees at each bar):")
print(f"{'Time':22s} {'Open':>8s} {'High':>8s} {'Low':>8s} {'Close':>8s} {'Range':>8s} {'Dir':>5s}")
print("-" * 80)

prev_close = None
for i in range(max(0, len(h1) - 120), len(h1)):
    row = h1.iloc[i]
    rng = row['high'] - row['low']
    if prev_close:
        direction = "▲ UP" if row['close'] > prev_close else "▼ DN" if row['close'] < prev_close else "─"
    else:
        direction = ""
    marker = " ◄── RALLY?" if row['close'] - h1.iloc[max(0,i-4)]['close'] > 500 else ""

    print(f"{str(row['timestamp'])[:19]:22s} {row['open']:>8.1f} {row['high']:>8.1f} {row['low']:>8.1f} {row['close']:>8.1f} {rng:>8.1f} {direction:>5s}{marker}")
    prev_close = row['close']

# Show the context window per H1 bar evaluation from today
print(f"\n{'='*80}")
print("What the 96-bar (4-day) context looked like at each evaluation:")
print(f"{'='*80}")

for target_hour in [5, 6, 10, 14, 17]:
    target_ts = pd.Timestamp(f"2026-05-26 {target_hour:02d}:00:00", tz="UTC")
    idx = int((h1['timestamp'] <= target_ts).sum() - 1)
    if idx < 96: continue
    window = h1.iloc[idx - 95:idx + 1]
    close_start = window['close'].iloc[0]
    close_end = window['close'].iloc[-1]
    high_max = window['high'].max()
    low_min = window['low'].min()
    pct_change = (close_end / close_start - 1) * 100

    # Count direction of bars in window
    up_bars = (window['close'].values[1:] > window['close'].values[:-1]).sum()
    dn_bars = 96 - up_bars

    # Simple trend: linear regression slope
    closes = window['close'].values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    slope_pct = slope / closes.mean() * 100 * 96  # % change over the window

    print(f"\n  At {target_hour:02d}:00 UTC (past 96 bars):")
    print(f"    Price range: {low_min:.1f} → {high_max:.1f} (${high_max - low_min:.0f} range)")
    print(f"    Close change: {close_start:.1f} → {close_end:.1f} ({pct_change:+.2f}%)")
    print(f"    Bar direction: {up_bars} up / {dn_bars} down")
    print(f"    Trend slope: {slope:+.2f} per bar ({slope_pct:+.1f}% over window)")
    if slope > 0.5:
        print(f"    Human read: CLEAR UPTREND — model should see TREND_UP")
    elif slope < -0.5:
        print(f"    Human read: CLEAR DOWNTREND — model should see TREND_DOWN")
    else:
        print(f"    Human read: CHOPPY/SIDEWAYS — unclear direction")

mt5.shutdown()
