"""
Fetch BTC/USDT from Binance via CCXT and compare with LSE data.
Compares close prices on matching timestamps from 2026-01-01 onward.
"""
import pandas as pd
import ccxt
from datetime import datetime, timezone
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/TrainingData"

TIMEFRAMES = {
    "4h": "4h",
    "1h": "1h",
    "15min": "15m",
}

LSE_FILES = {
    "4h": f"{DATA_DIR}/(4h)_btc_usd_dataset_London-Strategic-Edge.csv",
    "1h": f"{DATA_DIR}/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv",
    "15min": f"{DATA_DIR}/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv",
}

SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SINCE_MS = int(SINCE.timestamp() * 1000)

exchange = ccxt.binance({"enableRateLimit": True})

def fetch_ohlcv(tf_key, limit=1000):
    """Fetch OHLCV from Binance, paginating from SINCE to now."""
    symbol = "BTC/USDT"
    all_bars = []
    since = SINCE_MS
    while True:
        bars = exchange.fetch_ohlcv(symbol, tf_key, since=since, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= since or len(bars) < limit:
            break
        since = last_ts + 1  # next batch
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df

def load_lse(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def main():
    for label, tf_ccxt in TIMEFRAMES.items():
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {label}")
        print(f"{'='*60}")

        # Load LSE
        lse_path = LSE_FILES[label]
        lse_df = load_lse(lse_path)
        lse_df = lse_df[lse_df["timestamp"] >= SINCE]
        print(f"LSE bars from 2026-01-01: {len(lse_df)}")

        # Fetch Binance
        print(f"Fetching Binance {label} data...")
        binance_df = fetch_ohlcv(tf_ccxt)
        print(f"Binance bars from 2026-01-01: {len(binance_df)}")

        if len(binance_df) == 0:
            print("ERROR: No Binance data fetched!")
            continue

        # Show date ranges
        print(f"LSE range:      {lse_df['timestamp'].min()} -> {lse_df['timestamp'].max()}")
        print(f"Binance range:  {binance_df['timestamp'].min()} -> {binance_df['timestamp'].max()}")

        # Merge on timestamp
        merged = lse_df.merge(binance_df, on="timestamp", suffixes=("_lse", "_binance"))

        if len(merged) == 0:
            print("ERROR: No matching timestamps! Checking alignment...")
            print(f"LSE sample timestamps: {lse_df['timestamp'].head(3).tolist()}")
            print(f"Binance sample timestamps: {binance_df['timestamp'].head(3).tolist()}")
            continue

        print(f"Matching bars: {len(merged)} / {len(lse_df)} LSE, {len(binance_df)} Binance")

        # Compare close prices
        correlation = merged["close_lse"].corr(merged["close_binance"])
        max_diff_pct = (
            (merged["close_lse"] - merged["close_binance"]) / merged["close_binance"]
        ).abs().max() * 100

        # Also show O/H/L/C all-up comparison
        print(f"\n--- Close Price Comparison ---")
        print(f"Correlation:     {correlation:.6f}  (target: >0.9999)")
        print(f"Max difference:  {max_diff_pct:.4f}%  (target: <0.1%)")

        # Show stats for all OHLC columns
        for col in ["open", "high", "low", "close"]:
            col_lse = f"{col}_lse"
            col_bin = f"{col}_binance"
            corr = merged[col_lse].corr(merged[col_bin])
            max_diff = ((merged[col_lse] - merged[col_bin]) / merged[col_bin]).abs().max() * 100
            mean_abs_diff = ((merged[col_lse] - merged[col_bin]) / merged[col_bin]).abs().mean() * 100
            print(f"  {col:>5}: corr={corr:.6f}  max_diff={max_diff:.4f}%  mean_abs_diff={mean_abs_diff:.4f}%")

        # Show the 5 worst differences
        merged["close_diff_pct"] = (
            (merged["close_lse"] - merged["close_binance"]) / merged["close_binance"]
        ).abs() * 100
        worst = merged.nlargest(5, "close_diff_pct")[
            ["timestamp", "close_lse", "close_binance", "close_diff_pct"]
        ]
        print(f"\nWorst 5 close differences:")
        for _, row in worst.iterrows():
            print(f"  {row['timestamp']}  LSE={row['close_lse']:.2f}  Binance={row['close_binance']:.2f}  diff={row['close_diff_pct']:.4f}%")

        # PASS/FAIL
        if correlation > 0.9999 and max_diff_pct < 0.1:
            print(f"\n>>> PASS: {label} data matches Binance")
        else:
            print(f"\n>>> FAIL: {label} data has discrepancies")

if __name__ == "__main__":
    main()
