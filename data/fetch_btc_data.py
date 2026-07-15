"""
Fetch BTC/USDT OHLCV from Binance via CCXT.
Saves in LSE-compatible format: timestamp,open,high,low,close,volume
Supports initial download and incremental updates.
"""
import pandas as pd
import ccxt
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig

TIMEFRAMES = {
    "1h": "1h",
    "15m": "15m",
    "4h": "4h",
}

SINCE = "2020-01-01"
SYMBOL = "BTC/USDT"


def fetch_binance_ohlcv(symbol=SYMBOL, timeframe="1h",
                        since=SINCE, limit=1000):
    """Fetch OHLCV from Binance with pagination. Returns DataFrame."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since_dt = pd.Timestamp(since, tz="UTC")
    since_ms = int(since_dt.timestamp() * 1000)

    all_bars = []
    fetch_since = since_ms
    n_requests = 0

    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
        n_requests += 1
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if last_ts <= fetch_since or len(bars) < limit:
            break
        fetch_since = last_ts + 1
        if n_requests % 10 == 0:
            dt = pd.to_datetime(last_ts, unit="ms", utc=True)
            print(f"  ... {len(all_bars)} bars fetched, latest: {dt}")

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def save_csv(df, data_dir, timeframe):
    """Save DataFrame to CSV in LSE-compatible format."""
    filename = f"BTCUSD_{timeframe}.csv"
    path = os.path.join(data_dir, filename)
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} bars to {path}")
    return path


def load_csv(data_dir, timeframe):
    """Load existing CSV, return DataFrame with UTC timestamps."""
    filename = f"BTCUSD_{timeframe}.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def update_incremental(data_dir, symbol=SYMBOL, timeframe="1h"):
    """Fetch only bars newer than the last bar in the existing CSV."""
    existing = load_csv(data_dir, timeframe)
    if existing is not None and len(existing) > 0:
        last_ts = existing["timestamp"].max()
        since = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Incremental update for {timeframe} from {since}")
    else:
        since = SINCE
        print(f"Full download for {timeframe} from {since}")

    new_df = fetch_binance_ohlcv(symbol=symbol, timeframe=timeframe, since=since)

    if existing is not None and len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        combined = combined.reset_index(drop=True)
    else:
        combined = new_df

    return combined


def fetch_all_timeframes(data_dir, symbol=SYMBOL, incremental=True):
    """Fetch all timeframes, either full or incremental."""
    results = {}
    for label, tf in TIMEFRAMES.items():
        print(f"\n{'='*50}")
        print(f"Fetching {label} ({tf})...")
        if incremental:
            df = update_incremental(data_dir, symbol=symbol, timeframe=tf)
        else:
            df = fetch_binance_ohlcv(symbol=symbol, timeframe=tf, since=SINCE)
        save_csv(df, data_dir, tf)
        results[tf] = df
        print(f"  Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return results


if __name__ == "__main__":
    config = BTCConfig()
    fetch_all_timeframes(config.data_dir, incremental=True)
