"""YTD performance report from live bot trade logs."""
import pandas as pd, numpy as np

df = pd.read_csv("D:/FiananceBot/BTC_BOT/logs/btc_all_trades.csv")
df["ts"] = pd.to_datetime(df["entry_ts"], utc=True)
df["month"] = df["ts"].dt.strftime("%Y-%m")

monthly = df.groupby("month").agg(
    trades=("pnl_dollar", "count"),
    pnl=("pnl_dollar", "sum"),
    avg_r=("pnl_r", "mean"),
    wr=("pnl_r", lambda x: (x > 0).mean() * 100),
    tp=("exit_reason", lambda x: (x == "tp_hit").sum()),
    sl=("exit_reason", lambda x: (x == "sl_hit").sum()),
    time_stop=("exit_reason", lambda x: (x == "Time stop").sum()),
).round(2)

# Add breakeven count
be_counts = {}
for month, grp in df.groupby("month"):
    be = ((grp["exit_reason"] == "sl_hit") & (grp["pnl_r"] > -0.5)).sum()
    be_counts[month] = int(be)
monthly["be"] = pd.Series(be_counts)

# Total
total_trades = len(df)
total_wr = (df["pnl_r"] > 0).mean() * 100
total_pnl = df["pnl_dollar"].sum()
total_avg_r = df["pnl_r"].mean()

rs = df["pnl_r"].values
tg = rs[rs > 0].sum()
tl = abs(rs[rs <= 0].sum())
pf = tg / max(tl, 0.001)
sharpe = rs.mean() / max(rs.std(), 0.001)
elo = 1500 + (total_wr - 50) * 10 + min((pf - 1) * 300, 500) + min(sharpe * 100, 300) + min(rs.sum() * 10, 500)

print("BTC BOT -- LIVE/DRY-RUN YTD PERFORMANCE")
print("=" * 80)
print(f"Period: {df['ts'].min().date()} to {df['ts'].max().date()}")
print()

header = f"{'Month':<10s} {'Trades':>7s} {'PnL':>10s} {'AvgR':>8s} {'WR':>7s} {'TP':>5s} {'SL':>5s} {'BE':>5s} {'Time':>5s}"
print(header)
print("-" * len(header))
for idx, row in monthly.iterrows():
    print(f"{idx:<10s} {int(row['trades']):>7d} ${row['pnl']:>+9.1f} {row['avg_r']:>+8.3f} {row['wr']:>6.1f}% {int(row['tp']):>5d} {int(row['sl']):>5d} {int(row['be']):>5d} {int(row['time_stop']):>5d}")

print("-" * len(header))
print(f"{'TOTAL':<10s} {total_trades:>7d} ${total_pnl:>+9.1f} {total_avg_r:>+8.3f} {total_wr:>6.1f}% {(df['exit_reason']=='tp_hit').sum():>5d} {(df['exit_reason']=='sl_hit').sum():>5d} {int(monthly['be'].sum()):>5d} {(df['exit_reason']=='Time stop').sum():>5d}")

print()
print(f"PF: {pf:.2f}  Sharpe: {sharpe:.2f}  Total R: {rs.sum():+.1f}  ELO: {elo:.0f}")
print(f"Best trade: ${df['pnl_dollar'].max():+,.1f} ({df['pnl_r'].max():+.2f}R)")
print(f"Worst trade: ${df['pnl_dollar'].min():+,.1f} ({df['pnl_r'].min():+.2f}R)")

print()
print("By Regime:")
regime = df.groupby("h1_regime").agg(
    trades=("pnl_dollar", "count"),
    pnl=("pnl_dollar", "sum"),
    avg_r=("pnl_r", "mean"),
    wr=("pnl_r", lambda x: (x > 0).mean() * 100),
).round(2)
print(regime.to_string())

print()
print("By Confirmation Method:")
conf = df.groupby("confirmation_method").agg(
    trades=("pnl_dollar", "count"),
    pnl=("pnl_dollar", "sum"),
    avg_r=("pnl_r", "mean"),
    wr=("pnl_r", lambda x: (x > 0).mean() * 100),
).round(2)
print(conf.to_string())

# Direction split
print()
print("By Direction:")
dir_split = df.groupby("direction").agg(
    trades=("pnl_dollar", "count"),
    pnl=("pnl_dollar", "sum"),
    avg_r=("pnl_r", "mean"),
    wr=("pnl_r", lambda x: (x > 0).mean() * 100),
).round(2)
print(dir_split.to_string())
