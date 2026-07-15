"""Compare V1 (H1+M15) vs V2 (H1+H4+M15) on 2026 YTD."""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from config_btc import BTCConfig
from execution.trading_system_btc import BTCTradingSystem
import logging;logging.basicConfig(level=logging.WARNING)

config=BTCConfig()

print("Running V1 (no H4 gate)...")
s_v1=BTCTradingSystem(config=config,bot_id="test",dry_run=True,live=False,risk_pct=0.02,max_daily_loss=0.05)
s_v1.h4_encoder=None  # disable H4 even if model exists
s_v1.from_date="2026-01-01";s_v1.to_date="2026-05-20"
s_v1.run_loop()
v1_trades=s_v1.trade_history;v1_bal=s_v1.balance

print("\nRunning V2 (H4 gate)...")
s_v2=BTCTradingSystem(config=config,bot_id="test",dry_run=True,live=False,risk_pct=0.02,max_daily_loss=0.05)
s_v2.from_date="2026-01-01";s_v2.to_date="2026-05-20"
s_v2.run_loop()
v2_trades=s_v2.trade_history;v2_bal=s_v2.balance

def summarize(trades,bal,label):
    wins=[t for t in trades if t["pnl_dollar"]>0];losses=[t for t in trades if t["pnl_dollar"]<=0]
    n=len(trades);wr=len(wins)/n*100 if n else 0
    tg=sum(t["pnl_r"]for t in wins);tl=abs(sum(t["pnl_r"]for t in losses))
    pf=tg/max(tl,0.001);pnl=sum(t["pnl_dollar"]for t in trades)
    avg_win=np.mean([t["pnl_r"]for t in wins])if wins else 0
    avg_loss=np.mean([t["pnl_r"]for t in losses])if losses else 0
    tp=sum(1 for t in trades if t["exit_reason"]=="tp_hit")
    return{"label":label,"n":n,"wr":wr,"pf":pf,"pnl":pnl,"bal":bal,"aw":avg_win,"al":avg_loss,"tp":tp}

v1=summarize(v1_trades,v1_bal,"V1 (H1+M15)")
v2=summarize(v2_trades,v2_bal,"V2 (H1+H4+M15)")

print("\n"+"="*80)
print("V1 vs V2 — 2026 YTD (Jan 1 – May 20)")
print("="*80)
print(f"  {'':<20s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'AvgWin':>8s} {'AvgLoss':>8s} {'TP':>5s} {'Return':>8s}")
print("  "+"-"*80)
for r in [v1,v2]:
    ret=(r["bal"]/10000-1)*100
    print(f"  {r['label']:<20s} {r['n']:7d} {r['wr']:6.1f}% {r['pf']:6.2f} ${r['pnl']:>9,.0f} {r['aw']:+7.3f}R {r['al']:+7.3f}R {r['tp']:4d} {ret:+7.1f}%")

print(f"\n  Delta: ΔTrades={v2['n']-v1['n']:+d}  ΔWR={v2['wr']-v1['wr']:+.1f}pp  ΔPF={v2['pf']-v1['pf']:+.2f}  ΔPnL=${v2['pnl']-v1['pnl']:+,.0f}")
print(f"  V2 is {'*** BETTER ***' if v2['pf']>v1['pf'] else 'worse'}")
