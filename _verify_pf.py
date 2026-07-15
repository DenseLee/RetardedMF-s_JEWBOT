"""Verify PF calculation and re-test V1 vs V2 consistently."""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from config_btc import BTCConfig
from execution.trading_system_btc import BTCTradingSystem
import logging;logging.basicConfig(level=logging.WARNING)

config=BTCConfig()

def run_and_report(label, disable_h4=False):
    s=BTCTradingSystem(config=config,bot_id="test",dry_run=True,live=False,risk_pct=0.02,max_daily_loss=0.05)
    if disable_h4: s.h4_encoder=None
    s.from_date="2026-01-01";s.to_date="2026-05-20"
    s.run_loop()

    trades=s.trade_history
    # Compute PF using R-multiples (pnl_r), not dollar PnL
    wins=[t for t in trades if t["pnl_r"]>0]
    losses=[t for t in trades if t["pnl_r"]<=0]
    n=len(trades);wr=len(wins)/n*100 if n else 0
    sum_win_r=sum(t["pnl_r"]for t in wins)
    sum_loss_r=abs(sum(t["pnl_r"]for t in losses))
    pf_r=sum_win_r/max(sum_loss_r,0.001)

    # Also compute using dollar PnL (what _print_summary uses)
    dwins=[t for t in trades if t["pnl_dollar"]>0]
    dlosses=[t for t in trades if t["pnl_dollar"]<=0]
    sum_win_d=sum(t["pnl_dollar"]for t in dwins)
    sum_loss_d=abs(sum(t["pnl_dollar"]for t in dlosses))
    pf_d=sum_win_d/max(sum_loss_d,0.001)

    avg_win=np.mean([t["pnl_r"]for t in wins])if wins else 0
    avg_loss=np.mean([t["pnl_r"]for t in losses])if losses else 0
    pnl_d=sum(t["pnl_dollar"]for t in trades)
    tp=sum(1 for t in trades if t["exit_reason"]=="tp_hit")

    # Check for inconsistencies
    n_wins_d=len(dwins);n_wins_r=len(wins)
    mismatch=n_wins_d!=n_wins_r

    print(f"\n  {label}")
    print(f"  {'':>15s} {'N':>6s} {'WR':>6s} {'PF(R)':>7s} {'PF($)':>7s} {'AvgW':>7s} {'AvgL':>7s} {'TP':>5s} {'PnL':>9s} {'Bal':>9s}")
    print(f"  {'':>15s} {n:6d} {wr:5.1f}% {pf_r:6.2f} {pf_d:6.2f} {avg_win:+6.3f}R {avg_loss:+6.3f}R {tp:4d} ${pnl_d:>8,.0f} ${s.balance:>8,.0f}")
    if mismatch:
        print(f"  *** WARNING: win classification mismatch! R-wins={n_wins_r}, $-wins={n_wins_d}")

    # Loss distribution
    loss_dist={"full_sl":0,"partial":0,"be":0,"time":0}
    for t in losses:
        if t["pnl_r"]<-0.9:loss_dist["full_sl"]+=1
        elif t["pnl_r"]<-0.4:loss_dist["partial"]+=1
        elif t["pnl_r"]<0:loss_dist["be"]+=1
        else:loss_dist["time"]+=1
    print(f"  Loss breakdown: full_SL={loss_dist['full_sl']} partial={loss_dist['partial']} near_BE={loss_dist['be']} other={loss_dist['time']}")

    return {"n":n,"wr":wr,"pf_r":pf_r,"pf_d":pf_d,"pnl":pnl_d,"aw":avg_win,"al":avg_loss,"tp":tp,"bal":s.balance}

print("="*80)
print("V1 vs V2 — PF Verification | 2026-01-01 to 2026-05-20")
print("="*80)

v1=run_and_report("V1 (H1+M15, no H4)",disable_h4=True)
v2=run_and_report("V2 (H1+H4+M15)",disable_h4=False)

print(f"\n  Delta V2-V1: Trades={v2['n']-v1['n']:+d}  WR={v2['wr']-v1['wr']:+.1f}pp  PF(R)={v2['pf_r']-v1['pf_r']:+.2f}  PnL=${v2['pnl']-v1['pnl']:+,.0f}")
print(f"  V2 is {'*** BETTER ***' if v2['pf_r']>v1['pf_r'] else 'worse'}")
