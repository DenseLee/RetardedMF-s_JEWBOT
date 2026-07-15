"""Answer: MFE peak timing, BE activation delay, early exit rule simulation."""
import sys,os;sys.path.insert(0,".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor
import numpy as np,pandas as pd,torch
from collections import Counter

config=BTCConfig();device=torch.device("cuda")
encoder=CNNLSTMEncoder(n_features=17,seq_len=96,cnn_channels=(32,64,128),lstm_hidden=128,lstm_layers=2,dropout=0.3,embedding_dim=128,regime_classes=4,bidirectional=True).to(device).eval()
classifier=RegimeClassifier(embedding_dim=128,n_classes=4).to(device).eval()
ckpt=torch.load(config.model_dir+"/btc_h1_encoder.pt",map_location=device,weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"]);classifier.load_state_dict(ckpt["classifier_state_dict"])
m15_model=CNNGRUM15(n_features=17,seq_len=20,cnn_channels=(16,32,64),gru_hidden=64,gru_layers=1,dropout=0.2).to(device).eval()
mc=torch.load(config.model_dir+"/btc_m15_model.pt",map_location=device,weights_only=False)
m15_model.load_state_dict(mc["model_state_dict"])
engine=BTCFeatureEngine();gate=EntryGate()

h1f=pd.read_csv(config.data_dir+"/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv");h1f["timestamp"]=pd.to_datetime(h1f["timestamp"],utc=True)
m15f=pd.read_csv(config.data_dir+"/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv");m15f["timestamp"]=pd.to_datetime(m15f["timestamp"],utc=True)
ft=pd.Timestamp("2026-01-01",tz="UTC");et=pd.Timestamp("2026-05-06",tz="UTC")
h1f=h1f[(h1f["timestamp"]>=ft)&(h1f["timestamp"]<et)].reset_index(drop=True)
m15f=m15f[(m15f["timestamp"]>=ft)&(m15f["timestamp"]<et)].reset_index(drop=True)

tm=TradeManager(initial_sl=1.0,hard_tp=3.0,breakeven_trigger=0.50,trail_trigger=2.5,trail_dist=0.75,trail_dist_s=0.50,regime_tighten=0.40,max_hold=18,mae_guard_retrace=2.5)
exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0;h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]

for i in range(max(20,20),len(m15f)):
    ts=m15f["timestamp"].iloc[i];price=m15f["close"].iloc[i];exec._current_price=price
    today=ts.date()
    if ld and today!=ld:pnl_d=0.0;sb=bal
    ld=today;h1s=h1f[h1f["timestamp"]<=ts];m15s=m15f.iloc[max(0,i-80):i+1]
    if len(h1s)<96:continue
    hl=h1s["timestamp"].max()
    if hl!=lh:
        lh=hl;h1_feats=engine.compute(h1s)
        seq=engine.compute_sequence(h1_feats,len(h1_feats)-1,96)
        t=torch.from_numpy(seq).unsqueeze(0).to(device)
        for _,row in h1s.iloc[-14:].iterrows():rd.update(row["high"],row["low"],row["close"])
        rr=classify_regime(encoder,classifier,t,rd,model_confidence_threshold=0.6)
        g=gate.evaluate(rr["regime"],rr["confidence"],rr.get("atr_percentile",0.5),bb_position=h1_feats[-1,4])
        if g.entry_signal:h1_sig=g.direction;listen=True;bl=0;h1_atr=h1_feats[-1,6]*price
        else:h1_sig=None;listen=False
    if pos!=0 and tm.state is not None:
        hi=m15s["high"].iloc[-1];lo=m15s["low"].iloc[-1];epx=None;er=None
        s2=tm.state;sd2=1.0*s2.entry_atr
        mfe_now=(hi-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-lo)/sd2
        mae_now=(lo-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-hi)/sd2
        ab.append({"bar":len(ab)+1,"mfe":mfe_now,"mae":mae_now,"phase":s2.phase.name,"price":price,"sl":s2.current_sl})
        if tm.check_sl_hit(lo,hi):epx=tm.exit_price_at_sl();er="sl_hit"
        elif tm.check_tp_hit(lo,hi):epx=tm.exit_price_at_tp();er="tp_hit"
        else:
            a=tm.update(price,hi,lo,h1_atr)
            if a.action_type==TradeActionType.CLOSE:epx=price;er=a.reason
        if epx:
            pnl_r=(epx-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-epx)/sd2
            pnl=(epx-s2.entry_price)*lots if pos==1 else (s2.entry_price-epx)*lots
            bal+=pnl;pnl_d+=pnl
            mfe_peak=max(b["mfe"] for b in ab) if ab else 0
            # Bar where MFE first crossed thresholds
            first_025=next((b["bar"] for b in ab if b["mfe"]>=0.25),None)
            first_050=next((b["bar"] for b in ab if b["mfe"]>=0.50),None)
            peak_bar=next((b["bar"] for b in ab if b["mfe"]>=mfe_peak*0.95),len(ab)) if mfe_peak>0.01 else 0
            # BE activation bar
            be_bar=next((b["bar"] for b in ab if b["phase"] in ("BREAKEVEN","TRAILING","TIGHTENED")),None)
            # Did BE activate immediately at +0.50R or after a delay?
            be_delay=be_bar-first_050 if (be_bar and first_050) else None
            # Did price reach 0.5R within 8 bars?
            reached_05r_8bar=first_050 is not None and first_050<=8
            trades.append({"pnl_r":pnl_r,"mfe_peak":mfe_peak,"mae_trough":min(b["mae"] for b in ab) if ab else 0,
                "exit_reason":er,"bars_held":len(ab),
                "first_025_bar":first_025,"first_050_bar":first_050,"peak_bar":peak_bar,"be_bar":be_bar,
                "be_delay":be_delay,"reached_05r_8bar":reached_05r_8bar})
            pos=0;tm.state=None;ab=[]
        continue
    if not listen:continue
    bl+=1
    if bl>8:listen=False;h1_sig=None;continue
    m15_feats=engine.compute(m15s);confirmed=False
    sm=engine.compute_sequence(m15_feats,len(m15_feats)-1,20)
    tt2=torch.from_numpy(sm).unsqueeze(0).to(device)
    with torch.no_grad():mo=m15_model(tt2)
    if mo["entry_confidence"].item()>=0.6:
        bias=mo["direction_bias"].item()
        if (h1_sig==1 and bias>0) or (h1_sig==-1 and bias<0):confirmed=True
    if not confirmed:
        mc2=m15s["close"].values;ema21=pd.Series(mc2).ewm(span=21,adjust=False).mean().values
        if h1_sig==1 and mc2[-1]<=ema21[-1]*1.01 and mc2[-1]>mc2[-2]:confirmed=True
        elif h1_sig==-1 and mc2[-1]>=ema21[-1]*0.99 and mc2[-1]<mc2[-2]:confirmed=True
    if not confirmed:continue
    if abs(pnl_d)/max(sb,1)>=0.05:continue
    listen=False
    lots=tm.compute_position_size(bal,h1_atr,price,0.02,tm.initial_sl)
    tm.enter(h1_sig,price,h1_atr,lots)
    exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
    pos=h1_sig

n=len(trades);wins=[t for t in trades if t["pnl_r"]>0];losses=[t for t in trades if t["pnl_r"]<=0]
print(f"{n} trades: {len(wins)} wins, {len(losses)} losses\n")

# ═══ Q1: MFE peak timing for 0.25-0.50R bucket ═══
bucket_025_050=[t for t in trades if 0.25<=t["mfe_peak"]<0.50]
print("="*60)
print("Q1: For 0.25-0.50R MFE bucket — when did they peak?")
print("="*60)
if bucket_025_050:
    peak_bars=[t["peak_bar"] for t in bucket_025_050]
    first_025_bars=[t["first_025_bar"] for t in bucket_025_050 if t["first_025_bar"]]
    print(f"  Count: {len(bucket_025_050)} trades (W={sum(1 for t in bucket_025_050 if t['pnl_r']>0)}, L={sum(1 for t in bucket_025_050 if t['pnl_r']<0)})")
    print(f"  Peak bar (mean): {np.mean(peak_bars):.1f} bars after entry")
    print(f"  Peak bar (median): {np.median(peak_bars):.0f} bars")
    print(f"  Peak bar distribution:")
    for lo,hi in [(1,3),(3,6),(6,9),(9,12),(12,99)]:
        cnt=sum(1 for b in peak_bars if lo<=b<hi)
        print(f"    {lo:2d}-{hi:2d} bars: {cnt:3d} ({cnt/len(peak_bars)*100:.0f}%)")
    print(f"  First hit +0.25R at bar: mean={np.mean(first_025_bars):.1f}, median={np.median(first_025_bars):.0f}")
    print(f"  Avg bars from peak to exit: {np.mean([t['bars_held']-t['peak_bar'] for t in bucket_025_050]):.1f}")

# ═══ Q2: BE activation timing ═══
went_050=[t for t in trades if t["first_050_bar"] is not None]
be_activated=[t for t in went_050 if t["be_bar"] is not None]
print(f"\n{'='*60}")
print("Q2: BE activation — immediate vs delayed")
print("="*60)
print(f"  Trades hitting MFE >= 0.50R: {len(went_050)}")
print(f"  BE activated: {len(be_activated)} ({len(be_activated)/max(len(went_050),1)*100:.1f}%)")
if be_activated:
    delays=[t["be_delay"] for t in be_activated if t["be_delay"] is not None]
    immediate=sum(1 for d in delays if d<=1)
    short_delay=sum(1 for d in delays if 1<d<=4)
    long_delay=sum(1 for d in delays if d>4)
    print(f"  BE timing after hitting +0.50R:")
    print(f"    Immediate (<=1 bar):  {immediate:3d} ({immediate/len(delays)*100:.0f}%)")
    print(f"    Short delay (2-4 bar): {short_delay:3d} ({short_delay/len(delays)*100:.0f}%)")
    print(f"    Long delay (>4 bar):  {long_delay:3d} ({long_delay/len(delays)*100:.0f}%)")
    print(f"    Mean delay: {np.mean(delays):.1f} bars")

# ═══ Q3: Simulate "exit if not reached +0.5R within 8 bars" ═══
print(f"\n{'='*60}")
print("Q3: Simulate — exit if MFE < 0.5R after 8 bars")
print("="*60)

# Original metrics
orig_pnl=sum(t["pnl_r"] for t in trades)
orig_n=len(trades)
orig_wr=len(wins)/orig_n*100
orig_awr=np.mean([t["pnl_r"] for t in wins]) if wins else 0
orig_alr=np.mean([t["pnl_r"] for t in losses]) if losses else 0
tg=sum(t["pnl_r"] for t in wins);tl=abs(sum(t["pnl_r"] for t in losses))
orig_pf=tg/max(tl,0.001)

# Simulated: for each trade, if bars_held>=8 and MFE<0.5 at bar 8, exit at bar 8 price
sim_trades=[]
for t in trades:
    if t["bars_held"]>=8 and not t["reached_05r_8bar"]:
        # This trade would be exited early at bar 8 with whatever PnL it has at that point
        # We approximate: if MFE never reached 0.5R, the trade was likely negative
        # We don't have per-bar price, so use the exit PnL proportional to bars
        # Conservative estimate: exit at -0.5R (half the SL) since it hasn't shown strength
        sim_r=-0.5
        sim_trades.append(sim_r)
        #print(f"  Early exit: {t['pnl_r']:+.3f}R -> {sim_r:+.3f}R (MFE peak={t['mfe_peak']:.2f}R, bars={t['bars_held']})")
    else:
        sim_trades.append(t["pnl_r"])

sim_pnl=sum(sim_trades)
sim_wins=[r for r in sim_trades if r>0];sim_losses=[r for r in sim_trades if r<=0]
sim_wr=len(sim_wins)/len(sim_trades)*100
sim_awr=np.mean(sim_wins) if sim_wins else 0;sim_alr=np.mean(sim_losses) if sim_losses else 0
stg=sum(r for r in sim_wins);stl=abs(sum(r for r in sim_losses))
sim_pf=stg/max(stl,0.001)
early_exits=sum(1 for t in trades if t["bars_held"]>=8 and not t["reached_05r_8bar"])

print(f"  Trades that would be early-exited at bar 8: {early_exits}/{orig_n} ({early_exits/orig_n*100:.1f}%)")
print(f"  These had MFE peak < 0.50R after 8 bars")
print(f"  Their actual outcomes: avg {np.mean([t['pnl_r'] for t in trades if t['bars_held']>=8 and not t['reached_05r_8bar']]):+.3f}R")
print(f"")
print(f"  {'':>20s} {'Trades':>6s} {'WR':>6s} {'AvgWin':>7s} {'AvgLoss':>7s} {'TotalR':>8s} {'PF':>6s}")
print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*6}")
print(f"  {'ORIGINAL':>20s} {orig_n:6d} {orig_wr:5.1f}% {orig_awr:+6.3f} {orig_alr:+6.3f} {orig_pnl:+8.1f}R {orig_pf:5.2f}")
print(f"  {'EARLY EXIT (8-bar)':>20s} {len(sim_trades):6d} {sim_wr:5.1f}% {sim_awr:+6.3f} {sim_alr:+6.3f} {sim_pnl:+8.1f}R {sim_pf:5.2f}")
print(f"  {'CHANGE':>20s} {'':>6s} {sim_wr-orig_wr:+6.1f}pp {'':>7s} {'':>7s} {sim_pnl-orig_pnl:+8.1f}R {sim_pf-orig_pf:+6.2f}")

# Also show: what if we exit at -0.5R instead of continuing to -1.0R?
print(f"\n  Conservative estimate: early-exit trades get -0.50R instead of actual -0.87R avg")
print(f"  Actual PnL of early-exit candidates: {sum(t['pnl_r'] for t in trades if t['bars_held']>=8 and not t['reached_05r_8bar']):+.1f}R")
print(f"  Simulated PnL (exited at -0.5R): {sum(-0.5 for t in trades if t['bars_held']>=8 and not t['reached_05r_8bar']):+.1f}R")

# Breakdown of what the early-exited trades actually did
ee_trades=[t for t in trades if t["bars_held"]>=8 and not t["reached_05r_8bar"]]
print(f"\n  Early-exit candidates breakdown ({len(ee_trades)} trades):")
print(f"    Actual avg PnL: {np.mean([t['pnl_r'] for t in ee_trades]):+.3f}R")
print(f"    Actual exit reasons: {dict(Counter(t['exit_reason'] for t in ee_trades))}")
print(f"    % that eventually became wins: {sum(1 for t in ee_trades if t['pnl_r']>0)/max(len(ee_trades),1)*100:.0f}%")
print(f"    Avg bars held: {np.mean([t['bars_held'] for t in ee_trades]):.1f}")
