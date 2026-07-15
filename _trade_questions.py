"""Answer 3 specific questions about trade behavior."""
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
        ab.append({"bar":len(ab),"mfe":mfe_now,"mae":mae_now,"phase":s2.phase.name,"price":price})
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
            mae_trough=min(b["mae"] for b in ab) if ab else 0
            peak_bar=next((b["bar"] for b in ab if b["mfe"]>=mfe_peak*0.95),len(ab)-1) if mfe_peak>0.01 else 0
            bars_after_peak=len(ab)-peak_bar
            first_025=next((b["bar"] for b in ab if b["mfe"]>=0.25),None)
            linger_bars=len(ab)-first_025 if first_025 is not None else 0
            crossed_05r=any(b["mfe"]>=0.50 for b in ab)
            be_activated=any(b["phase"] in ("BREAKEVEN","TRAILING","TIGHTENED") for b in ab)
            trades.append({"pnl_r":pnl_r,"mfe_peak":mfe_peak,"mae_trough":mae_trough,
                "exit_reason":er,"bars_held":len(ab),
                "breakeven_activated":be_activated,"crossed_05r":crossed_05r,
                "bars_after_peak":bars_after_peak,"linger_bars":linger_bars})
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

# Q1
went_05r=[t for t in trades if t["mfe_peak"]>=0.50]
be_activated=[t for t in went_05r if t["breakeven_activated"]]
be_not=[t for t in went_05r if not t["breakeven_activated"]]
print("="*60)
print("Q1: What % of +0.5R trades have stop moved to breakeven?")
print("="*60)
print(f"  Trades that reached MFE >= 0.50R: {len(went_05r)}/{n} ({len(went_05r)/n*100:.1f}%)")
print(f"  BE actually activated: {len(be_activated)}/{len(went_05r)} ({len(be_activated)/max(len(went_05r),1)*100:.1f}%)")
print(f"  BE NOT activated: {len(be_not)}/{len(went_05r)}")
if be_not:
    reasons=Counter(t["exit_reason"] for t in be_not)
    print(f"  Why BE didn't activate: {dict(reasons)}")
    print(f"  (These hit the original SL before price reached +0.50R for breakeven)")

# Q2
buckets=[(0.25,0.50),(0.50,1.00),(1.00,2.00),(2.00,99)]
print(f"\n{'='*60}")
print("Q2: For 0.25-0.50R and >0.50R buckets — linger time after MFE")
print("="*60)
print(f"  {'MFE Bucket':>15s} {'Count':>6s} {'Wins':>6s} {'Losses':>6s} {'BarsFromPeak':>13s} {'Linger025toExit':>15s} {'AvgExitR':>10s}")
for lo,hi in buckets:
    bucket=[t for t in trades if lo<=t["mfe_peak"]<hi]
    if not bucket:continue
    bw=[t for t in bucket if t["pnl_r"]>0];bl=[t for t in bucket if t["pnl_r"]<=0]
    bap=np.mean([t["bars_after_peak"] for t in bucket])
    linger=np.mean([t["linger_bars"] for t in bucket if t["linger_bars"]>0])
    ex=np.mean([t["pnl_r"] for t in bucket])
    print(f"  {f'{lo:.2f}-{hi:.2f}R':>15s} {len(bucket):6d} {len(bw):6d} {len(bl):6d} {bap:12.1f} bars {linger:14.1f} bars {ex:+9.3f}R")

# Also losses only in these buckets
loss_025_050=[t for t in losses if 0.25<=t["mfe_peak"]<0.50]
loss_050=[t for t in losses if t["mfe_peak"]>=0.50]
print(f"\n  LOSSES in these buckets:")
print(f"    0.25-0.50R losses: {len(loss_025_050)}, avg bars_after_peak={np.mean([t['bars_after_peak'] for t in loss_025_050]):.1f}, avg linger={np.mean([t['linger_bars'] for t in loss_025_050 if t['linger_bars']>0]):.1f}" if loss_025_050 else "")
print(f"    >0.50R losses: {len(loss_050)}, avg bars_after_peak={np.mean([t['bars_after_peak'] for t in loss_050]):.1f}, avg linger={np.mean([t['linger_bars'] for t in loss_050 if t['linger_bars']>0]):.1f}" if loss_050 else "")

# Q3
print(f"\n{'='*60}")
print("Q3: Avg MFE — Wins vs Losses")
print("="*60)
print(f"  {'':>15s} {'Count':>6s} {'AvgMFE':>8s} {'AvgExitR':>9s} {'MFECapture':>10s} {'AvgMAE':>9s}")
for label,group in [("WINS",wins),("LOSSES",losses),("ALL",trades)]:
    avg_mfe=np.mean([t["mfe_peak"] for t in group])
    avg_pnl=np.mean([t["pnl_r"] for t in group])
    avg_mae=np.mean([abs(t["mae_trough"]) for t in group])
    capture=avg_pnl/max(avg_mfe,0.01)*100
    print(f"  {label:>15s} {len(group):6d} {avg_mfe:+8.3f}R {avg_pnl:+8.3f}R {capture:9.1f}% {avg_mae:+9.3f}R")

wmfe=np.mean([t["mfe_peak"] for t in wins]);lmfe=np.mean([t["mfe_peak"] for t in losses])
print(f"\n  MFE ratio (wins/losses): {wmfe:.3f}R / {lmfe:.3f}R = {wmfe/max(lmfe,0.01):.1f}x")

# Additional: how many losses HAD enough MFE to be wins?
had_enough_mfe=[t for t in losses if t["mfe_peak"]>=0.75]
print(f"\n  Losses that had MFE >= 0.75R (could have been decent wins): {len(had_enough_mfe)}/{len(losses)} ({len(had_enough_mfe)/max(len(losses),1)*100:.1f}%)")
if had_enough_mfe:
    print(f"    Avg MFE: {np.mean([t['mfe_peak'] for t in had_enough_mfe]):.2f}R, avg exit: {np.mean([t['pnl_r'] for t in had_enough_mfe]):.3f}R")
    print(f"    Exit reasons: {dict(Counter(t['exit_reason'] for t in had_enough_mfe))}")
