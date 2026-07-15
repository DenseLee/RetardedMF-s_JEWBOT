"""Analyze SL-hit losses: directionally wrong vs bad timing."""
import sys,os,numpy as np,pandas as pd,torch
from collections import defaultdict
sys.path.insert(0,".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config=BTCConfig();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder=CNNLSTMEncoder(n_features=17,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=4,bidirectional=True).to(device).eval()
classifier=RegimeClassifier(embedding_dim=128,n_classes=4).to(device).eval()
ckpt=torch.load(config.model_dir+"/btc_h1_encoder.pt",map_location=device,weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"]);classifier.load_state_dict(ckpt["classifier_state_dict"])
m15_model=CNNGRUM15(n_features=17,seq_len=config.seq_len_m15,cnn_channels=config.gru_cnn_channels,gru_hidden=config.gru_hidden,gru_layers=config.gru_layers,dropout=config.gru_dropout).to(device).eval()
mc=torch.load(config.model_dir+"/btc_m15_model.pt",map_location=device,weights_only=False)
m15_model.load_state_dict(mc["model_state_dict"])
engine=BTCFeatureEngine();gate=EntryGate()

h1f=pd.read_csv(config.data_dir+"/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv");h1f["timestamp"]=pd.to_datetime(h1f["timestamp"],utc=True)
m15f=pd.read_csv(config.data_dir+"/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv");m15f["timestamp"]=pd.to_datetime(m15f["timestamp"],utc=True)
ft=pd.Timestamp("2026-01-01",tz="UTC");et=pd.Timestamp("2026-05-06",tz="UTC")
h1f=h1f[(h1f["timestamp"]>=ft)&(h1f["timestamp"]<et)].reset_index(drop=True)
m15f=m15f[(m15f["timestamp"]>=ft)&(m15f["timestamp"]<et)].reset_index(drop=True)

tm=TradeManager(initial_sl=1.0,hard_tp=config.hard_tp,breakeven_trigger=0.50,trail_trigger=2.5,trail_dist=0.75,trail_dist_s=0.50,regime_tighten=config.regime_tighten,max_hold=18,mae_guard_retrace=2.5)
exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0;h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]

for i in range(max(config.seq_len_m15,20),len(m15f)):
    ts=m15f["timestamp"].iloc[i];price=m15f["close"].iloc[i];exec._current_price=price
    today=ts.date()
    if ld and today!=ld:pnl_d=0.0;sb=bal
    ld=today;h1s=h1f[h1f["timestamp"]<=ts];m15s=m15f.iloc[max(0,i-config.seq_len_m15*4):i+1]
    if len(h1s)<config.seq_len_h1:continue
    hl=h1s["timestamp"].max()
    if hl!=lh:
        lh=hl;h1_feats=engine.compute(h1s)
        seq=engine.compute_sequence(h1_feats,len(h1_feats)-1,config.seq_len_h1)
        t=torch.from_numpy(seq).unsqueeze(0).to(device)
        for _,row in h1s.iloc[-14:].iterrows():rd.update(row["high"],row["low"],row["close"])
        rr=classify_regime(encoder,classifier,t,rd,model_confidence_threshold=config.min_regime_confidence)
        g=gate.evaluate(rr["regime"],rr["confidence"],rr.get("atr_percentile",0.5),bb_position=h1_feats[-1,4])
        if g.entry_signal:
            h1_sig=g.direction;listen=True;bl=0;h1_atr=h1_feats[-1,6]*price
            entry_regime=rr["regime"];entry_conf=g.confidence
        else:h1_sig=None;listen=False
    if pos!=0 and tm.state is not None:
        hi=m15s["high"].iloc[-1];lo=m15s["low"].iloc[-1];epx=None;er=None
        s2=tm.state;sd2=1.0*s2.entry_atr
        mfe_now=(hi-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-lo)/sd2
        mae_now=(lo-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-hi)/sd2
        ab.append({"bar":len(ab),"mfe":mfe_now,"mae":mae_now})
        if tm.check_sl_hit(lo,hi):epx=tm.exit_price_at_sl();er="sl_hit"
        elif tm.check_tp_hit(lo,hi):epx=tm.exit_price_at_tp();er="tp_hit"
        else:
            a=tm.update(price,hi,lo,h1_atr)
            if a.action_type==TradeActionType.CLOSE:epx=price;er=a.reason
        if epx:
            pnl_r=(epx-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-epx)/sd2
            pnl=(epx-s2.entry_price)*lots if pos==1 else (s2.entry_price-epx)*lots
            bal+=pnl;pnl_d+=pnl
            mfe_peak=max(b["mfe"] for b in ab) if ab else 0;mae_trough=min(b["mae"] for b in ab) if ab else 0
            # When did MFE peak? first bar that hit peak
            peak_bar=next((b["bar"] for b in ab if b["mfe"]>=mfe_peak*0.95),0) if mfe_peak>0.01 else 0
            # Did it go against us immediately?
            first_bar_mfe=ab[0]["mfe"] if ab else 0;first_bar_mae=ab[0]["mae"] if ab else 0
            # How many bars had positive MFE?
            bars_positive=sum(1 for b in ab if b["mfe"]>0.01)
            bars_negative=sum(1 for b in ab if b["mfe"]<=0.01)
            # Max MFE as % of total bars (did it peak early?)
            peak_pct=peak_bar/max(len(ab),1)*100

            trades.append({"pnl_dollar":pnl,"pnl_r":pnl_r,"mfe_peak":mfe_peak,"mae_trough":mae_trough,
                "exit_reason":er,"bars_held":len(ab),"direction":"LONG" if pos==1 else "SHORT",
                "first_bar_mfe":first_bar_mfe,"first_bar_mae":first_bar_mae,
                "bars_positive":bars_positive,"bars_negative":bars_negative,
                "peak_bar":peak_bar,"peak_pct":peak_pct,
                "regime":entry_regime if pos==h1_sig else "unknown",
                "entry_conf":entry_conf if pos==h1_sig else 0})
            pos=0;tm.state=None;ab=[]
        continue
    if not listen:continue
    bl+=1
    if bl>config.max_listen_bars:listen=False;h1_sig=None;continue
    m15_feats=engine.compute(m15s);confirmed=False
    sm=engine.compute_sequence(m15_feats,len(m15_feats)-1,config.seq_len_m15)
    tt2=torch.from_numpy(sm).unsqueeze(0).to(device)
    with torch.no_grad():mo=m15_model(tt2)
    if mo["entry_confidence"].item()>=config.min_entry_confidence:
        bias=mo["direction_bias"].item()
        if (h1_sig==1 and bias>0) or (h1_sig==-1 and bias<0):confirmed=True
    if not confirmed:
        mc2=m15s["close"].values;ema21=pd.Series(mc2).ewm(span=21,adjust=False).mean().values
        if h1_sig==1 and mc2[-1]<=ema21[-1]*1.01 and mc2[-1]>mc2[-2]:confirmed=True
        elif h1_sig==-1 and mc2[-1]>=ema21[-1]*0.99 and mc2[-1]<mc2[-2]:confirmed=True
    if not confirmed:continue
    if abs(pnl_d)/max(sb,1)>=config.max_daily_loss:continue
    listen=False
    lots=tm.compute_position_size(bal,h1_atr,price,config.risk_pct,tm.initial_sl)
    tm.enter(h1_sig,price,h1_atr,lots)
    exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
    pos=h1_sig;entry_regime=rr["regime"];entry_conf=g.confidence

# ── Analyze SL-hit losses ──
sl_losses=[t for t in trades if t["exit_reason"]=="sl_hit" and t["pnl_r"]<0]
n=len(sl_losses)
print(f"\n{'='*70}")
print(f"  SL-HIT LOSS ANALYSIS — {n} SL losses, {len(trades)} total trades")
print(f"{'='*70}")

# Categorize by MFE
never_profitable=[t for t in sl_losses if t["mfe_peak"]<=0.01]  # never went green
barely_profitable=[t for t in sl_losses if 0.01<t["mfe_peak"]<=0.25]  # micro green
moderate_mfe=[t for t in sl_losses if 0.25<t["mfe_peak"]<=0.5]  # decent move
good_mfe=[t for t in sl_losses if t["mfe_peak"]>0.5]  # good move then reversed

print(f"\n  MFE CATEGORIES:")
print(f"    Never profitable (MFE<=0.01R):  {len(never_profitable):4d} ({len(never_profitable)/n*100:.1f}%) — WRONG DIRECTION")
print(f"    Barely green   (0.01-0.25R):    {len(barely_profitable):4d} ({len(barely_profitable)/n*100:.1f}%) — noise/wick")
print(f"    Moderate MFE   (0.25-0.50R):    {len(moderate_mfe):4d} ({len(moderate_mfe)/n*100:.1f}%) — small move then SL")
print(f"    Good MFE       (>0.50R):         {len(good_mfe):4d} ({len(good_mfe)/n*100:.1f}%) — HAD PROFIT, reversed")

# MFE timing analysis
print(f"\n  MFE PEAK TIMING (when did the best moment occur?):")
early_peak=[t for t in sl_losses if t["peak_pct"]<20]  # peak in first 20% of bars
mid_peak=[t for t in sl_losses if 20<=t["peak_pct"]<60]
late_peak=[t for t in sl_losses if t["peak_pct"]>=60]
print(f"    Early peak (<20% of bars):  {len(early_peak):4d} — died quickly")
print(f"    Mid peak   (20-60%):         {len(mid_peak):4d} — had some life")
print(f"    Late peak  (>60%):           {len(late_peak):4d} — held then reversed")

# First-bar behavior
print(f"\n  FIRST BAR BEHAVIOR:")
first_went_right=[t for t in sl_losses if t["first_bar_mfe"]>0]  # first bar in our favor
first_went_wrong=[t for t in sl_losses if t["first_bar_mfe"]<=0 and t["first_bar_mae"]<0]  # went against
first_flat=[t for t in sl_losses if abs(t["first_bar_mfe"])<=0.01 and abs(t["first_bar_mae"])<=0.01]
print(f"    First bar in our favor:  {len(first_went_right):4d} ({len(first_went_right)/n*100:.1f}%) — entry timing OK")
print(f"    First bar against us:    {len(first_went_wrong):4d} ({len(first_went_wrong)/n*100:.1f}%) — immediate reversal")
print(f"    First bar flat:           {len(first_flat):4d} ({len(first_flat)/n*100:.1f}%)")

if first_went_wrong:
    avg_first_mae=np.mean([t["first_bar_mae"] for t in first_went_wrong])
    avg_bars=np.mean([t["bars_held"] for t in first_went_wrong])
    print(f"      Avg first-bar MAE: {avg_first_mae:.3f}R, avg bars held: {avg_bars:.1f}")

# Percentage of bars with positive MFE
pct_positive=np.mean([t["bars_positive"]/max(t["bars_held"],1) for t in sl_losses])*100
print(f"\n  TIME SPENT PROFITABLE: {pct_positive:.0f}% of bars had MFE>0 (avg across {n} SL losses)")

# Regime at entry
by_regime=defaultdict(list)
for t in sl_losses:
    by_regime[t["regime"]].append(t["pnl_r"])
print(f"\n  REGIME AT ENTRY (for SL losses):")
for regime,rlist in sorted(by_regime.items(),key=lambda x:-len(x[1])):
    print(f"    {regime:>15s}: {len(rlist):4d} losses, avg {np.mean(rlist):+.2f}R, "
          f"avg MFE={np.mean([t['mfe_peak'] for t in sl_losses if t['regime']==regime]):.2f}R")

# Direction
long_losses=[t for t in sl_losses if t["direction"]=="LONG"]
short_losses=[t for t in sl_losses if t["direction"]=="SHORT"]
print(f"\n  DIRECTION:")
print(f"    Long losses:  {len(long_losses):4d} avg MFE={np.mean([t['mfe_peak'] for t in long_losses]):.3f}R")
print(f"    Short losses: {len(short_losses):4d} avg MFE={np.mean([t['mfe_peak'] for t in short_losses]):.3f}R")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
wrong_dir=len(never_profitable);bad_timing=n-wrong_dir
print(f"  Directionally wrong (MFE<=0.01R): {wrong_dir}/{n} ({wrong_dir/n*100:.1f}%)")
print(f"  Bad timing/entry (MFE>0.01R):     {bad_timing}/{n} ({bad_timing/n*100:.1f}%)")
print(f"  Of bad timing: had MFE>0.50R:     {len(good_mfe)}/{bad_timing} ({len(good_mfe)/max(bad_timing,1)*100:.1f}%)")
