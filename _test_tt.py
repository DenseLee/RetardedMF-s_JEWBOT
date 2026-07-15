"""Compare TT=2.5 vs 3.0 vs OFF on YTD."""
import sys,os,time,numpy as np,pandas as pd,torch
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
ft=pd.Timestamp("2026-01-01",tz="UTC")
h1f=h1f[h1f["timestamp"]>=ft].reset_index(drop=True)
m15f=m15f[m15f["timestamp"]>=ft].reset_index(drop=True)

def run_tt(tt_val,label):
    td=0.75 if tt_val<10 else 99.0; tt=tt_val if tt_val<10 else 99.0
    tm=TradeManager(initial_sl=1.0,hard_tp=config.hard_tp,breakeven_trigger=0.50,trail_trigger=tt,trail_dist=td,trail_dist_s=td*0.67,regime_tighten=config.regime_tighten,max_hold=18,mae_guard_retrace=2.5)
    exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
    bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0;h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]
    t0=time.time()
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
            if g.entry_signal:h1_sig=g.direction;listen=True;bl=0;h1_atr=h1_feats[-1,6]*price
            else:h1_sig=None;listen=False
        if pos!=0 and tm.state is not None:
            hi=m15s["high"].iloc[-1];lo=m15s["low"].iloc[-1];epx=None;er=None
            s2=tm.state;sd2=1.0*s2.entry_atr
            mfe_now=(hi-s2.entry_price)/sd2 if pos==1 else (s2.entry_price-lo)/sd2
            ab.append({"mfe":mfe_now,"phase":s2.phase.name})
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
                trail_active=any(b["phase"]=="TRAILING" for b in ab)
                trades.append({"pnl_dollar":pnl,"pnl_r":pnl_r,"mfe_peak":mfe_peak,"exit_reason":er,"bars_held":len(ab),"trail_active":trail_active})
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
        pos=h1_sig

    n=len(trades);wins=[t for t in trades if t["pnl_dollar"]>0];losses=[t for t in trades if t["pnl_dollar"]<=0]
    wr=len(wins)/n*100;tp=sum(t["pnl_dollar"] for t in trades)
    awr=np.mean([t["pnl_r"] for t in wins]) if wins else 0
    alr=np.mean([t["pnl_r"] for t in losses]) if losses else 0
    tg=sum(t["pnl_dollar"] for t in wins);tl=abs(sum(t["pnl_dollar"] for t in losses))
    pf=tg/tl if tl>0 else float("inf")
    tp_hits=sum(1 for t in trades if t["exit_reason"]=="tp_hit")
    ts_wins=sum(1 for t in wins if t["exit_reason"]=="Time stop")
    sl_wins=sum(1 for t in wins if t["exit_reason"]=="sl_hit")
    trail_pct=sum(1 for t in trades if t["trail_active"])/n*100
    cum=np.cumsum([0]+[t["pnl_dollar"] for t in trades]);eq=10000+cum;peak=np.maximum.accumulate(eq)
    dd=float(np.max(np.where(peak>0,(peak-eq)/peak*100,0)));ret=(10000+tp)/10000-1
    elapsed=(time.time()-t0)/60
    sl_win_cap=np.mean([t["pnl_r"]/max(t["mfe_peak"],0.01) for t in wins if t["exit_reason"]=="sl_hit"])*100 if sl_wins else 0
    print(f"\n{'='*70}")
    print(f"  TT={label}  |  {n} trades  |  {elapsed:.1f} min")
    print(f"{'='*70}")
    print(f"  PnL: ${tp:+,.0f}  |  Ret: {ret*100:+.1f}%  |  WR: {wr:.1f}%  |  PF: {pf:.2f}  |  DD: {dd:.1f}%")
    print(f"  Avg Win: {awr:+.3f}R  |  Avg Loss: {alr:+.3f}R  |  TP: {tp_hits} ({tp_hits/n*100:.1f}%)")
    print(f"  Trail: {trail_pct:.1f}%  |  Exits: TP={tp_hits} TSwin={ts_wins} SLwin={sl_wins} Loss={len(losses)}")
    print(f"  SL win MFE cap: {sl_win_cap:.0f}%")
    return{"label":label,"n":n,"pnl":tp,"wr":wr,"pf":pf,"dd":dd,"awr":awr,"alr":alr,"tp_pct":tp_hits/n*100,"trail_pct":trail_pct,"ret":ret}

r25=run_tt(2.5,"2.5");r30=run_tt(3.0,"3.0");roff=run_tt(99,"OFF")
print(f"\n{'='*70}\n  COMPARISON\n{'='*70}")
print(f'  {"Config":>12s} {"PnL":>10s} {"WR":>7s} {"Ret":>8s} {"PF":>6s} {"DD":>7s} {"WinR":>7s} {"LossR":>7s} {"TP%":>6s} {"Trail%":>7s}')
for r in [r25,r30,roff]:
    print(f'  {r["label"]:>12s} ${r["pnl"]:>8,.0f} {r["wr"]:>6.1f}% {r["ret"]*100:>7.1f}% {r["pf"]:>5.2f} {r["dd"]:>6.1f}% {r["awr"]:>+6.3f} {r["alr"]:>+6.3f} {r["tp_pct"]:>5.1f}% {r["trail_pct"]:>6.1f}%')
