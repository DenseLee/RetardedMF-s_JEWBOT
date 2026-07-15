"""Test lower trail triggers that actually engage."""
import sys,os,time,itertools,numpy as np,pandas as pd,torch,multiprocessing as mp
from multiprocessing import cpu_count
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,".")
from config_btc import BTCConfig; cfg=BTCConfig()
h1f=pd.read_csv(cfg.data_dir+"/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv");h1f["timestamp"]=pd.to_datetime(h1f["timestamp"],utc=True)
m15f=pd.read_csv(cfg.data_dir+"/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv");m15f["timestamp"]=pd.to_datetime(m15f["timestamp"],utc=True)
ft=pd.Timestamp("2026-01-01",tz="UTC")
h1f=h1f[h1f["timestamp"]>=ft].reset_index(drop=True)
m15f=m15f[m15f["timestamp"]>=ft].reset_index(drop=True)

PARAMS=list(itertools.product([1.0,1.5],[0.75,1.0,1.25,1.5]))  # TT × TD = 8 combos

def run(pt):
    tt,td=pt
    from config_btc import BTCConfig
    from data.feature_engine_btc import BTCFeatureEngine
    from models.cnn_lstm_encoder import CNNLSTMEncoder
    from models.cnn_gru_m15 import CNNGRUM15
    from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
    from models.entry_gate import EntryGate
    from models.trade_manager_btc import TradeManager,TradeActionType
    from execution.mt5_executor_btc import DryRunExecutor
    config=BTCConfig();device=torch.device("cpu")
    encoder=CNNLSTMEncoder(n_features=17,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=4,bidirectional=True).to(device).eval()
    classifier=RegimeClassifier(embedding_dim=128,n_classes=4).to(device).eval()
    ckpt=torch.load(config.model_dir+"/btc_h1_encoder.pt",map_location=device,weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"]);classifier.load_state_dict(ckpt["classifier_state_dict"])
    m15_model=CNNGRUM15(n_features=17,seq_len=config.seq_len_m15,cnn_channels=config.gru_cnn_channels,gru_hidden=config.gru_hidden,gru_layers=config.gru_layers,dropout=config.gru_dropout).to(device).eval()
    mc=torch.load(config.model_dir+"/btc_m15_model.pt",map_location=device,weights_only=False)
    m15_model.load_state_dict(mc["model_state_dict"])
    engine=BTCFeatureEngine();gate=EntryGate()
    tm=TradeManager(initial_sl=1.0,hard_tp=config.hard_tp,breakeven_trigger=0.50,trail_trigger=tt,trail_dist=td,trail_dist_s=td*0.67,regime_tighten=config.regime_tighten,max_hold=18,mae_guard_retrace=2.5)
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

    if not trades:return{"tt":tt,"td":td,"trades":0,"pnl":0,"wr":0,"pf":0,"dd":0,"avg_win_r":0,"avg_loss_r":0,"tp_pct":0,"trail_pct":0,"score":-999}
    n=len(trades);wins=[t for t in trades if t["pnl_dollar"]>0];losses=[t for t in trades if t["pnl_dollar"]<=0]
    wr=len(wins)/n*100;tp_sum=sum(t["pnl_dollar"] for t in trades)
    awr=np.mean([t["pnl_r"] for t in wins]) if wins else 0;alr=np.mean([t["pnl_r"] for t in losses]) if losses else 0
    tg=sum(t["pnl_dollar"] for t in wins);tl=abs(sum(t["pnl_dollar"] for t in losses));pf=tg/tl if tl>0 else float("inf")
    tp_hits=sum(1 for t in trades if t["exit_reason"]=="tp_hit");tpp=tp_hits/n*100
    trail_pct=sum(1 for t in trades if t["trail_active"])/n*100
    cum=np.cumsum([0]+[t["pnl_dollar"] for t in trades]);eq=10000+cum;peak=np.maximum.accumulate(eq)
    dd=float(np.max(np.where(peak>0,(peak-eq)/peak*100,0)));ret=(10000+tp_sum)/10000-1;score=tp_sum/max(dd,0.5)
    return{"tt":tt,"td":td,"trades":n,"wr":wr,"pnl":tp_sum,"pf":pf,"dd":dd,"avg_win_r":awr,"avg_loss_r":alr,"tp_pct":tpp,"trail_pct":trail_pct,"score":score,"ret":ret}

if __name__=="__main__":
    nw=min(cpu_count(),4);mp.set_start_method("spawn",force=True)
    print(f"Combos: {len(PARAMS)}, Workers: {nw}");t0=time.time()
    with mp.Pool(nw) as pool:results=pool.map(run,PARAMS)
    print(f"Done: {(time.time()-t0)/60:.1f} min")
    df=pd.DataFrame(results).sort_values("score",ascending=False)
    df.to_csv(os.path.join(cfg.log_dir,"trail_low_search.csv"),index=False)

    print(f'\n{"TT":>5s} {"TD":>5s} {"Trd":>5s} {"WR":>6s} {"PnL":>8s} {"Ret%":>6s} {"WinR":>6s} {"LossR":>6s} {"TP%":>5s} {"Trail%":>7s} {"DD%":>6s} {"Score":>7s}')
    print("-"*85)
    for _,row in df.iterrows():
        print(f'{float(row["tt"]):5.1f} {float(row["td"]):5.2f} {int(row["trades"]):5d} {float(row["wr"]):5.1f}% ${float(row["pnl"]):7.0f} {float(row["ret"])*100:5.1f}% {float(row["avg_win_r"]):+6.3f} {float(row["avg_loss_r"]):+6.3f} {float(row["tp_pct"]):4.1f}% {float(row["trail_pct"]):6.1f}% {float(row["dd"]):5.1f}% {float(row["score"]):7.1f}')
    print(f'\nOLD (TT=2.5,TD=0.75): PnL=$10,084 WinR=+0.79 Trail%=~5%')
