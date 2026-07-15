"""Test SL/TP ratios + best grid config with v2 model."""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from collections import Counter
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config=BTCConfig();device=torch.device('cuda')
encoder=CNNLSTMEncoder(n_features=config.n_features,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=config.regime_classes,bidirectional=True).to(device).eval()
classifier=RegimeClassifier(embedding_dim=config.embedding_dim,n_classes=config.regime_classes).to(device).eval()
ckpt=torch.load(config.model_dir+'/btc_h1_encoder.pt',map_location=device,weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict']);classifier.load_state_dict(ckpt['classifier_state_dict'])
m15_v2=CNNGRUM15(n_features=config.n_features,seq_len=config.seq_len_m15,cnn_channels=config.gru_cnn_channels,gru_hidden=config.gru_hidden,gru_layers=config.gru_layers,dropout=config.gru_dropout).to(device).eval()
mc2=torch.load(config.model_dir+'/btc_m15_v2.pt',map_location=device,weights_only=False)
m15_v2.load_state_dict(mc2['model_state_dict'],strict=False)
engine=BTCFeatureEngine();gate=EntryGate()
h1f=pd.read_csv(config.data_dir+'/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv');h1f['timestamp']=pd.to_datetime(h1f['timestamp'],utc=True)
m15f=pd.read_csv(config.data_dir+'/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv');m15f['timestamp']=pd.to_datetime(m15f['timestamp'],utc=True)
ft=pd.Timestamp('2026-01-01',tz='UTC');et=pd.Timestamp('2026-05-06',tz='UTC')
h1f=h1f[(h1f['timestamp']>=ft)&(h1f['timestamp']<et)].reset_index(drop=True)
m15f=m15f[(m15f['timestamp']>=ft)&(m15f['timestamp']<et)].reset_index(drop=True)
BLOCKED={2,11,18,19,21,22,23}

def run(sl,tp,be,tt,td,mh):
    tm=TradeManager(initial_sl=sl,hard_tp=tp,breakeven_trigger=be,trail_trigger=tt,trail_dist=td,trail_dist_s=td*0.67,regime_tighten=0.40,max_hold=mh,mae_guard_retrace=2.5)
    exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
    bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0
    h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]
    entry_regime='';entry_conf=0.0
    for i in range(max(config.seq_len_m15,20),len(m15f)):
        ts=m15f['timestamp'].iloc[i];price=m15f['close'].iloc[i];exec._current_price=price
        today=ts.date()
        if ld and today!=ld:pnl_d=0.0;sb=bal
        ld=today;h1s=h1f[h1f['timestamp']<=ts];m15s=m15f.iloc[max(0,i-config.seq_len_m15*4):i+1]
        if len(h1s)<config.seq_len_h1:continue
        hl=h1s['timestamp'].max()
        if hl!=lh:
            lh=hl;h1_feats=engine.compute(h1s)
            seq=engine.compute_sequence(h1_feats,len(h1_feats)-1,config.seq_len_h1)
            t=torch.from_numpy(seq).unsqueeze(0).to(device)
            for _,row in h1s.iloc[-14:].iterrows():rd.update(row['high'],row['low'],row['close'])
            rr=classify_regime(encoder,classifier,t,rd,model_confidence_threshold=config.min_regime_confidence)
            g=gate.evaluate(rr['regime'],rr['confidence'],rr.get('atr_percentile',0.5),bb_position=h1_feats[-1,4])
            if g.entry_signal:
                h1_closes=h1s['close'].values
                if len(h1_closes)>=23:
                    h1_ema22=pd.Series(h1_closes).ewm(span=22,adjust=False).mean().values
                    h1_slope=(h1_ema22[-1]-h1_ema22[-2])/max(abs(float(h1_ema22[-2])),1e-12)
                    with_trend=((g.direction==1 and h1_slope>0)or(g.direction==-1 and h1_slope<0))
                    if not with_trend:h1_sig=None;listen=False;continue
                h1_sig=g.direction;listen=True;bl=0;h1_atr=h1_feats[-1,6]*price;entry_regime=rr['regime'];entry_conf=g.confidence
            else:h1_sig=None;listen=False
        if pos!=0 and tm.state is not None:
            hi=m15s['high'].iloc[-1];lo=m15s['low'].iloc[-1];epx=None;er=None;s2=tm.state;sd2=1.0*s2.entry_atr
            ab.append({'bar':len(ab),'mfe':float((hi-s2.entry_price)/sd2 if pos==1 else(s2.entry_price-lo)/sd2)})
            if tm.check_sl_hit(lo,hi):epx=tm.exit_price_at_sl();er='sl_hit'
            elif tm.check_tp_hit(lo,hi):epx=tm.exit_price_at_tp();er='tp_hit'
            else:
                a=tm.update(price,hi,lo,h1_atr)
                if a.action_type==TradeActionType.CLOSE:epx=price;er=a.reason
            if epx:
                pnl_r=(epx-s2.entry_price)/sd2 if pos==1 else(s2.entry_price-epx)/sd2
                pnl_dollar=(epx-s2.entry_price)*lots if pos==1 else(s2.entry_price-epx)*lots
                bal+=pnl_dollar;pnl_d+=pnl_dollar
                trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'mfe_peak':round(max(b['mfe']for b in ab)if ab else 0,4),'exit_reason':er})
                pos=0;tm.state=None;ab=[]
            continue
        if not listen:continue
        bl+=1
        if bl>config.max_listen_bars:listen=False;h1_sig=None;continue
        if ts.hour in BLOCKED:continue
        m15_feats=engine.compute(m15s);confirmed=False
        sm=engine.compute_sequence(m15_feats,len(m15_feats)-1,config.seq_len_m15)
        tt2=torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad():mo=m15_v2(tt2)
        conf=mo['entry_confidence'].item()if hasattr(mo['entry_confidence'],'item')else float(mo['entry_confidence'])
        if conf>=0.5:confirmed=True
        if not confirmed:
            mc2=m15s['close'].values;ema21=pd.Series(mc2).ewm(span=21,adjust=False).mean().values
            if h1_sig==1 and mc2[-1]<=ema21[-1]*1.01 and mc2[-1]>mc2[-2]:confirmed=True
            elif h1_sig==-1 and mc2[-1]>=ema21[-1]*0.99 and mc2[-1]<mc2[-2]:confirmed=True
        if not confirmed:continue
        if abs(pnl_d)/max(sb,1)>=config.max_daily_loss:continue
        listen=False
        lots=tm.compute_position_size(bal,h1_atr,price,config.risk_pct,tm.initial_sl)
        tm.enter(h1_sig,price,h1_atr,lots)
        exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
        pos=h1_sig
    wins=[t for t in trades if t['pnl_r']>0];losses=[t for t in trades if t['pnl_r']<=0]
    n=len(trades);wr=len(wins)/n*100 if n else 0
    tg=sum(t['pnl_r']for t in wins);tl=abs(sum(t['pnl_r']for t in losses))
    pf=tg/max(tl,0.001);pnl=sum(t['pnl_dollar']for t in trades)
    tp_pct=sum(1 for t in trades if t['exit_reason']=='tp_hit')/n*100 if n else 0
    return{'n':n,'wr':wr,'pf':pf,'pnl':pnl,'aw':np.mean([t['pnl_r']for t in wins])if wins else 0,'al':np.mean([t['pnl_r']for t in losses])if losses else 0,'tp':tp_pct,'exit':dict(Counter(t['exit_reason']for t in trades))}

configs=[
    (1.0,3.0,0.50,2.5,0.75,18,'CURRENT'),
    (1.0,3.0,0.50,2.0,0.75,18,'BEST_GRID'),
    (1.0,2.5,0.50,2.0,0.75,18,'TP=2.5'),
    (1.0,2.0,0.50,2.0,0.75,18,'TP=2.0'),
    (1.25,3.0,0.50,2.0,0.75,18,'SL=1.25'),
    (1.0,3.0,0.75,2.0,0.75,18,'BE=0.75'),
    (1.0,3.0,0.50,2.0,1.00,21,'TT=2.0/TD=1.0/MH=21'),
]

print('SL/TP RATIO + CONFIG TEST')
print('='*95)
print(f"{'Config':<28s} {'N':>5s} {'WR':>6s} {'PF':>6s} {'PnL':>10s} {'AW':>7s} {'AL':>7s} {'TP%':>6s} {'Exits'}")
for sl,tp,be,tt,td,mh,label in configs:
    print(f'{label}...',end=' ',flush=True)
    r=run(sl,tp,be,tt,td,mh)
    exits=','.join(f'{k}:{v}'for k,v in sorted(r['exit'].items()))
    print(f"N={r['n']:4d} WR={r['wr']:.1f}% PF={r['pf']:.2f} ${r['pnl']:,.0f} AW={r['aw']:+.3f}R AL={r['al']:+.3f}R TP={r['tp']:.1f}%  [{exits}]")
