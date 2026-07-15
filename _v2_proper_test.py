"""Proper V1 vs V2 comparison using inline backtest with pre-sampled H4 data."""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from collections import Counter,defaultdict
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config=BTCConfig();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# H1 encoder
encoder=CNNLSTMEncoder(n_features=config.n_features,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=config.regime_classes,bidirectional=True).to(device).eval()
from models.regime_classifier import RegimeClassifier
classifier=RegimeClassifier(embedding_dim=config.embedding_dim,n_classes=config.regime_classes).to(device).eval()
ckpt=torch.load(config.model_dir+'/btc_h1_encoder.pt',map_location=device,weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])
# M15 v2
m15_v2=CNNGRUM15(n_features=config.n_features,seq_len=config.seq_len_m15,cnn_channels=config.gru_cnn_channels,gru_hidden=config.gru_hidden,gru_layers=config.gru_layers,dropout=config.gru_dropout).to(device).eval()
mc2=torch.load(config.model_dir+'/btc_m15_v2.pt',map_location=device,weights_only=False)
m15_v2.load_state_dict(mc2['model_state_dict'],strict=False)
# H4 encoder (optional)
h4_encoder=None
h4_path=config.model_dir+'/btc_h4_encoder.pt'
if os.path.exists(h4_path):
    h4_ckpt=torch.load(h4_path,map_location=device,weights_only=False)
    h4_encoder=CNNLSTMEncoder(n_features=config.n_features,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=config.regime_classes,bidirectional=True).to(device).eval()
    h4_encoder.load_state_dict(h4_ckpt['encoder_state_dict'])
    print("H4 encoder loaded")

engine=BTCFeatureEngine();gate=EntryGate()

# Load data
h1f=pd.read_csv(config.data_dir+'/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv');h1f['timestamp']=pd.to_datetime(h1f['timestamp'],utc=True)
m15f=pd.read_csv(config.data_dir+'/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv');m15f['timestamp']=pd.to_datetime(m15f['timestamp'],utc=True)
ft=pd.Timestamp('2026-01-01',tz='UTC');et=pd.Timestamp('2026-05-20',tz='UTC')
h1f=h1f[(h1f['timestamp']>=ft)&(h1f['timestamp']<et)].reset_index(drop=True)
m15f=m15f[(m15f['timestamp']>=ft)&(m15f['timestamp']<et)].reset_index(drop=True)

# Pre-sample H4 data from H1
h1_indexed=h1f.set_index('timestamp')
h4f=h1_indexed.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
h4_feats_full=engine.compute(h4f) if h4_encoder else None
print(f"H1: {len(h1f)} bars, H4: {len(h4f)} bars")

BLOCKED={2,11,18,19,21,22,23}

def get_h4_regime(ts):
    """Get H4 regime at a given timestamp using pre-sampled data."""
    if h4_encoder is None or h4_feats_full is None: return None
    h4_slice=h4f[h4f['timestamp']<=ts]
    if len(h4_slice)<96: return None
    h4_feats=h4_feats_full[h4f['timestamp']<=ts][-min(96,len(h4_slice)):]
    if len(h4_feats)<96: return None
    # Pad to 96 if needed
    if len(h4_feats)<96:
        pad=np.zeros((96-len(h4_feats),17),dtype=np.float32)
        h4_feats=np.vstack([pad,h4_feats])
    seq=h4_feats[-96:]
    t=torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        out=h4_encoder(t)
    idx=out['regime_logits'].argmax(1).item()
    return ['TREND_UP','TREND_DOWN','RANGE','TRANSITION'][idx]

def run(use_h4=False):
    tm=TradeManager(initial_sl=config.initial_sl,hard_tp=config.hard_tp,breakeven_trigger=config.breakeven_trigger,trail_trigger=config.trail_trigger,trail_dist=config.trail_dist,trail_dist_s=config.trail_dist_s,regime_tighten=config.regime_tighten,max_hold=config.max_hold_bars,mae_guard_retrace=config.mae_guard_retrace)
    exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
    bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0
    h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]
    entry_regime='';entry_conf=0.0;equity=10000.0;max_eq=10000.0;dd_pct=0.0
    h4_blocks=0

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
                # H1 trend filter
                h1_closes=h1s['close'].values
                if len(h1_closes)>=23:
                    h1_ema22=pd.Series(h1_closes).ewm(span=22,adjust=False).mean().values
                    h1_slope=(h1_ema22[-1]-h1_ema22[-2])/max(abs(float(h1_ema22[-2])),1e-12)
                    with_trend=((g.direction==1 and h1_slope>0)or(g.direction==-1 and h1_slope<0))
                    if not with_trend:h1_sig=None;listen=False;continue
                # H4 trend gate (V2)
                if use_h4:
                    h4_regime=get_h4_regime(ts)
                    if h4_regime is not None:
                        h4_against=((g.direction==1 and h4_regime=='TREND_DOWN')or(g.direction==-1 and h4_regime=='TREND_UP'))
                        if h4_against:h4_blocks+=1;h1_sig=None;listen=False;continue
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
                bal+=pnl_dollar;pnl_d+=pnl_dollar;equity+=pnl_dollar
                if equity>max_eq:max_eq=equity
                dd=max(0,(max_eq-equity)/max_eq*100)
                if dd>dd_pct:dd_pct=dd
                mfe_peak=max(b['mfe']for b in ab)if ab else 0.0
                trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'mfe_peak':round(mfe_peak,4),'exit_reason':er,'month':ts.month})
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
    pf=tg/max(tl,0.001);total_pnl=sum(t['pnl_dollar']for t in trades)
    tp=sum(1 for t in trades if t['exit_reason']=='tp_hit')
    sl_hits=sum(1 for t in trades if t['exit_reason']=='sl_hit')
    return{'n':n,'wr':wr,'pf':pf,'pnl':total_pnl,'aw':np.mean([t['pnl_r']for t in wins])if wins else 0,'al':np.mean([t['pnl_r']for t in losses])if losses else 0,'tp':tp,'sl':sl_hits,'dd':dd_pct,'equity':equity,'bal':bal,'h4_blocks':h4_blocks}


print("\nRunning V1 (no H4)...",flush=True)
v1=run(use_h4=False)
print(f"V1: {v1['n']} trades, WR={v1['wr']:.1f}%, PF={v1['pf']:.2f}, PnL=${v1['pnl']:,.0f}, DD={v1['dd']:.1f}%")

print("Running V2 (H4 gate)...",flush=True)
v2=run(use_h4=True)
print(f"V2: {v2['n']} trades, WR={v2['wr']:.1f}%, PF={v2['pf']:.2f}, PnL=${v2['pnl']:,.0f}, DD={v2['dd']:.1f}%, H4 blocks={v2['h4_blocks']}")

print("\n"+"="*80)
print("V1 vs V2 — 2026 YTD (Jan 1 – May 20)")
print("="*80)
print(f"  {'':<20s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'AvgW':>7s} {'AvgL':>7s} {'TP':>5s} {'SL':>6s} {'DD':>7s} {'Return':>8s}")
print("  "+"-"*90)
for r in [v1,v2]:
    ret=(r['equity']/10000-1)*100
    print(f"  {r['n']:7d} {r['wr']:6.1f}% {r['pf']:6.2f} ${r['pnl']:>9,.0f} {r['aw']:+6.3f}R {r['al']:+6.3f}R {r['tp']:4d} {r['sl']:5d} {r['dd']:6.1f}% {ret:+7.1f}%")
print(f"\n  Delta: Trades={v2['n']-v1['n']:+d}  WR={v2['wr']-v1['wr']:+.1f}pp  PF={v2['pf']-v1['pf']:+.2f}  PnL=${v2['pnl']-v1['pnl']:+,.0f}")
print(f"  H4 blocks: {v2['h4_blocks']}")
print(f"  V2 is {'*** BETTER ***' if v2['pf']>v1['pf'] else 'same' if v2['pf']==v1['pf'] else 'worse'}")
