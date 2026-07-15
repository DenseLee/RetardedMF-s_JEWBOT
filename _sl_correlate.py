"""Sweep SL ratios and find what correlates with needing wider/tighter SL."""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from collections import defaultdict,Counter
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
ft=pd.Timestamp('2026-01-01',tz='UTC');et=pd.Timestamp('2026-05-20',tz='UTC')
h1f=h1f[(h1f['timestamp']>=ft)&(h1f['timestamp']<et)].reset_index(drop=True)
m15f=m15f[(m15f['timestamp']>=ft)&(m15f['timestamp']<et)].reset_index(drop=True)
BLOCKED={2,11,18,19,21,22,23}

def run_sl(sl_val, record_meta=False):
    tp_val=sl_val*2.4
    tm=TradeManager(initial_sl=sl_val,hard_tp=tp_val,breakeven_trigger=sl_val*0.5,trail_trigger=tp_val*0.8,trail_dist=sl_val*0.75,trail_dist_s=sl_val*0.5,regime_tighten=0.40,max_hold=18,mae_guard_retrace=2.5)
    exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
    bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0
    h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]
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
                h1_sig=g.direction;listen=True;bl=0;h1_atr=h1_feats[-1,6]*price
                # Capture entry metadata
                entry_vr=h1_feats[-1,8] # volatility ratio (ATR14/ATR100)
                entry_bb=h1_feats[-1,4] # BB position
                entry_adx=h1_feats[-1,10] # ADX
                entry_regime=rr['regime']
            else:h1_sig=None;listen=False
        if pos!=0 and tm.state is not None:
            hi=m15s['high'].iloc[-1];lo=m15s['low'].iloc[-1];epx=None;er=None;s2=tm.state;sd2=sl_val*s2.entry_atr
            ab.append({'bar':len(ab)})
            if tm.check_sl_hit(lo,hi):epx=tm.exit_price_at_sl();er='sl_hit'
            elif tm.check_tp_hit(lo,hi):epx=tm.exit_price_at_tp();er='tp_hit'
            else:
                a=tm.update(price,hi,lo,h1_atr)
                if a.action_type==TradeActionType.CLOSE:epx=price;er=a.reason
            if epx:
                pnl_r=(epx-s2.entry_price)/sd2 if pos==1 else(s2.entry_price-epx)/sd2
                pnl_dollar=(epx-s2.entry_price)*lots if pos==1 else(s2.entry_price-epx)*lots
                bal+=pnl_dollar;pnl_d+=pnl_dollar
                if record_meta:
                    trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'exit':er,
                        'vr':entry_vr,'bb':entry_bb,'adx':entry_adx,'regime':entry_regime,'hour':ts.hour,'atr':h1_atr})
                else:
                    trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'exit':er})
                pos=0;tm.state=None;ab=[]
            continue
        if not listen:continue;bl+=1
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
        if abs(pnl_d)/max(sb,1)>=0.05:continue
        listen=False
        lots=tm.compute_position_size(bal,h1_atr,price,config.risk_pct,sl_val)
        tm.enter(h1_sig,price,h1_atr,lots)
        exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
        pos=h1_sig
    wins=[t for t in trades if t['pnl_r']>0];losses=[t for t in trades if t['pnl_r']<=0]
    n=len(trades);wr=len(wins)/n*100 if n else 0
    tg=sum(t['pnl_r']for t in wins);tl=abs(sum(t['pnl_r']for t in losses))
    pf=tg/max(tl,0.001);pnl=sum(t['pnl_dollar']for t in trades)
    return n,wr,pf,pnl,trades

# Phase 1: SL sweep
print('SL RATIO SWEEP (TP = SL * 2.4)')
print('%-12s %7s %6s %6s %10s %6s %7s %7s' % ('SL:TP','Trades','WR','PF','PnL','TP%','AvgW','AvgL'))
print('-'*65)
for sl in [0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
    n,wr,pf,pnl,td=run_sl(sl)
    tp_n=sum(1 for t in td if t['exit']=='tp_hit')
    aw=np.mean([t['pnl_r']for t in td if t['pnl_r']>0])if n else 0
    al=np.mean([t['pnl_r']for t in td if t['pnl_r']<=0])if n else 0
    print('%-12s %7d %5.1f%% %5.2f $%9.0f %5.1f%% %+6.3fR %+6.3fR' % (
        '%.1f:%.1f'%(sl,sl*2.4),n,wr,pf,pnl,tp_n/n*100 if n else 0,aw,al))

# Phase 2: Correlate entry conditions with SL sensitivity
print('\nRunning SL=0.5 and SL=0.7 with metadata...')
_,_,_,_,td05=run_sl(0.5,record_meta=True)
_,_,_,_,td07=run_sl(0.7,record_meta=True)

# Align by entry order (approximate since trade counts differ)
n_common=min(len(td05),len(td07))
# Classify: did wider SL help or hurt?
helped=[];hurt=[];both_lose=[];both_win=[]
for i in range(n_common):
    t5=td05[i];t7=td07[i]
    if t5['pnl_r']<=0 and t7['pnl_r']>0:helped.append(t5)  # wider SL saved it
    elif t5['pnl_r']>0 and t7['pnl_r']<=0:hurt.append(t5)   # wider SL made it worse
    elif t5['pnl_r']<=0 and t7['pnl_r']<=0:both_lose.append(t5)
    else:both_win.append(t5)

print('\nWider SL (0.5->0.7) impact on aligned trades:')
print('  Helped (loss->win): %d'%len(helped))
print('  Hurt (win->loss): %d'%len(hurt))
print('  Both lose: %d'%len(both_lose))
print('  Both win: %d'%len(both_win))

# Compare entry conditions between groups
print('\nEntry condition comparison:')
for label,group in [('HELPED',helped),('HURT',hurt),('BOTH_LOSE',both_lose),('BOTH_WIN',both_win)]:
    if not group:continue
    vrs=[t.get('vr',1) for t in group];bbs=[t.get('bb',0) for t in group]
    adxs=[t.get('adx',0) for t in group];atrs=[t.get('atr',0) for t in group]
    regimes=Counter(t.get('regime','?') for t in group)
    print('  %-12s: n=%4d  VR=%.2f  BB=%+.3f  ADX=%.3f  ATR=$%.0f  regimes=%s' % (
        label,len(group),np.mean(vrs)if vrs else 0,np.mean(bbs)if bbs else 0,
        np.mean(adxs)if adxs else 0,np.mean(atrs)if atrs else 0,
        dict(regimes.most_common(2))))

# Correlation: VR vs PnL at each SL level
print('\nVolatility Ratio correlation with PnL:')
for sl in [0.5,0.7]:
    _,_,_,_,td=run_sl(sl,record_meta=True)
    vrs=[t.get('vr',1) for t in td];pnls=[t['pnl_r'] for t in td]
    if len(vrs)>10:
        r=np.corrcoef(vrs,pnls)[0,1]
        # High VR vs Low VR
        med=np.median(vrs)
        high=[t for t in td if t.get('vr',1)>=med];low=[t for t in td if t.get('vr',1)<med]
        wr_h=sum(1 for t in high if t['pnl_r']>0)/max(len(high),1)*100
        wr_l=sum(1 for t in low if t['pnl_r']>0)/max(len(low),1)*100
        print('  SL=%.1f: r=%.3f  HighVR(>%.2f): WR=%.1f%%  LowVR: WR=%.1f%%'%(sl,r,med,wr_h,wr_l))
