"""Full YTD comparison: current vs new SL/TP with replacement trade analysis."""
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

def run_full(sl,tp,be,tt,label):
    tm=TradeManager(initial_sl=sl,hard_tp=tp,breakeven_trigger=be,trail_trigger=tt,trail_dist=0.75,trail_dist_s=0.50,regime_tighten=0.40,max_hold=18,mae_guard_retrace=2.5)
    exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
    bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0;equity=10000.0;max_eq=10000.0;dd=0.0
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
            else:h1_sig=None;listen=False
        if pos!=0 and tm.state is not None:
            hi=m15s['high'].iloc[-1];lo=m15s['low'].iloc[-1];epx=None;er=None;s2=tm.state;sd2=sl*s2.entry_atr
            mfe_now=(hi-s2.entry_price)/sd2 if pos==1 else(s2.entry_price-lo)/sd2
            ab.append({'bar':len(ab),'mfe':float(mfe_now)})
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
                dd=max(dd,(max_eq-equity)/max_eq*100)if max_eq>0 else 0
                mfe_peak=max(b['mfe']for b in ab)if ab else 0.0
                trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'mfe':round(mfe_peak,4),'exit':er,'entry_ts':ts,'bars':len(ab),'atr':s2.entry_atr,'dir':'LONG' if pos==1 else 'SHORT'})
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
        lots=tm.compute_position_size(bal,h1_atr,price,config.risk_pct,sl)
        tm.enter(h1_sig,price,h1_atr,lots)
        exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
        pos=h1_sig
    return trades,bal,dd

print('Running current config (1:2.5)...')
cur_trades,cur_bal,cur_dd=run_full(1.0,2.5,0.50,2.0,'current')
print('Running new config (0.5:1.2)...')
new_trades,new_bal,new_dd=run_full(0.5,1.2,0.25,1.2,'new')

def analyze(trades,bal,dd,label):
    wins=[t for t in trades if t['pnl_r']>0];losses=[t for t in trades if t['pnl_r']<=0]
    n=len(trades);wr=len(wins)/n*100 if n else 0
    tg=sum(t['pnl_r']for t in wins);tl=abs(sum(t['pnl_r']for t in losses))
    pf=tg/max(tl,0.001);pnl=sum(t['pnl_dollar']for t in trades)
    aw=np.mean([t['pnl_r']for t in wins])if wins else 0
    al=np.mean([t['pnl_r']for t in losses])if losses else 0
    tp_n=sum(1 for t in trades if t['exit']=='tp_hit')
    sl_n=sum(1 for t in trades if t['exit']=='sl_hit')
    ts_n=sum(1 for t in trades if 'Time' in t['exit'] or 'time' in t['exit'])
    # Noise trades: loss with MFE < 0.25*SL
    noise=[t for t in losses if t['mfe']<0.25]
    # Trades that had profit (MFE>0.5R) but flipped to loss
    flipped=[t for t in losses if t['mfe']>=0.5]
    # Back-to-back losses
    bb=sum(1 for j in range(1,n) if trades[j]['pnl_r']<=0 and trades[j-1]['pnl_r']<=0)
    # Profit captured: MFE / exit PnL ratio for wins
    mfe_win=np.mean([t['mfe']for t in wins])if wins else 0
    mfe_loss=np.mean([t['mfe']for t in losses])if losses else 0
    return{'label':label,'n':n,'wr':wr,'pf':pf,'pnl':pnl,'bal':bal,'dd':dd,
           'aw':aw,'al':al,'tp':tp_n,'tp_pct':tp_n/n*100,'sl':sl_n,'sl_pct':sl_n/n*100,
           'ts':ts_n,'noise':len(noise),'noise_pct':len(noise)/n*100,
           'flipped':len(flipped),'flipped_pct':len(flipped)/n*100,
           'bb_losses':bb,'bb_pct':bb/max(n-1,1)*100,
           'mfe_win':mfe_win,'mfe_loss':mfe_loss}

c=analyze(cur_trades,cur_bal,cur_dd,'Current 1:2.5')
n=analyze(new_trades,new_bal,new_dd,'New 0.5:1.2')

print('\n'+'='*90)
print('YTD COMPARISON — Current vs New SL/TP (Jan 1 – May 20)')
print('='*90)
HDR="  %-20s %6s %6s %6s %10s %7s %7s %5s %5s %5s %6s %6s %6s %6s" % ('','Trades','WR','PF','PnL','AvgW','AvgL','TP%','SL%','DD%','Noise%','Flip%','BB%','MFEw')
print(HDR)
print('  '+'-'*90)
for r in [c,n]:
    print("  %-20s %6d %5.1f%% %5.2f $%9.0f %+6.3fR %+6.3fR %4.1f%% %4.1f%% %4.1f%% %5.1f%% %5.1f%% %4.1f%% %+5.2fR" % (
        r['label'],r['n'],r['wr'],r['pf'],r['pnl'],r['aw'],r['al'],
        r['tp_pct'],r['sl_pct'],r['dd'],r['noise_pct'],r['flipped_pct'],r['bb_pct'],r['mfe_win']))

print('\nDelta: Trades=%+d WR=%+.1fpp PF=%+.2f PnL=$%+.0f Noise=%+.1fpp Flipped=%+.1fpp BB=%+.1fpp' % (
    n['n']-c['n'],n['wr']-c['wr'],n['pf']-c['pf'],n['pnl']-c['pnl'],
    n['noise_pct']-c['noise_pct'],n['flipped_pct']-c['flipped_pct'],n['bb_pct']-c['bb_pct']))

# Monthly breakdown
print('\n'+'='*90)
print('MONTHLY PnL')
print('='*90)
for label,td in [('Current',cur_trades),('New',new_trades)]:
    monthly=defaultdict(float)
    for t in td:
        monthly[t['entry_ts'].month]+=t['pnl_dollar']
    parts=['M%d:$%+.0f'%(m,v) for m,v in sorted(monthly.items())]
    print('  %s: %s' % (label, ' | '.join(parts)))

# Exit reason breakdown
print('\nExit reasons:')
for label,td in [('Current',cur_trades),('New',new_trades)]:
    ec=Counter(t['exit']for t in td)
    parts=['%s:%d(%.0f%%)'%(k,v,v/len(td)*100)for k,v in ec.most_common()]
    print('  %s: %s'%(label,', '.join(parts)))

# Win distribution
print('\nWin size distribution (in R-multiples):')
for label,td in [('Current',cur_trades),('New',new_trades)]:
    wins=[t for t in td if t['pnl_r']>0]
    buckets=[(0,0.25),(0.25,0.5),(0.5,1.0),(1.0,2.0),(2.0,99)]
    parts=[]
    for lo,hi in buckets:
        cnt=sum(1 for t in wins if lo<=t['pnl_r']<hi)
        parts.append('%sR:%d'%('TP' if hi==99 else '%.2f-%.2f'%(lo,hi),cnt))
    print('  %s: %s'%(label,', '.join(parts)))
