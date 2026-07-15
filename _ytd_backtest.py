"""YTD backtest with monthly PnL breakdown"""
import sys,os;sys.path.insert(0,".")
import numpy as np,pandas as pd,torch
from collections import defaultdict
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config=BTCConfig();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

tm=TradeManager(initial_sl=config.initial_sl,hard_tp=config.hard_tp,breakeven_trigger=config.breakeven_trigger,trail_trigger=config.trail_trigger,trail_dist=config.trail_dist,trail_dist_s=config.trail_dist_s,regime_tighten=config.regime_tighten,max_hold=config.max_hold_bars,mae_guard_retrace=config.mae_guard_retrace)
exec=DryRunExecutor(symbol=config.symbol,initial_balance=10000.0)
bal=10000.0;pnl_d=0.0;ld=None;trades=[];sb=10000.0
h1_sig=None;listen=False;bl=0;rd=RuleBasedRegimeDetector();lh=None;h1_atr=0.0;lots=0.0;pos=0;ab=[]
entry_regime='';entry_conf=0.0;equity=10000.0;max_equity=10000.0;dd_pct=0.0

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
            equity+=pnl_dollar
            if equity>max_equity:max_equity=equity
            dd=max(0,(max_equity-equity)/max_equity*100)
            if dd>dd_pct:dd_pct=dd
            mfe_peak=max(b['mfe']for b in ab)if ab else 0.0
            trades.append({'pnl_r':round(pnl_r,4),'pnl_dollar':round(pnl_dollar,2),'mfe_peak':round(mfe_peak,4),'exit_reason':er,'month':ts.month,'lots':round(lots,6),'entry_atr':round(h1_atr,2),'entry_price':round(price,2)})
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

monthly=defaultdict(lambda:{'pnl':0,'trades':0,'wins':0,'losses':0,'tp':0})
for t in trades:
    m=t['month'];monthly[m]['pnl']+=t['pnl_dollar'];monthly[m]['trades']+=1
    if t['pnl_r']>0:monthly[m]['wins']+=1
    else:monthly[m]['losses']+=1
    if t['exit_reason']=='tp_hit':monthly[m]['tp']+=1

print('='*80)
print('BTC BOT — YTD BACKTEST (Jan 1 – May 20, 2026)')
print(f'Config: BE={config.breakeven_trigger} TT={config.trail_trigger} TD={config.trail_dist} MH={config.max_hold_bars} SL={config.initial_sl} TP={config.hard_tp}')
print(f'Filters: hour+trend+v2@0.5')
print('='*80)
print(f"{'Month':<10s} {'Trades':>7s} {'Wins':>6s} {'Losses':>7s} {'WR':>7s} {'TP':>5s} {'PnL':>10s} {'Return':>8s} {'CumPnL':>10s}")
print('-'*80)
cum=0
for m in sorted(monthly):
    d=monthly[m];wr_m=d['wins']/d['trades']*100 if d['trades'] else 0
    ret_m=d['pnl']/10000*100;cum+=d['pnl']
    mn=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]
    print(f"{mn:<10s} {d['trades']:7d} {d['wins']:6d} {d['losses']:7d} {wr_m:6.1f}% {d['tp']:4d} ${d['pnl']:>9,.0f} {ret_m:7.1f}% ${cum:>9,.0f}")

print('-'*80)
tp_total=sum(1 for t in trades if t['exit_reason']=='tp_hit')
print(f"{'TOTAL':<10s} {n:7d} {len(wins):6d} {len(losses):7d} {wr:6.1f}% {tp_total:4d} ${total_pnl:>9,.0f} {total_pnl/10000*100:7.1f}%")

print(f"\n  Profit Factor: {pf:.2f}  |  Win Rate: {wr:.1f}%  |  Max DD: {dd_pct:.1f}%")
print(f"  Avg Win: {np.mean([t['pnl_r']for t in wins]):+.3f}R  |  Avg Loss: {np.mean([t['pnl_r']for t in losses]):+.3f}R")
print(f"  Expectancy: {np.mean([t['pnl_r']for t in trades]):+.4f}R/trade")
print(f"  Start: $10,000  ->  Final: ${equity:,.0f}  ({(equity/10000-1)*100:+.1f}%)")

print(f"\n  Exit breakdown: sl_hit={sum(1 for t in trades if t['exit_reason']=='sl_hit')}, tp_hit={tp_total}, time_stop={sum(1 for t in trades if 'Time' in t['exit_reason'])}")

# ── Lot Size Analysis ──
lots_vals=[t['lots'] for t in trades]
atrs=[t['entry_atr'] for t in trades]
print(f"\n{'='*80}")
print("LOT SIZE ANALYSIS")
print("="*80)
print(f"  Lot size range: {min(lots_vals):.6f} – {max(lots_vals):.6f} BTC")
print(f"  Mean lot: {np.mean(lots_vals):.6f}  |  Median: {np.median(lots_vals):.6f}  |  SD: {np.std(lots_vals):.6f}")
print(f"  ATR range: ${min(atrs):,.0f} – ${max(atrs):,.0f}  |  Mean ATR: ${np.mean(atrs):,.0f}")

# Bucket lot sizes into tertiles
p33=np.percentile(lots_vals,33);p67=np.percentile(lots_vals,67)
buckets=[(0,p33,'Small'),(p33,p67,'Medium'),(p67,999,'Large')]
print(f"\n  Lot size buckets (tertiles):")
print(f"    Small:  <{p33:.4f} BTC")
print(f"    Medium: {p33:.4f}–{p67:.4f} BTC")
print(f"    Large:  >{p67:.4f} BTC")
print(f"\n  {'Bucket':<10s} {'Trades':>7s} {'Wins':>6s} {'Losses':>7s} {'WR':>7s} {'AvgPnLR':>9s} {'SD_PnLR':>9s} {'AvgMFE':>9s} {'AvgATR':>10s} {'AvgLot':>10s}")
print("  "+"-"*85)
for lo,hi,label in buckets:
    b=[t for t in trades if lo<=t['lots']<hi]
    if not b:continue
    bw=[t for t in b if t['pnl_r']>0];bl=[t for t in b if t['pnl_r']<=0]
    wr_b=len(bw)/len(b)*100;avg_r=np.mean([t['pnl_r']for t in b]);sd_r=np.std([t['pnl_r']for t in b])
    avg_mfe=np.mean([t['mfe_peak']for t in b]);avg_atr=np.mean([t['entry_atr']for t in b])
    avg_lot=np.mean([t['lots']for t in b])
    print(f"  {label:<10s} {len(b):7d} {len(bw):6d} {len(bl):7d} {wr_b:6.1f}% {avg_r:+8.4f}R {sd_r:8.4f}R {avg_mfe:+8.4f}R ${avg_atr:>9,.0f} {avg_lot:9.6f}")

# Also bucket by ATR (the driver of lot size variation)
atr_p33=np.percentile(atrs,33);atr_p67=np.percentile(atrs,67)
print(f"\n  ATR buckets (tertiles — ATR drives lot size):")
print(f"    Low ATR:   <${atr_p33:,.0f}")
print(f"    Medium ATR: ${atr_p33:,.0f}–${atr_p67:,.0f}")
print(f"    High ATR:  >${atr_p67:,.0f}")
print(f"\n  {'ATR':<12s} {'Trades':>7s} {'WR':>7s} {'AvgPnLR':>9s} {'AvgMFE':>9s} {'AvgLot':>10s}")
print("  "+"-"*65)
for lo,hi,label in [(0,atr_p33,'Low Vol'),(atr_p33,atr_p67,'Med Vol'),(atr_p67,999999,'High Vol')]:
    b=[t for t in trades if lo<=t['entry_atr']<hi]
    if not b:continue
    bw=[t for t in b if t['pnl_r']>0]
    wr_b=len(bw)/len(b)*100;avg_r=np.mean([t['pnl_r']for t in b])
    avg_mfe=np.mean([t['mfe_peak']for t in b]);avg_lot=np.mean([t['lots']for t in b])
    print(f"  {label:<12s} {len(b):7d} {wr_b:6.1f}% {avg_r:+8.4f}R {avg_mfe:+8.4f}R {avg_lot:9.6f}")

# Correlation: lot size vs PnL
from scipy import stats
r,_=stats.pearsonr(lots_vals,[t['pnl_r']for t in trades])
print(f"\n  Correlation (lot size vs PnL_R): r={r:.3f}")
print(f"  (Negative = smaller lots perform better, Positive = larger lots perform better)")
