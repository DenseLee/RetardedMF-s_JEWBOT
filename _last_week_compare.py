"""Compare current vs new SL/TP config on last week's trades with replacement analysis."""
import MetaTrader5 as mt5,pandas as pd,numpy as np,torch,sys,os
sys.path.insert(0,"D:/FiananceBot/BTC_BOT")
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier,RuleBasedRegimeDetector,classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager,TradeActionType
from execution.mt5_executor_btc import DryRunExecutor
from config_btc import BTCConfig

config=BTCConfig();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder=CNNLSTMEncoder(n_features=config.n_features,seq_len=config.seq_len_h1,cnn_channels=config.cnn_channels,lstm_hidden=config.lstm_hidden,lstm_layers=config.lstm_layers,dropout=config.lstm_dropout,embedding_dim=config.embedding_dim,regime_classes=config.regime_classes,bidirectional=True).to(device).eval()
classifier=RegimeClassifier(embedding_dim=config.embedding_dim,n_classes=config.regime_classes).to(device).eval()
ckpt=torch.load(config.model_dir+'/btc_h1_encoder.pt',map_location=device,weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict']);classifier.load_state_dict(ckpt['classifier_state_dict'])
m15_v2=CNNGRUM15(n_features=config.n_features,seq_len=config.seq_len_m15,cnn_channels=config.gru_cnn_channels,gru_hidden=config.gru_hidden,gru_layers=config.gru_layers,dropout=config.gru_dropout).to(device).eval()
mc2=torch.load(config.model_dir+'/btc_m15_v2.pt',map_location=device,weights_only=False)
m15_v2.load_state_dict(mc2['model_state_dict'],strict=False)
engine=BTCFeatureEngine();gate=EntryGate()
BLOCKED={2,11,18,19,21,22,23}

# Fetch MT5 bars for last week
mt5.initialize()
h1_rates=mt5.copy_rates_from_pos('BTCUSD',mt5.TIMEFRAME_H1,0,200)
m15_rates=mt5.copy_rates_from_pos('BTCUSD',mt5.TIMEFRAME_M15,0,600)
h1f=pd.DataFrame(h1_rates);h1f.rename(columns={'tick_volume':'volume'},inplace=True)
h1f['timestamp']=pd.to_datetime(h1f['time'],unit='s')
m15f=pd.DataFrame(m15_rates);m15f.rename(columns={'tick_volume':'volume'},inplace=True)
m15f['timestamp']=pd.to_datetime(m15f['time'],unit='s')
# Filter to last 3 days
ft=pd.Timestamp('2026-05-20',tz=None);et=pd.Timestamp('2026-05-23',tz=None)
h1f=h1f[(h1f['timestamp']>=ft)&(h1f['timestamp']<et)].reset_index(drop=True)
m15f=m15f[(m15f['timestamp']>=ft)&(m15f['timestamp']<et)].reset_index(drop=True)
print(f'Data: {len(h1f)} H1 bars, {len(m15f)} M15 bars ({h1f["timestamp"].min()} to {h1f["timestamp"].max()})')

def run_config(sl,tp,be,tt,label):
    tm=TradeManager(initial_sl=sl,hard_tp=tp,breakeven_trigger=be,trail_trigger=tt,trail_dist=sl*0.75,trail_dist_s=sl*0.50,regime_tighten=0.40,max_hold=18,mae_guard_retrace=2.5)
    exec=DryRunExecutor(symbol='BTCUSD',initial_balance=10000.0)
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
            else:h1_sig=None;listen=False
        if pos!=0 and tm.state is not None:
            hi=m15s['high'].iloc[-1];lo=m15s['low'].iloc[-1];epx=None;er=None;s2=tm.state;sd2=sl*s2.entry_atr
            if tm.check_sl_hit(lo,hi):epx=tm.exit_price_at_sl();er='sl_hit'
            elif tm.check_tp_hit(lo,hi):epx=tm.exit_price_at_tp();er='tp_hit'
            else:
                a=tm.update(price,hi,lo,h1_atr)
                if a.action_type==TradeActionType.CLOSE:epx=price;er=a.reason
            if epx:
                pnl_r=(epx-s2.entry_price)/sd2 if pos==1 else(s2.entry_price-epx)/sd2
                pnl_dollar=(epx-s2.entry_price)*lots if pos==1 else(s2.entry_price-epx)*lots
                bal+=pnl_dollar;pnl_d+=pnl_dollar
                trades.append({'entry_ts':s2.bars_held, 'exit_ts':ts,'dir':'LONG' if pos==1 else 'SHORT',
                    'entry_price':s2.entry_price,'exit_price':epx,'pnl_r':round(pnl_r,4),
                    'pnl_dollar':round(pnl_dollar,2),'exit':er,'atr':s2.entry_atr,
                    'bars_held':len(ab)+1,'mfe':round(max(b.get('mfe',0)for b in ab)if ab else 0,4)})
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
        listen=False
        lots=tm.compute_position_size(bal,h1_atr,price,config.risk_pct,sl)
        tm.enter(h1_sig,price,h1_atr,lots)
        exec.open_position(h1_sig,lots,tm.state.current_sl,tm.state.current_tp)
        pos=h1_sig
    return trades

print('\nRunning current config (1:2.5)...')
current=run_config(1.0,2.5,0.50,2.0,'current')
print(f'Current: {len(current)} trades')
for i,t in enumerate(current):
    ep=t['entry_price'];xp=t['exit_price'];atr=t['atr'];dr=t['dir'];pnl=t['pnl_r'];ex=t['exit'];bh=t['bars_held']
    print('  T%d: %s @ %d -> %d (%+.2fR) [%s] ATR=%d bars=%d' % (i+1,dr,ep,xp,pnl,ex,atr,bh))

print('\nRunning new config (0.5:1.2)...')
new=run_config(0.5,1.2,0.25,1.2,'new')
print(f'New: {len(new)} trades')
for i,t in enumerate(new):
    ep=t['entry_price'];xp=t['exit_price'];atr=t['atr'];dr=t['dir'];pnl=t['pnl_r'];ex=t['exit'];bh=t['bars_held']
    print('  T%d: %s @ %d -> %d (%+.2fR) [%s] ATR=%d bars=%d' % (i+1,dr,ep,xp,pnl,ex,atr,bh))

# Analyze: which new trades are replacements? Compare sequences
print('\n' + '='*70)
print('REPLACEMENT ANALYSIS')
print('='*70)
c_times=[(t['entry_ts'],t['exit_ts']) for t in current]
n_times=[(t['entry_ts'],t['exit_ts']) for t in new]
# Trades not in current = replacements
# Show back-to-back patterns
seq=[];prev_exit=None
for i,t in enumerate(new):
    gap=''
    if i>0:
        prev=new[i-1]
        gap_min=(t['entry_ts']-prev['exit_ts']).total_seconds()/60 if hasattr(t['entry_ts'],'total_seconds') else 0
        gap=f' ({gap_min:.0f}min after prev exit)'
    outcome='WIN' if t['pnl_r']>0 else 'LOSS'
    matched=any(abs((t['entry_ts']-ct[0]).total_seconds())<300 for ct in c_times) if hasattr(t['entry_ts'],'total_seconds') else False
    tag='[SAME]' if matched else '[NEW - replacement]'
    dr=t['dir'];pnl=t['pnl_r']
    print(f'  {outcome:<5s} {tag:<20s} {dr} {pnl:+.2f}R {gap}')

# Count losses back-to-back
bb_losses=0
for i in range(1,len(new)):
    if new[i]['pnl_r']<=0 and new[i-1]['pnl_r']<=0:
        bb_losses+=1
print(f'\n  Back-to-back losses: {bb_losses}/{len(new)-1} pairs')
cpnl=sum(t['pnl_dollar'] for t in current)
npnl=sum(t['pnl_dollar'] for t in new)
print(f'  Current config net PnL: ${cpnl:,.2f}')
print(f'  New config net PnL:     ${npnl:,.2f}')
