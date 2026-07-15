"""Test: rule-wins-conflict fix — before vs after."""
import MetaTrader5 as mt5, pandas as pd, numpy as np, torch, sys, os
from datetime import datetime
sys.path.insert(0, 'D:/FiananceBot/BTC_BOT')
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.cnn_gru_m15 import CNNGRUM15
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType

cfg = BTCConfig()
engine = BTCFeatureEngine()

encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir+'/btc_h1_encoder.pt', map_location='cpu', weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])

m15 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
m15.load_state_dict(torch.load(cfg.model_dir+'/btc_m15_model.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2,11,18,19,21,22,23}

def run_ytd(with_fix):
    mt5.initialize()
    h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, datetime(2026,1,1), datetime(2026,5,25,12))
    m15r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15, datetime(2026,1,1), datetime(2026,5,25,12))
    mt5.shutdown()
    h1 = pd.DataFrame(h1r).rename(columns={'time':'ts','tick_volume':'volume'})
    m15d = pd.DataFrame(m15r).rename(columns={'time':'ts','tick_volume':'volume'})
    h1['ts'] = pd.to_datetime(h1['ts'], unit='s', utc=True)
    m15d['ts'] = pd.to_datetime(m15d['ts'], unit='s', utc=True)
    h1 = h1.sort_values('ts').reset_index(drop=True)
    m15d = m15d.sort_values('ts').reset_index(drop=True)
    h1f = engine.compute(h1); m15f = engine.compute(m15d)

    m15c = np.zeros(len(m15d), dtype=np.float32); m15b = np.zeros(len(m15d), dtype=np.float32)
    for i in range(cfg.seq_len_m15, len(m15d)):
        seq = engine.compute_sequence(m15f, i, cfg.seq_len_m15)
        with torch.no_grad():
            o = m15(torch.from_numpy(seq).unsqueeze(0))
        m15c[i] = float(o['entry_confidence'].squeeze().numpy())
        m15b[i] = float(o['direction_bias'].squeeze().numpy())

    rd = RuleBasedRegimeDetector()
    for i in range(cfg.seq_len_h1):
        rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

    listening=False; h1_sig=0; bl=0; pos=0; tm=None; trades=[]; last_h1=-1
    overrides = 0

    for mi in range(cfg.seq_len_h1*4, len(m15d)):
        ts=m15d['ts'].iloc[mi];price=float(m15d['close'].iloc[mi])
        mc=float(m15c[mi]);bias=float(m15b[mi])
        hi=int((h1['ts']<=ts).sum()-1)
        if hi>=cfg.seq_len_h1 and hi!=last_h1:
            last_h1=hi
            seq=engine.compute_sequence(h1f,hi,cfg.seq_len_h1)
            t=torch.from_numpy(seq).unsqueeze(0)
            for j in range(max(0,hi-13),hi+1):
                rd.update(float(h1['high'].iloc[j]),float(h1['low'].iloc[j]),float(h1['close'].iloc[j]))
            rr=classify_regime(encoder,classifier,t,rd,cfg.min_regime_confidence,temperature=4.0)

            if with_fix:
                rule_out = rd._classify()
                trending = {'TREND_UP', 'TREND_DOWN'}
                if (rr['regime'] in trending and rule_out['regime'] in trending
                    and rr['regime'] != rule_out['regime']):
                    rr['regime'] = rule_out['regime']
                    rr['confidence'] = rule_out['confidence']
                    rr['source'] = 'rule'
                    overrides += 1

            gd=gate.evaluate(rr['regime'],rr['confidence'],float(rr.get('atr_percentile',0.5)),bb_position=float(h1f[hi,4]))
            if gd.entry_signal:
                hc=h1['close'].values[:hi+1]
                if len(hc)>=23:
                    e22=pd.Series(hc).ewm(span=22,adjust=False).mean().values
                    slp=(e22[-1]-e22[-2])/max(abs(float(e22[-2])),1e-12)
                    if (gd.direction==1 and slp>0) or (gd.direction==-1 and slp<0):
                        h1_sig=gd.direction;listening=True;bl=0
                    else:listening=False
                else:listening=False
            else:listening=False

        if pos!=0 and tm is not None and tm.state is not None:
            s=tm.state;hip=float(m15d['high'].iloc[mi]);lop=float(m15d['low'].iloc[mi])
            epx=None;reason=''
            if tm.check_sl_hit(lop,hip):epx=tm.exit_price_at_sl();reason='sl_hit'
            elif tm.check_tp_hit(lop,hip):epx=tm.exit_price_at_tp();reason='tp_hit'
            else:
                act=tm.update(price,hip,lop,h1f[hi,6]*price)
                if act.action_type==TradeActionType.CLOSE:epx=price;reason=act.reason
            if epx:
                d=1 if pos==1 else -1
                if d==1:pnl_d=(epx-s.entry_price)*s.lots
                else:pnl_d=(s.entry_price-epx)*s.lots
                pnl_r=pnl_d/max(s.entry_atr*s.lots*cfg.initial_sl,1e-9)
                trades.append({'pnl_d':round(pnl_d,2),'pnl_r':round(float(pnl_r),4),'reason':reason,
                               'month':str(ts)[:7]})
                pos=0;tm=None
                continue

        if not listening:continue
        bl+=1
        if bl>cfg.max_listen_bars:listening=False;continue
        if ts.hour in BLOCKED:continue
        ok=mc>=cfg.min_entry_confidence and ((h1_sig==1 and bias>0) or (h1_sig==-1 and bias<0))
        if ok:
            h1_atr=float(h1f[hi,6]*price)
            tm=TradeManager(initial_sl=cfg.initial_sl,hard_tp=cfg.hard_tp,
                breakeven_trigger=cfg.breakeven_trigger,trail_trigger=cfg.trail_trigger,
                trail_dist=cfg.trail_dist,trail_dist_s=cfg.trail_dist_s,
                regime_tighten=cfg.regime_tighten,
                max_hold=cfg.max_hold_bars,mae_guard_retrace=cfg.mae_guard_retrace)
            lots=TradeManager.compute_position_size(10000.0,h1_atr,price,cfg.risk_pct,cfg.initial_sl)
            tm.enter(h1_sig,price,h1_atr,lots,regime=rr['regime'])
            pos=h1_sig;listening=False

    if not trades: return None
    n=len(trades);wins=[t for t in trades if t['pnl_d']>0];losses=[t for t in trades if t['pnl_d']<0]
    wr=len(wins)/n*100 if n else 0
    rs=np.array([t['pnl_r'] for t in trades])
    tg=sum(r for r in rs if r>0);tl=abs(sum(r for r in rs if r<=0))
    pf=tg/max(tl,0.001);total_pnl=sum(t['pnl_d'] for t in trades)
    avg_r=np.mean(rs);total_r=rs.sum()
    tp=sum(1 for t in trades if 'tp_hit' in t['reason'])
    sl=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']<-0.5)
    be=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']>-0.5)
    sharpe=avg_r/max(np.std(rs),0.001) if n>1 else 0
    elo=1500+(wr-50)*10+min((pf-1)*300,500)+min(sharpe*100,300)+min(total_r*10,500)
    return {'n':n,'wr':wr,'pf':pf,'pnl':total_pnl,'avg_r':avg_r,'total_r':total_r,
            'elo':max(0,min(3000,elo)),'tp':tp,'sl':sl,'be':be,'overrides':overrides,
            'trades':trades}

print('Running BASELINE...')
base = run_ytd(False)
print('Running FIX...')
fix = run_ytd(True)

print()
print('='*65)
print('BEFORE vs AFTER — Rule wins TREND conflicts')
print('='*65)
print('{:25s} {:>12s} {:>12s} {:>10s}'.format('','BASELINE','FIX','DELTA'))
print('-'*60)
for key, label, fmt in [
    ('n','Trades','d'),('wr','WR%','.1f'),('pf','PF','.2f'),
    ('pnl','PnL','.1f'),('avg_r','AvgR','.3f'),('elo','ELO','.0f'),
    ('tp','TP hits','d'),('sl','Full SL','d'),('be','BE hits','d')]:
    b=base[key]; f=fix[key]
    if fmt=='d':
        print('{:25s} {:>12d} {:>12d} {:>+10d}'.format(label,int(b),int(f),int(f)-int(b)))
    else:
        print(('{:25s} {:>12'+fmt+'} {:>12'+fmt+'} {:>+10'+fmt+'}').format(label,b,f,f-b))

print()
print('Conflict overrides: {}'.format(fix['overrides']))

# Monthly
print()
print('{:10s} {:>10s} {:>10s} {:>10s}'.format('Month','Base PnL','Fix PnL','Delta'))
df_base = pd.DataFrame(base['trades'])
df_fix = pd.DataFrame(fix['trades'])
for month in sorted(set(df_base['month'].unique()) | set(df_fix['month'].unique())):
    bp = df_base[df_base['month']==month]['pnl_d'].sum() if month in df_base['month'].values else 0
    fp = df_fix[df_fix['month']==month]['pnl_d'].sum() if month in df_fix['month'].values else 0
    print('{:10s} ${:>+9.1f} ${:>+9.1f} ${:>+9.1f}'.format(month,bp,fp,fp-bp))
