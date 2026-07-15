"""Sweep breakeven_trigger on v1 and v2 — 1 month."""
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

# H1
encoder = CNNLSTMEncoder(n_features=cfg.n_features, seq_len=cfg.seq_len_h1,
    cnn_channels=cfg.cnn_channels, lstm_hidden=cfg.lstm_hidden,
    lstm_layers=cfg.lstm_layers, dropout=cfg.lstm_dropout,
    embedding_dim=cfg.embedding_dim, regime_classes=cfg.regime_classes,
    bidirectional=cfg.lstm_bidirectional).eval()
classifier = RegimeClassifier(embedding_dim=cfg.embedding_dim, n_classes=cfg.regime_classes).eval()
ckpt = torch.load(cfg.model_dir + '/btc_h1_encoder.pt', map_location='cpu', weights_only=False)
encoder.load_state_dict(ckpt['encoder_state_dict'])
classifier.load_state_dict(ckpt['classifier_state_dict'])

# v1
v1 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
v1.load_state_dict(torch.load(cfg.model_dir+'/btc_m15_model.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)

# v2 (fixed)
v2 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
v2r = torch.load(cfg.model_dir+'/btc_m15_v2.pt',map_location='cpu',weights_only=False)['model_state_dict']
bs = {'conv1':0,'conv2':4,'conv3':8}
remapped = {}
for ok,val in v2r.items():
    pf=ok.split('.')[0]
    if pf in bs:
        rest=ok.split('.',1)[1];si=int(rest.split('.')[0])
        param=rest.split('.',1)[1];fi=bs[pf]+si
        nk='cnn.{}.{}'.format(fi,param)
    elif ok.startswith('entry_head.'): nk=ok.replace('entry_head.','entry_conf.',1)
    else: nk=ok
    remapped[nk]=val
v2.load_state_dict(remapped,strict=False)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2,11,18,19,21,22,23}

def run_sweep(model, use_bias, start_dt, end_dt, be_trigger):
    mt5.initialize()
    h1r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_H1, start_dt, end_dt)
    m15r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15, start_dt, end_dt)
    mt5.shutdown()
    h1 = pd.DataFrame(h1r).rename(columns={'time':'ts','tick_volume':'volume'})
    m15 = pd.DataFrame(m15r).rename(columns={'time':'ts','tick_volume':'volume'})
    h1['ts'] = pd.to_datetime(h1['ts'], unit='s', utc=True)
    m15['ts'] = pd.to_datetime(m15['ts'], unit='s', utc=True)
    h1 = h1.sort_values('ts').reset_index(drop=True)
    m15 = m15.sort_values('ts').reset_index(drop=True)
    if len(h1) < cfg.seq_len_h1+10: return None
    h1f = engine.compute(h1); m15f = engine.compute(m15)

    m15c = np.zeros(len(m15), dtype=np.float32)
    m15b = np.zeros(len(m15), dtype=np.float32)
    for i in range(cfg.seq_len_m15, len(m15)):
        seq = engine.compute_sequence(m15f, i, cfg.seq_len_m15)
        with torch.no_grad():
            o = model(torch.from_numpy(seq).unsqueeze(0))
        m15c[i] = float(o['entry_confidence'].squeeze().numpy())
        m15b[i] = float(o['direction_bias'].squeeze().numpy())

    rd = RuleBasedRegimeDetector()
    for i in range(cfg.seq_len_h1):
        rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

    listening=False; h1_sig=0; bl=0; pos=0; tm=None; trades=[]; last_h1=-1

    for mi in range(cfg.seq_len_h1*4, len(m15)):
        ts=m15['ts'].iloc[mi];price=float(m15['close'].iloc[mi])
        mc=float(m15c[mi]);bias=float(m15b[mi])
        hi=int((h1['ts']<=ts).sum()-1)
        if hi>=cfg.seq_len_h1 and hi!=last_h1:
            last_h1=hi
            seq=engine.compute_sequence(h1f,hi,cfg.seq_len_h1)
            t=torch.from_numpy(seq).unsqueeze(0)
            for j in range(max(0,hi-13),hi+1):
                rd.update(float(h1['high'].iloc[j]),float(h1['low'].iloc[j]),float(h1['close'].iloc[j]))
            rr=classify_regime(encoder,classifier,t,rd,cfg.min_regime_confidence,temperature=4.0)
            gd=gate.evaluate(rr['regime'],rr['confidence'],float(rr.get('atr_percentile',0.5)),bb_position=float(h1f[hi,4]))
            if gd.entry_signal:
                hc=h1['close'].values[:hi+1]
                if len(hc)>=23:
                    e22=pd.Series(hc).ewm(span=22,adjust=False).mean().values
                    slp=(e22[-1]-e22[-2])/max(abs(float(e22[-2])),1e-12)
                    if (gd.direction==1 and slp>0) or (gd.direction==-1 and slp<0):
                        h1_sig=gd.direction;listening=True;bl=0
                    else: listening=False
                else: listening=False
            else: listening=False

        if pos!=0 and tm is not None and tm.state is not None:
            s=tm.state;hip=float(m15['high'].iloc[mi]);lop=float(m15['low'].iloc[mi])
            epx=None;reason=''
            if tm.check_sl_hit(lop,hip): epx=tm.exit_price_at_sl();reason='sl_hit'
            elif tm.check_tp_hit(lop,hip): epx=tm.exit_price_at_tp();reason='tp_hit'
            else:
                act=tm.update(price,hip,lop,h1f[hi,6]*price)
                if act.action_type==TradeActionType.CLOSE: epx=price;reason=act.reason
            if epx:
                d=1 if pos==1 else -1
                if d==1: pnl_d=(epx-s.entry_price)*s.lots
                else: pnl_d=(s.entry_price-epx)*s.lots
                pnl_r=pnl_d/max(s.entry_atr*s.lots*cfg.initial_sl,1e-9)
                trades.append({'pnl_d':round(pnl_d,2),'pnl_r':round(float(pnl_r),4),'reason':reason})
                pos=0;tm=None
                continue

        if not listening: continue
        bl+=1
        if bl>cfg.max_listen_bars: listening=False;continue
        if ts.hour in BLOCKED: continue

        if use_bias:
            ok = mc >= cfg.min_entry_confidence and ((h1_sig==1 and bias>0) or (h1_sig==-1 and bias<0))
        else:
            ok = mc >= 0.5

        if ok:
            h1_atr=float(h1f[hi,6]*price)
            tm=TradeManager(initial_sl=cfg.initial_sl,hard_tp=cfg.hard_tp,
                breakeven_trigger=be_trigger, trail_trigger=cfg.trail_trigger,
                trail_dist=cfg.trail_dist,trail_dist_s=cfg.trail_dist_s,
                regime_tighten=cfg.regime_tighten,
                max_hold=cfg.max_hold_bars,mae_guard_retrace=cfg.mae_guard_retrace)
            lots=TradeManager.compute_position_size(10000.0,h1_atr,price,cfg.risk_pct,cfg.initial_sl)
            tm.enter(h1_sig,price,h1_atr,lots,regime=rr['regime'])
            pos=h1_sig;listening=False

    if not trades: return None
    n=len(trades);wins=[t for t in trades if t['pnl_d']>0];losses=[t for t in trades if t['pnl_d']<0]
    wr=len(wins)/n*100 if n else 0
    tg=sum(t['pnl_r'] for t in wins);tl=abs(sum(t['pnl_r'] for t in losses))
    pf=tg/max(tl,0.001);pnl=sum(t['pnl_d'] for t in trades)
    tp=sum(1 for t in trades if 'tp_hit' in t['reason'])
    be=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']>-0.5)
    sl=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']<-0.5)
    return {'n':n,'wr':wr,'pf':pf,'pnl':pnl,'avg_r':np.mean([t['pnl_r'] for t in trades]),
            'tp':tp,'be':be,'sl':sl}

# Run sweep
be_values = [0.30, 0.50, 0.60, 0.75, 0.90, 1.10, 1.50]
sd = datetime(2026,4,25); ed = datetime(2026,5,25,12)

print('BREAKEVEN SWEEP — 1 Month (Apr 25 - May 25)')
print()
print('{:>10s} | {:>6s} {:>6s} {:>6s} {:>10s} {:>6s} {:>5s} {:>5s} {:>5s}'.format(
    'BE trig','Trades','WR%','PF','PnL','AvgR','TP','BE','SL'))
print('-'*70)

for be in be_values:
    for model, use_bias, lbl in [(v1,True,'v1'),(v2,False,'v2')]:
        r = run_sweep(model, use_bias, sd, ed, be)
        if r:
            print('{:>10s} {:>3s} {:>6d} {:>5.1f} {:>6.2f} ${:>+9.1f} {:>+6.3f} {:>4d} {:>4d} {:>4d}'.format(
                '{:.2f}'.format(be), lbl, r['n'], r['wr'], r['pf'], r['pnl'], r['avg_r'], r['tp'], r['be'], r['sl']))
    print()

# Show best for each model
print()
print('=== BEST BREAKEVEN PER MODEL ===')
best_v1 = None; best_v2 = None
for be in be_values:
    for model, use_bias, lbl in [(v1,True,'v1'),(v2,False,'v2')]:
        r = run_sweep(model, use_bias, sd, ed, be)
        if r:
            if lbl=='v1' and (best_v1 is None or r['pnl']>best_v1['pnl']): best_v1={'be':be,'r':r}
            if lbl=='v2' and (best_v2 is None or r['pnl']>best_v2['pnl']): best_v2={'be':be,'r':r}
if best_v1:
    r=best_v1['r'];print('v1 best: BE={:.2f}  Trades={}  WR={:.1f}%  PF={:.2f}  PnL=${:+.1f}  TP={}  BE={}  SL={}'.format(
        best_v1['be'],r['n'],r['wr'],r['pf'],r['pnl'],r['tp'],r['be'],r['sl']))
if best_v2:
    r=best_v2['r'];print('v2 best: BE={:.2f}  Trades={}  WR={:.1f}%  PF={:.2f}  PnL=${:+.1f}  TP={}  BE={}  SL={}'.format(
        best_v2['be'],r['n'],r['wr'],r['pf'],r['pnl'],r['tp'],r['be'],r['sl']))
