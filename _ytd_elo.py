"""YTD backtest with ELO rating + per-regime breakdown."""
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

# M15 v1 (best model)
m15 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
m15.load_state_dict(torch.load(cfg.model_dir+'/btc_m15_model.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2,11,18,19,21,22,23}

print('Fetching YTD data...')
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
print('H1: {}  M15: {}'.format(len(h1), len(m15d)))

h1f = engine.compute(h1); m15f = engine.compute(m15d)

m15c = np.zeros(len(m15d), dtype=np.float32)
m15b = np.zeros(len(m15d), dtype=np.float32)
print('Precomputing M15 confs...')
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
regime_at_entry = ''; entry_c=0; entry_b=0; entry_ts=None; entry_px=0

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
            # Track MFE from TradeManager state
            mfe_r = s.mfe_r if hasattr(s,'mfe_r') else 0
            trades.append({
                'entry_ts':str(entry_ts)[:19],'dir':'L' if pos==1 else 'S',
                'entry_px':entry_px,'exit_px':round(epx,1),
                'pnl_d':round(pnl_d,2),'pnl_r':round(float(pnl_r),4),
                'mfe_r':round(float(mfe_r),4),'reason':reason,
                'regime':regime_at_entry,'conf':entry_c,'bias':entry_b,
                'bars':s.bars_held,
            })
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
        entry_c=round(float(mc),4);entry_b=round(float(bias),4)
        entry_ts=ts;entry_px=round(price,1);regime_at_entry=rr['regime']

# --- ELO Rating ---
if not trades:
    print('No trades generated')
    exit()

n=len(trades); wins=[t for t in trades if t['pnl_d']>0]; losses=[t for t in trades if t['pnl_d']<0]
wr=len(wins)/n*100
rs=np.array([t['pnl_r'] for t in trades])
tg=sum(r for r in rs if r>0);tl=abs(sum(r for r in rs if r<=0))
pf=tg/max(tl,0.001);avg_r=np.mean(rs);std_r=np.std(rs)
sharpe=avg_r/max(std_r,0.001) if n>1 else 0
total_pnl=sum(t['pnl_d'] for t in trades)

# ELO: composite score from 0-3000 (chess-like scale)
# Base 1500, adjusted by: WR(+), PF(+), consistency(+), avg_R(+), total_R(+)
total_r = rs.sum()
elo_base = 1500
elo_wr = (wr - 50) * 10  # -500 to +500
elo_pf = min((pf - 1) * 300, 500)  # 0 to 500
elo_sharpe = min(sharpe * 100, 300)  # 0 to 300
elo_r = min(total_r * 10, 500)  # -500 to 500
elo = elo_base + elo_wr + elo_pf + elo_sharpe + elo_r
elo = max(0, min(3000, elo))

# Per-regime breakdown
regimes = {}
for t in trades:
    r = t['regime']
    if r not in regimes: regimes[r] = []
    regimes[r].append(t)

print()
print('='*65)
print('YTD BACKTEST — {} to {}'.format(
    str(h1['ts'].iloc[cfg.seq_len_h1])[:10],
    str(h1['ts'].iloc[-1])[:10]))
print('='*65)
print('Config: SL={:.1f}xATR  TP={:.1f}xATR  BE={:.2f}R  Trail={:.1f}R  MaxHold={}h'.format(
    cfg.initial_sl, cfg.hard_tp, cfg.breakeven_trigger, cfg.trail_trigger, cfg.max_hold_bars))
print('Model: H1 encoder + M15 v1 (direction_bias)')
print()
print('--- OVERALL ---')
print('Trades: {:>4d}'.format(n))
print('Win Rate: {:>5.1f}%'.format(wr))
print('Profit Factor: {:>6.2f}'.format(pf))
print('Avg R: {:>+8.3f}'.format(avg_r))
print('Std R: {:>8.3f}'.format(std_r))
print('Sharpe: {:>8.2f}'.format(sharpe))
print('Total PnL: ${:>+9.1f}'.format(total_pnl))
print('Total R:   {:>+9.3f}R'.format(total_r))
print()
print('*** ELO RATING: {:.0f} ***'.format(elo))
# Rating tier
if elo >= 1800: tier = 'Expert (profitable + consistent)'
elif elo >= 1600: tier = 'Advanced (profitable but volatile)'
elif elo >= 1400: tier = 'Intermediate (breakeven-ish)'
elif elo >= 1200: tier = 'Novice (losing slightly)'
else: tier = 'Beginner (needs significant improvement)'
print('Tier: {}'.format(tier))
print()

# Per-regime
print('='*65)
print('PER-REGIME BREAKDOWN')
print('='*65)
print('{:15s} {:>6s} {:>7s} {:>7s} {:>10s} {:>8s} {:>6s} {:>6s}'.format(
    'Regime','Trades','WR%','PF','PnL','AvgR','TP','SL'))
print('-'*70)

for rname in ['TREND_UP','TREND_DOWN','RANGE','TRANSITION']:
    rt = regimes.get(rname, [])
    if not rt: continue
    rn=len(rt);rw=sum(1 for t in rt if t['pnl_d']>0)
    rwr=rw/rn*100 if rn else 0
    rrs=[t['pnl_r'] for t in rt]
    rtg=sum(r for r in rrs if r>0);rtl=abs(sum(r for r in rrs if r<=0))
    rpf=rtg/max(rtl,0.001);rpnl=sum(t['pnl_d'] for t in rt)
    ravg=np.mean(rrs)
    rtp=sum(1 for t in rt if 'tp_hit' in t['reason'])
    rsl=sum(1 for t in rt if t['reason']=='sl_hit' and t['pnl_r']<-0.5)
    print('{:15s} {:>4d}  {:>5.1f}% {:>6.2f} ${:>+9.1f} {:>+7.3f} {:>4d}  {:>4d}'.format(
        rname,rn,rwr,rpf,rpnl,ravg,rtp,rsl))

# Per-regime ELO
print()
print('--- Per-Regime ELO ---')
for rname in ['TREND_UP','TREND_DOWN','RANGE','TRANSITION']:
    rt = regimes.get(rname, [])
    if not rt or len(rt) < 3: continue
    rn=len(rt);rw=sum(1 for t in rt if t['pnl_d']>0)
    rwr=rw/rn*100;rrs=[t['pnl_r'] for t in rt]
    rtg=sum(r for r in rrs if r>0);rtl=abs(sum(r for r in rrs if r<=0))
    rpf=rtg/max(rtl,0.001);ravg=np.mean(rrs)
    r_total_r = sum(rrs)
    r_elo = 1500 + (rwr-50)*10 + min((rpf-1)*300,500) + min((ravg/max(np.std(rrs),0.001))*100,300) + min(r_total_r*10,500)
    r_elo = max(0, min(3000, r_elo))
    r_tier = 'A' if r_elo>=1600 else 'B' if r_elo>=1400 else 'C' if r_elo>=1200 else 'D'
    print('  {:15s}: ELO={:.0f} (tier {})  {} trades'.format(rname, r_elo, r_tier, rn))

# Monthly breakdown
print()
print('--- Monthly ---')
df = pd.DataFrame(trades)
df['month'] = pd.to_datetime(df['entry_ts']).dt.strftime('%Y-%m')
for month, grp in df.groupby('month'):
    mn=len(grp);mw=sum(1 for _,t in grp.iterrows() if t['pnl_d']>0)
    mpnl=grp['pnl_d'].sum();mravg=grp['pnl_r'].mean()
    print('  {}: {:>3d} trades  WR={:.0f}%  PnL=${:+.1f}  AvgR={:+.3f}'.format(
        month,mn,mw/mn*100,mpnl,mravg))
