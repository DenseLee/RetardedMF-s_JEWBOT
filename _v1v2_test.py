"""V1 vs V2 M15 model backtest — YTD and 1-month."""
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

# M15 v1
m15_v1 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
v1_ckpt = torch.load(cfg.model_dir + '/btc_m15_model.pt', map_location='cpu', weights_only=False)
m15_v1.load_state_dict(v1_ckpt['model_state_dict'], strict=True)
print('v1: all keys matched')

# M15 v2 (fixed)
m15_v2 = CNNGRUM15(n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
v2_ckpt = torch.load(cfg.model_dir + '/btc_m15_v2.pt', map_location='cpu', weights_only=False)
sd = v2_ckpt['model_state_dict']
bs = {'conv1': 0, 'conv2': 4, 'conv3': 8}
remapped = {}
for ok, val in sd.items():
    pf = ok.split('.')[0]
    if pf in bs:
        rest = ok.split('.', 1)[1]; si = int(rest.split('.')[0])
        param = rest.split('.', 1)[1]; fi = bs[pf] + si
        nk = 'cnn.{}.{}'.format(fi, param)
    elif ok.startswith('entry_head.'):
        nk = ok.replace('entry_head.', 'entry_conf.', 1)
    else:
        nk = ok
    remapped[nk] = val
m15_v2.load_state_dict(remapped, strict=False)
print('v2: remapped and loaded')

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2, 11, 18, 19, 21, 22, 23}

def run_bt(model, is_v2, start_dt, end_dt):
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

    if len(h1) < cfg.seq_len_h1 + 10:
        return None

    h1f = engine.compute(h1)
    m15f = engine.compute(m15)

    m15c = np.zeros(len(m15), dtype=np.float32)
    for i in range(cfg.seq_len_m15, len(m15)):
        seq = engine.compute_sequence(m15f, i, cfg.seq_len_m15)
        with torch.no_grad():
            o = model(torch.from_numpy(seq).unsqueeze(0))
        m15c[i] = float(o['entry_confidence'].squeeze().numpy())

    rd = RuleBasedRegimeDetector()
    for i in range(cfg.seq_len_h1):
        rd.update(float(h1['high'].iloc[i]), float(h1['low'].iloc[i]), float(h1['close'].iloc[i]))

    listening = False; h1_sig = 0; bl = 0
    pos = 0; tm = None; trades = []; last_h1 = -1

    for mi in range(cfg.seq_len_h1 * 4, len(m15)):
        ts = m15['ts'].iloc[mi]; price = float(m15['close'].iloc[mi])
        mc = float(m15c[mi])

        hi = int((h1['ts'] <= ts).sum() - 1)
        if hi >= cfg.seq_len_h1 and hi != last_h1:
            last_h1 = hi
            seq = engine.compute_sequence(h1f, hi, cfg.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0)
            for j in range(max(0, hi-13), hi+1):
                rd.update(float(h1['high'].iloc[j]), float(h1['low'].iloc[j]), float(h1['close'].iloc[j]))
            rr = classify_regime(encoder, classifier, t, rd, cfg.min_regime_confidence, temperature=4.0)
            gd = gate.evaluate(rr['regime'], rr['confidence'],
                               float(rr.get('atr_percentile',0.5)),
                               bb_position=float(h1f[hi,4]))
            if gd.entry_signal:
                hc = h1['close'].values[:hi+1]
                if len(hc) >= 23:
                    e22 = pd.Series(hc).ewm(span=22, adjust=False).mean().values
                    slp = (e22[-1]-e22[-2])/max(abs(float(e22[-2])),1e-12)
                    if (gd.direction==1 and slp>0) or (gd.direction==-1 and slp<0):
                        h1_sig=gd.direction; listening=True; bl=0
                    else:
                        listening=False
                else:
                    listening=False
            else:
                listening=False

        if pos != 0 and tm is not None and tm.state is not None:
            s = tm.state; hi_p = float(m15['high'].iloc[mi]); lo_p = float(m15['low'].iloc[mi])
            epx = None; reason = ''
            if tm.check_sl_hit(lo_p, hi_p):
                epx = tm.exit_price_at_sl(); reason = 'sl_hit'
            elif tm.check_tp_hit(lo_p, hi_p):
                epx = tm.exit_price_at_tp(); reason = 'tp_hit'
            else:
                act = tm.update(price, hi_p, lo_p, h1f[hi,6]*price)
                if act.action_type == TradeActionType.CLOSE:
                    epx = price; reason = act.reason
            if epx:
                d = 1 if pos==1 else -1
                if d==1: pnl_d = (epx-s.entry_price)*s.lots
                else: pnl_d = (s.entry_price-epx)*s.lots
                pnl_r = pnl_d/max(s.entry_atr*s.lots*cfg.initial_sl,1e-9)
                trades.append({'pnl_d':round(pnl_d,2),'pnl_r':round(float(pnl_r),4),
                               'reason':reason,'conf':entry_c,'dir':'L' if pos==1 else 'S',
                               'ts':str(entry_ts)[:19]})
                pos=0; tm=None
                continue

        if not listening: continue
        bl += 1
        if bl > cfg.max_listen_bars: listening=False; continue
        if ts.hour in BLOCKED: continue

        if is_v2:
            ok = mc >= 0.5
        else:
            seq2 = engine.compute_sequence(m15f, mi, cfg.seq_len_m15)
            t2 = torch.from_numpy(seq2).unsqueeze(0)
            with torch.no_grad():
                o2 = model(t2)
            bias = float(o2['direction_bias'].squeeze().numpy())
            ok = mc >= cfg.min_entry_confidence and (
                (h1_sig==1 and bias>0) or (h1_sig==-1 and bias<0))

        if ok:
            h1_atr = float(h1f[hi,6]*price)
            tm = TradeManager(initial_sl=cfg.initial_sl, hard_tp=cfg.hard_tp,
                breakeven_trigger=cfg.breakeven_trigger,
                trail_trigger=cfg.trail_trigger, trail_dist=cfg.trail_dist,
                trail_dist_s=cfg.trail_dist_s, regime_tighten=cfg.regime_tighten,
                max_hold=cfg.max_hold_bars, mae_guard_retrace=cfg.mae_guard_retrace)
            lots = TradeManager.compute_position_size(10000.0, h1_atr, price, cfg.risk_pct, cfg.initial_sl)
            tm.enter(h1_sig, price, h1_atr, lots, regime=rr['regime'])
            pos=h1_sig; listening=False; entry_c=round(float(mc),4); entry_ts=ts

    if not trades: return None
    n=len(trades); wins=[t for t in trades if t['pnl_d']>0]
    losses=[t for t in trades if t['pnl_d']<0]
    wr=len(wins)/n*100 if n else 0
    tg=sum(t['pnl_r'] for t in wins); tl=abs(sum(t['pnl_r'] for t in losses))
    pf=tg/max(tl,0.001)
    tp=sum(1 for t in trades if 'tp_hit' in t['reason'])
    sl=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']<-0.5)
    be=sum(1 for t in trades if t['reason']=='sl_hit' and t['pnl_r']>-0.5)
    return {'n':n,'wr':wr,'pf':pf,'pnl':sum(t['pnl_d'] for t in trades),
            'avg_r':np.mean([t['pnl_r'] for t in trades]),'tp':tp,
            'full_sl':sl,'be':be,'trades':trades}

# Run
for plbl, sd, ed in [
    ('YTD (Jan 1 - now)', datetime(2026,1,1), datetime(2026,5,25,12)),
    ('1 Month (Apr 25 - now)', datetime(2026,4,25), datetime(2026,5,25,12)),
]:
    print()
    print('='*65)
    print(plbl)
    print('='*65)
    for model, is_v2, lbl in [(m15_v1, False, 'v1 (conf + direction_bias)'),
                               (m15_v2, True, 'v2 (conf only)')]:
        r = run_bt(model, is_v2, sd, ed)
        if r:
            print()
            print('  {}:'.format(lbl))
            print('    Trades: {:>4d}  WR: {:>5.1f}%  PF: {:>6.2f}  PnL: ${:>+8.1f}  AvgR: {:>+7.3f}  TP: {:>2d}  FullSL: {:>2d}  BE: {:>2d}'.format(
                r['n'], r['wr'], r['pf'], r['pnl'], r['avg_r'], r['tp'], r['full_sl'], r['be']))
            for t in r['trades'][-3:]:
                print('      {} conf={:.4f} PnL=${:+.1f} {} {}'.format(
                    t['ts'], t['conf'], t['pnl_d'], t['reason'], t['dir']))
        else:
            print('  {}: No trades'.format(lbl))
