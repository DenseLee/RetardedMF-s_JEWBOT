"""Compare v1 vs v2 M15 model on backtester YTD."""
import sys, os, numpy as np, pandas as pd, torch
from datetime import datetime
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_gru_m15 import CNNGRUM15
from backtest.data_manager import BacktestDataManager, N_FEATURES

cfg = BTCConfig()
engine = BTCFeatureEngine()

# Load dataset once
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-05-25", use_cache=True)
print(f"Data: {ds.n_h1} H1, {ds.n_m15} M15, {ds.n_m1} M1")

# Load v1 model
v1 = CNNGRUM15(n_features=N_FEATURES, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
v1.load_state_dict(torch.load(cfg.model_dir+'/btc_m15_model.pt', map_location='cpu', weights_only=False)['model_state_dict'], strict=False)

# Load v2 model
v2 = CNNGRUM15(n_features=N_FEATURES, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).eval()
ckpt_v2 = torch.load(cfg.model_dir+'/btc_m15_v2.pt', map_location='cpu', weights_only=False)
state_dict = ckpt_v2['model_state_dict']
# Remap v2 keys
if "conv1.0.weight" in state_dict:
    block_starts = {"conv1": 0, "conv2": 4, "conv3": 8}
    remapped = {}
    for old_key, val in state_dict.items():
        prefix = old_key.split(".")[0]
        if prefix in block_starts:
            rest = old_key.split(".", 1)[1]
            sub_idx = int(rest.split(".")[0])
            param = rest.split(".", 1)[1]
            flat_idx = block_starts[prefix] + sub_idx
            remapped[f"cnn.{flat_idx}.{param}"] = val
        elif old_key.startswith("entry_head."):
            remapped[old_key.replace("entry_head.", "entry_conf.", 1)] = val
        else:
            remapped[old_key] = val
    state_dict = remapped
v2.load_state_dict(state_dict, strict=False)

# Compute M15 confidence for both models
n = ds.n_m15
seq_len = cfg.seq_len_m15
v1_conf = np.zeros(n, dtype=np.float32)
v1_bias = np.zeros(n, dtype=np.float32)
v2_conf = np.zeros(n, dtype=np.float32)

for i in range(seq_len - 1, n):
    seq = engine.compute_sequence(ds.m15_features, i, seq_len)
    t = torch.from_numpy(seq).unsqueeze(0)
    with torch.no_grad():
        o1 = v1(t)
        o2 = v2(t)
    v1_conf[i] = float(o1['entry_confidence'].squeeze().numpy())
    v1_bias[i] = float(o1['direction_bias'].squeeze().numpy())
    v2_conf[i] = float(o2['entry_confidence'].squeeze().numpy())

del v1, v2

# --- Run backtest for both ---
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2, 11, 18, 19, 21, 22, 23}

h1_df = ds.h1_df; m15_df = ds.m15_df
h1_feats = ds.h1_features
h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile
h1_rule_regime = ds.h1_rule_regime

def run_backtest(m15_conf_arr, m15_bias_arr, label, v1_mode=False):
    """Run drip-feed backtest. v1_mode adds direction_bias check and uses 0.6 threshold."""
    min_h1 = cfg.seq_len_h1
    warm_h1_ts = h1_df['timestamp'].iloc[min_h1 - 1]
    warm_m15 = int((m15_df['timestamp'] > warm_h1_ts).sum())
    warm_m15 = max(warm_m15, cfg.seq_len_m15)

    listening = False; h1_signal = 0; bl = 0
    position = 0; last_h1_idx = -1; last_regime = None
    max_listen = cfg.max_listen_bars
    tm = TradeManager(initial_sl=cfg.initial_sl, hard_tp=cfg.hard_tp,
        breakeven_trigger=cfg.breakeven_trigger, trail_trigger=cfg.trail_trigger,
        trail_dist=cfg.trail_dist, trail_dist_s=cfg.trail_dist_s,
        regime_tighten=cfg.regime_tighten, max_hold=cfg.max_hold_bars,
        mae_guard_retrace=cfg.mae_guard_retrace)
    balance = 10000.0
    trades = []

    for m15_i in range(warm_m15, len(m15_df)):
        ts = m15_df['timestamp'].iloc[m15_i]
        price = float(m15_df['close'].iloc[m15_i])
        h1_i = int((h1_df['timestamp'] <= ts).sum() - 1)

        if h1_i != last_h1_idx and h1_i >= 0:
            last_h1_idx = h1_i
            ri = h1_regime_list[h1_i]
            current_regime = ri['regime']
            atr_pct = float(h1_atr_pctl[h1_i])
            current_atr = float(h1_feats[h1_i, 6]) * float(h1_df['close'].iloc[h1_i])
            bb_pos = float(h1_feats[h1_i, 4])

            # Rule-wins-conflict
            if h1_rule_regime is not None and h1_rule_regime[h1_i] is not None:
                trending = {'TREND_UP', 'TREND_DOWN'}
                rule_r = h1_rule_regime[h1_i]
                if (current_regime in trending and rule_r in trending
                        and current_regime != rule_r):
                    current_regime = rule_r

            gd = gate.evaluate(current_regime, ri['confidence'], atr_pct, bb_position=bb_pos)

            if gd.entry_signal:
                h1_closes = h1_df['close'].values[:h1_i + 1]
                if len(h1_closes) >= 23:
                    ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                    slope = (ema22[-1] - ema22[-2]) / max(abs(float(ema22[-2])), 1e-12)
                    with_trend = ((gd.direction == 1 and slope > 0) or
                                  (gd.direction == -1 and slope < 0))
                    if not with_trend:
                        listening = False; h1_signal = 0; continue
                h1_signal = gd.direction; listening = True; bl = 0
                last_regime = current_regime
            else:
                h1_signal = 0; listening = False

        # Manage position
        if position != 0 and tm.state is not None:
            hip = float(m15_df['high'].iloc[m15_i]); lop = float(m15_df['low'].iloc[m15_i])
            epx = None; reason = ''
            if tm.check_sl_hit(lop, hip): epx = tm.exit_price_at_sl(); reason = 'sl_hit'
            elif tm.check_tp_hit(lop, hip): epx = tm.exit_price_at_tp(); reason = 'tp_hit'
            else:
                act = tm.update(price, hip, lop, h1_feats[h1_i, 6] * price)
                if act.action_type == TradeActionType.CLOSE: epx = price; reason = act.reason
            if epx:
                d = 1 if position == 1 else -1
                if d == 1: pnl_d = (epx - tm.state.entry_price) * tm.state.lots
                else: pnl_d = (tm.state.entry_price - epx) * tm.state.lots
                pnl_r = pnl_d / max(tm.state.entry_atr * tm.state.lots * cfg.initial_sl, 1e-9)
                trades.append({'pnl_d': round(pnl_d, 2), 'pnl_r': round(float(pnl_r), 4), 'reason': reason})
                balance += pnl_d; position = 0; tm.state = None
                continue

        if not listening: continue
        bl += 1
        if bl > max_listen: listening = False; continue
        if ts.hour in BLOCKED: continue
        if position != 0: continue

        conf = float(m15_conf_arr[m15_i])
        bias = float(m15_bias_arr[m15_i]) if m15_bias_arr is not None else 0.0

        if v1_mode:
            ok = conf >= cfg.min_entry_confidence and ((h1_signal == 1 and bias > 0) or (h1_signal == -1 and bias < 0))
        else:
            ok = conf >= 0.5

        if ok:
            h1_atr = float(h1_feats[h1_i, 6] * price)
            lots = TradeManager.compute_position_size(balance, h1_atr, price, cfg.risk_pct, cfg.initial_sl)
            tm.enter(h1_signal, price, h1_atr, lots, regime=last_regime or "")
            position = h1_signal; listening = False

    if not trades: return None
    n_t = len(trades)
    wins = [t for t in trades if t['pnl_d'] > 0]
    losses = [t for t in trades if t['pnl_d'] < 0]
    wr = len(wins) / n_t * 100 if n_t else 0
    rs = np.array([t['pnl_r'] for t in trades])
    tg = sum(r for r in rs if r > 0); tl = abs(sum(r for r in rs if r <= 0))
    pf = tg / max(tl, 0.001); avg_r = np.mean(rs); total_r = rs.sum()
    total_pnl = sum(t['pnl_d'] for t in trades)
    sharpe = avg_r / max(np.std(rs), 0.001) if n_t > 1 else 0
    elo = 1500 + (wr - 50) * 10 + min((pf - 1) * 300, 500) + min(sharpe * 100, 300) + min(total_r * 10, 500)
    elo = max(0, min(3000, elo))
    tp_hits = sum(1 for t in trades if 'tp_hit' in t['reason'])
    sl_hits = sum(1 for t in trades if t['reason'] == 'sl_hit' and t['pnl_r'] < -0.5)
    be_hits = sum(1 for t in trades if t['reason'] == 'sl_hit' and t['pnl_r'] > -0.5)
    return {'label': label, 'n': n_t, 'wr': wr, 'pf': pf, 'pnl': total_pnl,
            'avg_r': avg_r, 'total_r': total_r, 'sharpe': sharpe, 'elo': int(elo),
            'tp': tp_hits, 'sl': sl_hits, 'be': be_hits, 'trades': trades}

print("\nRunning V2 (conf>=0.5, no dir_bias)...")
v2_result = run_backtest(v2_conf, None, "V2", v1_mode=False)

print("Running V1 (conf>=0.6, with dir_bias)...")
v1_result = run_backtest(v1_conf, v1_bias, "V1", v1_mode=True)

print()
print("=" * 70)
print("V1 vs V2 M15 Model — YTD Backtest")
print("=" * 70)
print(f"{'Metric':25s} {'V2 (conf>=0.5)':>15s} {'V1 (conf>=0.6+bias)':>20s}")
print("-" * 65)
for key, label, fmt in [
    ('n', 'Trades', 'd'), ('wr', 'Win Rate %', '.1f'), ('pf', 'Profit Factor', '.2f'),
    ('pnl', 'PnL $', '.1f'), ('avg_r', 'Avg R', '.3f'), ('total_r', 'Total R', '.1f'),
    ('sharpe', 'Sharpe', '.2f'), ('elo', 'ELO', '.0f'),
    ('tp', 'TP hits', 'd'), ('sl', 'Full SL', 'd'), ('be', 'BE hits', 'd')]:
    v2v = v2_result[key]; v1v = v1_result[key]
    if fmt == 'd':
        print(f"{label:25s} {int(v2v):>15d} {int(v1v):>20d}")
    else:
        print(f"{label:25s} {v2v:>15{fmt}} {v1v:>20{fmt}}")

# Monthly breakdown
print()
print(f"{'Month':10s} {'V2 Trades':>10s} {'V2 PnL':>10s} {'V1 Trades':>10s} {'V1 PnL':>10s}")
print("-" * 55)
df_v2 = pd.DataFrame(v2_result['trades'])
# Timestamps embedded in the trades dict aren't there, skip monthly for now

# M15 model stats comparison
print()
print("M15 Confidence Stats:")
print(f"  V2: mean={v2_conf[v2_conf>0].mean():.4f} median={np.median(v2_conf[v2_conf>0]):.4f}  >0.5: {(v2_conf>0.5).sum()} ({((v2_conf>0.5).sum()/len(v2_conf[v2_conf>0])*100):.1f}%)")
print(f"  V1: mean={v1_conf[v1_conf>0].mean():.4f} median={np.median(v1_conf[v1_conf>0]):.4f}  >0.6: {(v1_conf>0.6).sum()} ({((v1_conf>0.6).sum()/len(v1_conf[v1_conf>0])*100):.1f}%)")

# V1 direction_bias stats
pos_bias = (v1_bias > 0).sum()
neg_bias = (v1_bias < 0).sum()
print(f"  V1 bias: positive={pos_bias} negative={neg_bias}")

# Combined: conf>=0.6 AND bias matches
v1_long_ok = (v1_conf >= 0.6) & (v1_bias > 0)
v1_short_ok = (v1_conf >= 0.6) & (v1_bias < 0)
print(f"  V1 passes (conf>=0.6 + bias_dir): long={v1_long_ok.sum()} short={v1_short_ok.sum()}")
