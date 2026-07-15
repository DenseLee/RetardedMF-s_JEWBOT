"""Minimal backtest trace for one week to find entry bottleneck."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from backtest.data_manager import BacktestDataManager
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType

cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-05", "2026-01-11", use_cache=False, force_refresh=True)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = set()

h1_df = ds.h1_df; m15_df = ds.m15_df
h1_feats = ds.h1_features; h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile; h1_rule_regime = ds.h1_rule_regime

min_h1 = cfg.seq_len_h1
warm_h1_ts = h1_df['timestamp'].iloc[min_h1 - 1]
warm_m15 = int((m15_df['timestamp'] >= warm_h1_ts).sum())  # Fixed: >= instead of >
warm_m15 = max(warm_m15, cfg.seq_len_m15)

listening = False; h1_signal = 0; bl = 0
position = 0; last_h1_idx = -1
tm = TradeManager(phase1_sl=cfg.initial_sl, hard_tp=cfg.hard_tp,
    phase1_bars=cfg.phase1_bars, phase2_mfe_min=cfg.phase2_mfe_min,
    phase3_trail=cfg.phase3_trail, phase4_trail=cfg.phase4_trail,
    phase4_start=cfg.phase4_start, max_hold=cfg.max_hold_bars)
trades = []

print(f"Trace: {len(m15_df)-warm_m15} M15 bars from {m15_df['timestamp'].iloc[warm_m15]}")
print(f"{'Time':20s} {'H1_eval':8s} {'Signal':8s} {'Listen':5s} {'Pos':5s} {'Turn':6s} {'Action':20s} {'PnL':>8s}")
print("-" * 100)

for m15_i in range(warm_m15, len(m15_df)):
    ts = m15_df['timestamp'].iloc[m15_i]
    price = float(m15_df['close'].iloc[m15_i])
    h1_i = int((h1_df['timestamp'] <= ts).sum() - 1)

    h1_eval = ""
    action = ""
    pnl_str = ""

    if h1_i != last_h1_idx and h1_i >= 0:
        last_h1_idx = h1_i
        ri = h1_regime_list[h1_i]
        current_regime = ri['regime']
        atr_pct = float(h1_atr_pctl[h1_i])
        bb_pos = float(h1_feats[h1_i, 4])

        if h1_rule_regime is not None and h1_rule_regime[h1_i] is not None:
            trending = {'TREND_UP', 'TREND_DOWN'}
            rule_r = h1_rule_regime[h1_i]
            if (current_regime in trending and rule_r in trending and current_regime != rule_r):
                current_regime = rule_r

        gd = gate.evaluate(current_regime, ri['confidence'], atr_pct, bb_position=bb_pos)
        h1_eval = f"H1:{current_regime[:4]}"

        if gd.entry_signal:
            h1_signal = gd.direction; listening = True; bl = 0
        else:
            h1_signal = 0; listening = False

    # Position mgmt
    exit_info = None
    if position != 0 and tm.state is not None:
        hip = float(m15_df['high'].iloc[m15_i]); lop = float(m15_df['low'].iloc[m15_i])
        if tm.check_sl_hit(lop, hip): exit_info = ('sl', tm.exit_price_at_sl())
        elif tm.check_tp_hit(lop, hip): exit_info = ('tp', tm.exit_price_at_tp())
        else:
            act = tm.update(price, hip, lop, h1_feats[h1_i,6]*price)
            if act.action_type == TradeActionType.CLOSE: exit_info = (act.reason[:20], price)

        if exit_info:
            s = tm.state
            d = 1 if position == 1 else -1
            if d == 1: pnl_d = (exit_info[1] - s.entry_price) * s.lots
            else: pnl_d = (s.entry_price - exit_info[1]) * s.lots
            pnl_r = pnl_d / max(s.entry_atr * s.lots * cfg.initial_sl, 1e-9)
            pnl_str = f"${pnl_d:+.1f}"
            action = f"EXIT:{exit_info[0]}"
            position = 0; tm.state = None

    signal_str = f"{'L' if h1_signal==1 else 'S' if h1_signal==-1 else '-'}{gd.direction if 'gd' in dir() else ''}"
    listen_str = "Y" if listening else "-"
    pos_str = f"{'L' if position==1 else 'S' if position==-1 else '-'}"

    # M15 confirm
    if listening and position == 0:
        bl += 1
        closes = m15_df['close'].values[:m15_i+1]
        turning = False
        if len(closes) >= 3:
            if h1_signal == 1: turning = closes[-1] > closes[-2]
            elif h1_signal == -1: turning = closes[-1] < closes[-2]

        turn_str = "T" if turning else "-"

        if turning:
            h1_atr = float(h1_feats[h1_i,6]*price)
            lots = TradeManager.compute_position_size(10000.0, h1_atr, price, cfg.risk_pct, cfg.initial_sl)
            tm.enter(h1_signal, price, h1_atr, lots)
            position = h1_signal; listening = False
            action = f"ENTER {h1_signal:+d}"
        elif bl > cfg.max_listen_bars:
            listening = False; action = "EXPIRED"
    else:
        turn_str = "-"

    if h1_eval or action or exit_info:
        print(f"{str(ts)[:19]:20s} {h1_eval:8s} {signal_str:8s} {listen_str:5s} {pos_str:5s} {turn_str:6s} {action:20s} {pnl_str:>8s}")

print(f"\nTrades: {len(trades)}")
