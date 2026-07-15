"""Trace backtester entry pipeline to find the bottleneck."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from backtest.data_manager import BacktestDataManager
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager

cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-05-25", use_cache=True)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)
BLOCKED = {2, 11, 18, 19, 21, 22, 23}

h1_df = ds.h1_df; m15_df = ds.m15_df
h1_feats = ds.h1_features
h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile
h1_rule_regime = ds.h1_rule_regime

min_h1 = cfg.seq_len_h1  # 96
warm_h1_ts = h1_df['timestamp'].iloc[min_h1 - 1]
warm_m15 = int((m15_df['timestamp'] > warm_h1_ts).sum())
warm_m15 = max(warm_m15, cfg.seq_len_m15)
max_listen = cfg.max_listen_bars

# Counters
h1_bars_evaluated = 0
h1_gate_signals = 0
h1_gate_blocked = 0
listening_windows_opened = 0
m15_confirm_checks = 0
m15_confirmed = 0
m15_blocked_hour = 0
m15_blocked_position = 0
m15_blocked_expiry = 0
m15_blocked_turning = 0
entries = 0

listening = False; h1_signal = 0; bl = 0
position = 0; last_h1_idx = -1
tm = TradeManager(phase1_sl=cfg.initial_sl, hard_tp=cfg.hard_tp,
    phase1_bars=cfg.phase1_bars, phase2_mfe_min=cfg.phase2_mfe_min,
    phase3_trail=cfg.phase3_trail, phase4_trail=cfg.phase4_trail,
    phase4_start=cfg.phase4_start, max_hold=cfg.max_hold_bars)

for m15_i in range(warm_m15, len(m15_df)):
    ts = m15_df['timestamp'].iloc[m15_i]
    price = float(m15_df['close'].iloc[m15_i])
    h1_i = int((h1_df['timestamp'] <= ts).sum() - 1)

    if h1_i != last_h1_idx and h1_i >= 0:
        last_h1_idx = h1_i
        h1_bars_evaluated += 1
        ri = h1_regime_list[h1_i]
        current_regime = ri['regime']
        atr_pct = float(h1_atr_pctl[h1_i])
        bb_pos = float(h1_feats[h1_i, 4])

        if h1_rule_regime is not None and h1_rule_regime[h1_i] is not None:
            trending = {'TREND_UP', 'TREND_DOWN'}
            rule_r = h1_rule_regime[h1_i]
            if (current_regime in trending and rule_r in trending
                    and current_regime != rule_r):
                current_regime = rule_r

        gd = gate.evaluate(current_regime, ri['confidence'], atr_pct, bb_position=bb_pos)

        if gd.entry_signal:
            h1_gate_signals += 1
            h1_signal = gd.direction
            listening = True
            bl = 0
            listening_windows_opened += 1
        else:
            h1_gate_blocked += 1
            h1_signal = 0
            listening = False

    # Position management (simplified)
    if position != 0 and tm.state is not None:
        # Skip detailed exit logic - just track that position blocks
        pass

    if not listening: continue
    bl += 1
    if bl > max_listen: listening = False; m15_blocked_expiry += 1; continue
    if ts.hour in BLOCKED: m15_blocked_hour += 1; continue
    if position != 0: m15_blocked_position += 1; continue

    m15_confirm_checks += 1

    # Turning check (same as backtester)
    m15_closes = m15_df['close'].values[:m15_i + 1]
    if len(m15_closes) >= 3:
        if h1_signal == 1:
            turning = m15_closes[-1] > m15_closes[-2]
        elif h1_signal == -1:
            turning = m15_closes[-1] < m15_closes[-2]
        else:
            turning = False
    else:
        turning = False

    if turning:
        m15_confirmed += 1
        entries += 1
        h1_atr = float(h1_feats[h1_i, 6] * price)
        lots = TradeManager.compute_position_size(10000.0, h1_atr, price, cfg.risk_pct, cfg.initial_sl)
        tm.enter(h1_signal, price, h1_atr, lots)
        position = h1_signal
        listening = False
    else:
        m15_blocked_turning += 1

print(f"H1 bars evaluated: {h1_bars_evaluated}")
print(f"H1 gate signals: {h1_gate_signals}")
print(f"H1 gate blocked: {h1_gate_blocked}")
print(f"Listening windows: {listening_windows_opened}")
print(f"M15 confirm checks: {m15_confirm_checks}")
print(f"  Confirmed (turning): {m15_confirmed}")
print(f"  Blocked by turning: {m15_blocked_turning}")
print(f"  Blocked by hour:    {m15_blocked_hour}")
print(f"  Blocked by position:{m15_blocked_position}")
print(f"  Blocked by expiry:  {m15_blocked_expiry}")
print(f"Entries: {entries}")
print(f"")
print(f"Pipeline efficiency:")
print(f"  H1 signals → listening: {listening_windows_opened}/{h1_gate_signals} ({listening_windows_opened/h1_gate_signals*100:.0f}%)")
print(f"  Listening → M15 checks: {m15_confirm_checks}/{listening_windows_opened} ({m15_confirm_checks/max(listening_windows_opened,1)*100:.0f}%)")
print(f"  M15 checks → confirmed: {m15_confirmed}/{m15_confirm_checks} ({m15_confirmed/max(m15_confirm_checks,1)*100:.0f}%)")
print(f"  H1 signals → entries:   {entries}/{h1_gate_signals} ({entries/h1_gate_signals*100:.1f}%)")
