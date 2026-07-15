"""Trace why backtester has so few entries. Run Jan only for speed."""
import sys, os, numpy as np, pandas as pd, torch
from datetime import datetime
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from backtest.data_manager import BacktestDataManager
from backtest.backtester_btc import BTCBacktester, BLOCKED_HOURS
from backtest.slippage_model import SlippageConfig
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager

cfg = BTCConfig()
dm = BacktestDataManager(cfg)
ds = dm.prepare("2026-01-01", "2026-01-31", use_cache=False, force_refresh=True)

gate = EntryGate(min_confidence=cfg.min_regime_confidence,
                 min_atr_pct=cfg.min_atr_percentile, max_atr_pct=cfg.max_atr_percentile)

h1_df = ds.h1_df; m15_df = ds.m15_df
h1_feats = ds.h1_features
h1_regime_list = ds.h1_combined_regime
h1_atr_pctl = ds.h1_atr_percentile
h1_rule_regime = ds.h1_rule_regime
m15_conf = ds.m15_confidence

min_h1 = cfg.seq_len_h1
warm_h1_ts = h1_df['timestamp'].iloc[min_h1 - 1]
warm_m15 = int((m15_df['timestamp'] > warm_h1_ts).sum())
warm_m15 = max(warm_m15, cfg.seq_len_m15)

listening = False; h1_signal = 0; bl = 0
position = 0; last_h1_idx = -1
max_listen = cfg.max_listen_bars

# Counters
h1_gate_signals = 0
ema22_blocks = 0
h4_blocks = 0
m15_confirmed = 0
m15_fallback = 0
m15_blocked_conf = 0
m15_blocked_hour = 0
m15_blocked_position = 0
m15_blocked_expiry = 0
total_entries = 0

# Track trade chains
listening_windows = []  # (start_m15, end_m15, direction)
trade_entries = []

for m15_i in range(warm_m15, len(m15_df)):
    ts = m15_df['timestamp'].iloc[m15_i]
    price = m15_df['close'].iloc[m15_i]
    h1_i = int((h1_df['timestamp'] <= ts).sum() - 1)

    if h1_i != last_h1_idx and h1_i >= 0:
        last_h1_idx = h1_i
        ri = h1_regime_list[h1_i]
        current_regime = ri['regime']
        atr_pct = float(h1_atr_pctl[h1_i])
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
            h1_gate_signals += 1
            # EMA22 filter
            h1_closes = h1_df['close'].values[:h1_i + 1]
            with_trend = True
            if len(h1_closes) >= 23:
                ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                slope = (ema22[-1] - ema22[-2]) / max(abs(float(ema22[-2])), 1e-12)
                with_trend = ((gd.direction == 1 and slope > 0) or
                              (gd.direction == -1 and slope < 0))
            if not with_trend:
                ema22_blocks += 1
                listening = False
                h1_signal = 0
                continue  # goes to next M15 bar

            h1_signal = gd.direction
            listening = True
            bl = 0
        else:
            h1_signal = 0
            listening = False

    # Position management (simplified)
    if position != 0:
        # Just track that position is blocking entries
        pass

    if not listening:
        continue

    bl += 1
    if bl > max_listen:
        m15_blocked_expiry += 1
        listening = False
        continue

    if ts.hour in BLOCKED_HOURS:
        m15_blocked_hour += 1
        continue

    if position != 0:
        m15_blocked_position += 1
        continue

    conf = float(m15_conf[m15_i])
    if conf >= 0.5:
        m15_confirmed += 1
        total_entries += 1
        position = 1  # simplified - just mark as in trade
        listening = False
        trade_entries.append({'m15_i': m15_i, 'ts': str(ts)[:19], 'h1_signal': h1_signal,
                              'm15_conf': conf, 'price': price})
        # Auto-close after 4 M15 bars for tracing
        # (In reality TradeManager handles this)
    else:
        m15_blocked_conf += 1

print(f"=== Jan 2026 Entry Trace ===")
print(f"Total M15 bars: {len(m15_df) - warm_m15}")
print(f"H1 gate signals: {h1_gate_signals}")
print(f"  EMA22 blocked: {ema22_blocks}")
print(f"  Active signals: {h1_gate_signals - ema22_blocks}")
print(f"")
print(f"During listening windows:")
print(f"  M15 confirmed (conf>=0.5): {m15_confirmed}")
print(f"  M15 confidence <0.5: {m15_blocked_conf}")
print(f"  M15 blocked by hour: {m15_blocked_hour}")
print(f"  M15 blocked by position: {m15_blocked_position}")
print(f"  Listening window expired: {m15_blocked_expiry}")
print(f"")
print(f"Total entries: {total_entries}")
