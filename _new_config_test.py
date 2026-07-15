"""Test new config: simplified entry (no M15 NN/EMA), hour filter, early MAE exit, 3-bar min wait."""
import sys, os, json, numpy as np, pandas as pd, torch
from collections import Counter
sys.path.insert(0, ".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = CNNLSTMEncoder(
    n_features=config.n_features, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
    bidirectional=True).to(device).eval()
classifier = RegimeClassifier(embedding_dim=config.embedding_dim, n_classes=config.regime_classes).to(device).eval()
ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder.pt"), map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

engine = BTCFeatureEngine()
gate = EntryGate()

h1f = pd.read_csv(os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv"))
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)

ft = pd.Timestamp("2026-01-01", tz="UTC")
et = pd.Timestamp("2026-05-06", tz="UTC")
h1f = h1f[(h1f["timestamp"] >= ft) & (h1f["timestamp"] < et)].reset_index(drop=True)
m15f = m15f[(m15f["timestamp"] >= ft) & (m15f["timestamp"] < et)].reset_index(drop=True)

tm = TradeManager(initial_sl=config.initial_sl, hard_tp=config.hard_tp,
                  breakeven_trigger=config.breakeven_trigger,
                  trail_trigger=config.trail_trigger,
                  trail_dist=config.trail_dist, trail_dist_s=config.trail_dist_s,
                  regime_tighten=config.regime_tighten,
                  max_hold=config.max_hold_bars,
                  mae_guard_retrace=config.mae_guard_retrace)
executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)

# ── NEW CONFIG ──
ALLOWED_HOURS = {1, 5, 10, 13, 14, 16, 20}
MIN_LISTEN_BARS = 3          # minimum bars before entry (spike filter)
EARLY_MAE_THRESHOLD = -0.30  # exit if MAE < -0.3R in first 3 bars
EARLY_MAE_MAX_BARS = 3       # only check early MAE in first 3 bars

bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []

entry_ts = None; entry_hour = 0; direction_str = ""; entry_price_val = 0.0
h1_regime_val = ""; h1_regime_conf_val = 0.0
entry_regime = ""; entry_conf = 0.0

signals_generated = 0
signals_blocked_hour = 0
signals_blocked_bars = 0
signals_expired = 0

for i in range(max(config.seq_len_m15, 20), len(m15f)):
    ts = m15f["timestamp"].iloc[i]
    price = m15f["close"].iloc[i]
    executor._current_price = price

    today = ts.date()
    if ld and today != ld:
        pnl_d = 0.0
        sb = bal
    ld = today

    h1s = h1f[h1f["timestamp"] <= ts]
    m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
    if len(h1s) < config.seq_len_h1:
        continue

    hl = h1s["timestamp"].max()
    if hl != lh:
        lh = hl
        h1_feats = engine.compute(h1s)
        seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
        t = torch.from_numpy(seq).unsqueeze(0).to(device)
        for _, row in h1s.iloc[-14:].iterrows():
            rd.update(row["high"], row["low"], row["close"])
        rr = classify_regime(encoder, classifier, t, rd,
                             model_confidence_threshold=config.min_regime_confidence)
        g = gate.evaluate(rr["regime"], rr["confidence"],
                          rr.get("atr_percentile", 0.5),
                          bb_position=h1_feats[-1, 4])
        if g.entry_signal:
            h1_sig = g.direction
            listen = True
            bl = 0
            h1_atr = h1_feats[-1, 6] * price
            entry_regime = rr["regime"]
            entry_conf = g.confidence
            signals_generated += 1
        else:
            h1_sig = None
            listen = False

    if pos != 0 and tm.state is not None:
        hi = m15s["high"].iloc[-1]
        lo = m15s["low"].iloc[-1]
        epx = None; er = None
        s2 = tm.state
        sd2 = 1.0 * s2.entry_atr
        mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
        mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
        ab.append({"bar": len(ab), "mfe": float(mfe_now), "mae": float(mae_now),
                   "phase": s2.phase.name, "price": float(price), "sl": float(s2.current_sl)})

        # ── EARLY MAE EXIT: if MAE < -0.3R within first 3 bars, cut loss ──
        if len(ab) <= EARLY_MAE_MAX_BARS and mae_now < EARLY_MAE_THRESHOLD:
            epx = price
            er = "early_mae"
        elif tm.check_sl_hit(lo, hi):
            epx = tm.exit_price_at_sl()
            er = "sl_hit"
        elif tm.check_tp_hit(lo, hi):
            epx = tm.exit_price_at_tp()
            er = "tp_hit"
        else:
            a = tm.update(price, hi, lo, h1_atr)
            if a.action_type == TradeActionType.CLOSE:
                epx = price
                er = a.reason

        if epx:
            pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
            pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
            bal += pnl_dollar
            pnl_d += pnl_dollar

            mfe_peak = max(b["mfe"] for b in ab) if ab else 0.0
            mae_trough = min(b["mae"] for b in ab) if ab else 0.0
            peak_bar = next((b["bar"] for b in ab if b["mfe"] >= mfe_peak * 0.95), len(ab) - 1) \
                       if mfe_peak > 0.01 else 0

            trades.append({
                "entry_ts": entry_ts.isoformat() if entry_ts else "",
                "entry_hour_utc": entry_hour,
                "direction": direction_str,
                "entry_price": entry_price_val,
                "exit_price": round(epx, 2),
                "h1_regime": h1_regime_val,
                "h1_regime_confidence": round(h1_regime_conf_val, 4),
                "bars_listened": bl if pos == h1_sig else 0,
                "pnl_dollar": round(pnl_dollar, 2),
                "pnl_r": round(pnl_r, 4),
                "mfe_peak": round(mfe_peak, 4),
                "mae_trough": round(mae_trough, 4),
                "bars_held": len(ab),
                "exit_reason": er,
                "peak_bar": peak_bar,
                "per_bar": json.dumps(ab) if ab else "[]",
            })

            pos = 0
            tm.state = None
            ab = []
        continue

    if not listen:
        continue

    bl += 1
    if bl > config.max_listen_bars:
        listen = False
        h1_sig = None
        signals_expired += 1
        continue

    # ── NEW ENTRY CONFIRMATION ──
    # 1. Minimum 3 bars to develop (spike filter — avoids peak bar 1-2 entries)
    if bl < MIN_LISTEN_BARS:
        signals_blocked_bars += 1
        continue

    # 2. Hour filter — only trade during best-performing hours
    if ts.hour not in ALLOWED_HOURS:
        signals_blocked_hour += 1
        continue

    # 3. Daily loss limit
    if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss:
        continue

    # ── ENTRY — simplified, no NN model / EMA rule checks ──
    listen = False

    entry_ts = ts
    entry_hour = ts.hour
    direction_str = "LONG" if h1_sig == 1 else "SHORT"
    entry_price_val = price
    h1_regime_val = entry_regime
    h1_regime_conf_val = entry_conf

    lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
    tm.enter(h1_sig, price, h1_atr, lots)
    executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
    pos = h1_sig

# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
wins = [t for t in trades if t["pnl_r"] > 0]
losses = [t for t in trades if t["pnl_r"] <= 0]
n = len(trades)
wr = len(wins) / n * 100 if n else 0
tg = sum(t["pnl_r"] for t in wins)
tl = abs(sum(t["pnl_r"] for t in losses))
pf = tg / max(tl, 0.001)
total_pnl = sum(t["pnl_dollar"] for t in trades)
avg_win = np.mean([t["pnl_r"] for t in wins]) if wins else 0
avg_loss = np.mean([t["pnl_r"] for t in losses]) if losses else 0
avg_mfe_win = np.mean([t["mfe_peak"] for t in wins]) if wins else 0
avg_mfe_loss = np.mean([t["mfe_peak"] for t in losses]) if losses else 0

print("=" * 60)
print("NEW CONFIG TEST RESULTS")
print("=" * 60)
print(f"  Simplified entry (no M15 NN/EMA), {MIN_LISTEN_BARS}-bar min wait")
print(f"  Allowed hours: {sorted(ALLOWED_HOURS)} UTC")
print(f"  Early MAE exit: {EARLY_MAE_THRESHOLD}R within {EARLY_MAE_MAX_BARS} bars")
print(f"  Period: {ft.date()} to {et.date()}")
print()
print(f"  Signals generated:  {signals_generated}")
print(f"  Blocked by <{MIN_LISTEN_BARS} bars: {signals_blocked_bars}")
print(f"  Blocked by hour filter: {signals_blocked_hour}")
print(f"  Expired (>{config.max_listen_bars} bars): {signals_expired}")
print()
print(f"  Total trades:  {n}")
print(f"  Wins:          {len(wins)}")
print(f"  Losses:        {len(losses)}")
print(f"  Win Rate:      {wr:.1f}%")
print(f"  Profit Factor: {pf:.2f}")
print(f"  Total PnL:     ${total_pnl:,.0f}")
print(f"  Return:        {total_pnl/10000*100:.1f}%")
print(f"  Avg Win:       {avg_win:+.3f}R")
print(f"  Avg Loss:      {avg_loss:+.3f}R")
print(f"  Avg MFE (wins):  {avg_mfe_win:+.3f}R")
print(f"  Avg MFE (losses):{avg_mfe_loss:+.3f}R")
print(f"  Expectancy:    {np.mean([t['pnl_r'] for t in trades]):+.4f}R/trade")

print(f"\n  Exit reason breakdown:")
exit_counts = Counter(t["exit_reason"] for t in trades)
for reason, count in exit_counts.most_common():
    r_trades = [t for t in trades if t["exit_reason"] == reason]
    r_avg = np.mean([t["pnl_r"] for t in r_trades])
    print(f"    {reason:>15s}: {count:4d} ({count/n*100:5.1f}%)  avg={r_avg:+.3f}R")

print(f"\n  Win size distribution:")
for lo, hi, label in [(0, 0.25, "0.00-0.25R (micro)"), (0.25, 0.50, "0.25-0.50R"),
                        (0.50, 1.00, "0.50-1.00R"), (1.00, 2.00, "1.00-2.00R"),
                        (2.00, 3.00, "2.00-3.00R"), (3.00, 99, "3.00R (TP)")]:
    bucket = [t for t in wins if lo <= t["pnl_r"] < hi]
    pct = len(bucket) / len(wins) * 100 if wins else 0
    print(f"    {label:>20s}: {len(bucket):4d} ({pct:5.1f}% of wins)")

print(f"\n  Hour distribution:")
hourly = Counter(t["entry_hour_utc"] for t in trades)
for hr in sorted(hourly):
    h_trades = [t for t in trades if t["entry_hour_utc"] == hr]
    h_wins = [t for t in h_trades if t["pnl_r"] > 0]
    h_wr = len(h_wins) / len(h_trades) * 100 if h_trades else 0
    bad = " <-- BLOCKED" if hr not in ALLOWED_HOURS else ""
    print(f"    UTC {hr:2d}: {len(h_trades):4d} trades, WR={h_wr:.0f}%, sumPnL={sum(t['pnl_r'] for t in h_trades):+.1f}R{bad}")

print(f"\n  Peak bar vs win rate:")
for lo, hi, label in [(1, 2, "bar 1-2"), (3, 4, "bar 3-4"), (5, 10, "bar 5-10"), (11, 999, "bar 11+")]:
    bucket = [t for t in trades if lo <= t["peak_bar"] <= hi]
    if not bucket: continue
    b_wins = [t for t in bucket if t["pnl_r"] > 0]
    b_wr = len(b_wins) / len(bucket) * 100
    avg_pnl = np.mean([t["pnl_r"] for t in bucket])
    print(f"    {label:>12s}: {len(bucket):4d} trades, WR={b_wr:.0f}%, avg={avg_pnl:+.3f}R")

# ── Baseline comparison ──
print(f"\n{'='*60}")
print("BASELINE COMPARISON")
print("=" * 60)
BASELINE = {"trades": 697, "wr": 56.4, "pf": 1.18, "pnl": 9681, "avg_win": 0.792, "avg_loss": -0.867}
print(f"  {'':>20s}  {'BASELINE':>10s}  {'NEW':>10s}  {'DELTA':>10s}")
print(f"  {'Trades':>20s}  {BASELINE['trades']:10d}  {n:10d}  {n - BASELINE['trades']:+10d}")
print(f"  {'Win Rate':>20s}  {BASELINE['wr']:9.1f}%  {wr:9.1f}%  {wr - BASELINE['wr']:+9.1f}pp")
print(f"  {'Profit Factor':>20s}  {BASELINE['pf']:10.2f}  {pf:10.2f}  {pf - BASELINE['pf']:+10.2f}")
print(f"  {'Total PnL ($)':>20s}  {BASELINE['pnl']:10,.0f}  {total_pnl:10,.0f}  {total_pnl - BASELINE['pnl']:+10,.0f}")
print(f"  {'Avg Win (R)':>20s}  {BASELINE['avg_win']:+10.3f}  {avg_win:+10.3f}  {avg_win - BASELINE['avg_win']:+10.3f}")
print(f"  {'Avg Loss (R)':>20s}  {BASELINE['avg_loss']:+10.3f}  {avg_loss:+10.3f}  {avg_loss - BASELINE['avg_loss']:+10.3f}")
print(f"  {'Expectancy (R)':>20s}  {BASELINE['pnl']/10000/10:.10f}  {np.mean([t['pnl_r'] for t in trades]):+10.4f}")
