"""Comprehensive per-trade logger: entry quality, MFE/MAE per bar, CSV export, 6-question analysis."""
import sys, os, json, numpy as np, pandas as pd, torch
from collections import defaultdict
sys.path.insert(0, ".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
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

m15_model = CNNGRUM15(
    n_features=config.n_features, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
mc = torch.load(os.path.join(config.model_dir, "btc_m15_model.pt"), map_location=device, weights_only=False)
m15_model.load_state_dict(mc["model_state_dict"])

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

bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []

# Entry-time metrics (set at entry, used at exit)
entry_ts = None; entry_hour = 0; direction_str = ""; entry_price_val = 0.0
h1_regime_val = ""; h1_regime_conf_val = 0.0
ema_dist_r = 0.0; three_bar_mom_r = 0.0; atr_ratio_val = 0.0
realized_vol = 0.0; m15_conf_val = 0.0; m15_bias_val = 0.0
conf_method_str = ""; bars_listened_val = 0; with_trend_bool = False

# Store H1 regime info from signal generation (may differ from entry time)
entry_regime = ""; entry_conf = 0.0

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

        if tm.check_sl_hit(lo, hi):
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

            first_3 = [b for b in ab if b["bar"] < 3]
            first_3bar_mae = min(b["mae"] for b in first_3) if first_3 else 0.0

            trades.append({
                "entry_ts": entry_ts.isoformat() if entry_ts else "",
                "entry_hour_utc": entry_hour,
                "direction": direction_str,
                "entry_price": entry_price_val,
                "exit_price": round(epx, 2),
                "h1_regime": h1_regime_val,
                "h1_regime_confidence": round(h1_regime_conf_val, 4),
                "m15_ema_distance_r": round(ema_dist_r, 4),
                "m15_3bar_momentum_r": round(three_bar_mom_r, 4),
                "m15_atr_ratio": round(atr_ratio_val, 4),
                "m15_5min_realized_vol": round(realized_vol, 4),
                "m15_entry_confidence": round(m15_conf_val, 4),
                "m15_direction_bias": round(m15_bias_val, 4),
                "confirmation_method": conf_method_str,
                "bars_listened": bars_listened_val,
                "pnl_dollar": round(pnl_dollar, 2),
                "pnl_r": round(pnl_r, 4),
                "mfe_peak": round(mfe_peak, 4),
                "mae_trough": round(mae_trough, 4),
                "bars_held": len(ab),
                "exit_reason": er,
                "peak_bar": peak_bar,
                "first_3bar_mae": round(first_3bar_mae, 4),
                "with_h1_trend": with_trend_bool,
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
        continue

    m15_feats = engine.compute(m15s)
    confirmed = False
    conf_method = "none"
    sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
    tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
    with torch.no_grad():
        mo = m15_model(tt2)
    m15_conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
    m15_bias = mo["direction_bias"].item() if hasattr(mo["direction_bias"], "item") else float(mo["direction_bias"])

    if m15_conf >= config.min_entry_confidence:
        if (h1_sig == 1 and m15_bias > 0) or (h1_sig == -1 and m15_bias < 0):
            confirmed = True
            conf_method = "nn_model"

    if not confirmed:
        mc2 = m15s["close"].values
        ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
        if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]:
            confirmed = True
            conf_method = "ema_rule"
        elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]:
            confirmed = True
            conf_method = "ema_rule"

    if not confirmed:
        continue

    if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss:
        continue

    # ── ENTRY — capture all entry-time metrics ──
    listen = False

    entry_ts = ts
    entry_hour = ts.hour
    direction_str = "LONG" if h1_sig == 1 else "SHORT"
    entry_price_val = price
    h1_regime_val = entry_regime
    h1_regime_conf_val = entry_conf
    bars_listened_val = bl

    # 1. M15 EMA distance
    m15_closes = m15s["close"].values
    ema21_vals = pd.Series(m15_closes).ewm(span=21, adjust=False).mean().values
    current_ema21 = ema21_vals[-1]
    ema_dist_r = (price - current_ema21) / max(h1_atr, 1e-12)

    # 2. 3-bar momentum
    if len(m15_closes) >= 4:
        chg1 = m15_closes[-1] - m15_closes[-2]
        chg2 = m15_closes[-2] - m15_closes[-3]
        chg3 = m15_closes[-3] - m15_closes[-4]
        three_bar_mom_r = (chg1 + chg2 + chg3) / max(h1_atr, 1e-12)
    else:
        three_bar_mom_r = 0.0

    # 3. ATR ratio: current bar ATR / 20-bar average ATR
    h_vals = m15s["high"].values; l_vals = m15s["low"].values; c_vals = m15s["close"].values
    prev_c = np.roll(c_vals, 1); prev_c[0] = c_vals[0]
    tr_vals = np.maximum(h_vals - l_vals,
                         np.maximum(np.abs(h_vals - prev_c), np.abs(l_vals - prev_c)))
    atr_ema = pd.Series(tr_vals).ewm(span=14, adjust=False).mean().values
    current_atr_val = atr_ema[-1]
    avg_atr_20 = np.mean(tr_vals[-20:]) if len(tr_vals) >= 20 else current_atr_val
    atr_ratio_val = current_atr_val / max(avg_atr_20, 1e-12)

    # 4. 5-min realized volatility
    realized_vol = (h_vals[-1] - l_vals[-1]) / max(c_vals[-1], 1e-12)

    # 5. M15 model outputs
    m15_conf_val = m15_conf
    m15_bias_val = m15_bias
    conf_method_str = conf_method

    # 6. With H1 trend check
    h1_closes = h1s["close"].values
    if len(h1_closes) >= 23:
        h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
        h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
    else:
        h1_slope = 0.0
    with_trend_bool = (direction_str == "LONG" and h1_slope > 0) or (direction_str == "SHORT" and h1_slope < 0)

    # Execute entry
    lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
    tm.enter(h1_sig, price, h1_atr, lots)
    executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
    pos = h1_sig

# ═══════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════
wins = [t for t in trades if t["pnl_r"] > 0]
losses = [t for t in trades if t["pnl_r"] <= 0]

CSV_COLUMNS = [
    "entry_ts", "entry_hour_utc", "direction", "entry_price", "exit_price",
    "h1_regime", "h1_regime_confidence",
    "m15_ema_distance_r", "m15_3bar_momentum_r", "m15_atr_ratio",
    "m15_5min_realized_vol", "m15_entry_confidence", "m15_direction_bias",
    "confirmation_method", "bars_listened",
    "pnl_dollar", "pnl_r", "mfe_peak", "mae_trough",
    "bars_held", "exit_reason", "peak_bar", "first_3bar_mae",
    "with_h1_trend", "per_bar",
]

log_dir = config.log_dir
pd.DataFrame(wins, columns=CSV_COLUMNS).to_csv(os.path.join(log_dir, "btc_wins.csv"), index=False)
pd.DataFrame(losses, columns=CSV_COLUMNS).to_csv(os.path.join(log_dir, "btc_losses.csv"), index=False)
pd.DataFrame(trades, columns=CSV_COLUMNS).to_csv(os.path.join(log_dir, "btc_all_trades.csv"), index=False)

n = len(trades)
wr = len(wins) / n * 100 if n else 0
tg = sum(t["pnl_r"] for t in wins)
tl = abs(sum(t["pnl_r"] for t in losses))
pf = tg / max(tl, 0.001)
total_pnl = sum(t["pnl_dollar"] for t in trades)
avg_win = np.mean([t["pnl_r"] for t in wins]) if wins else 0
avg_loss = np.mean([t["pnl_r"] for t in losses]) if losses else 0

print(f"Wins: {len(wins)}, Losses: {len(losses)}, Total: {n}")
print(f"WR: {wr:.1f}%, PF: {pf:.2f}, Total PnL: ${total_pnl:,.0f}")
print(f"Avg Win: {avg_win:+.3f}R, Avg Loss: {avg_loss:+.3f}R")
print(f"CSVs written to {log_dir}/")

# ═══════════════════════════════════════════════════════════════════
# 6-QUESTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════
report = []
R = "=" * 70

report.append(R)
report.append("BTC TRADE ANALYSIS REPORT")
report.append(f"Period: {ft.date()} to {et.date()}")
report.append(f"Total: {n} trades  |  Wins: {len(wins)}  |  Losses: {len(losses)}")
report.append(f"Win Rate: {wr:.1f}%  |  PF: {pf:.2f}  |  Total PnL: ${total_pnl:,.0f}")
report.append(f"Avg Win: {avg_win:+.3f}R  |  Avg Loss: {avg_loss:+.3f}R")
report.append(R)

# ── Q1: Entry Confirmation Strength ──
report.append("\nQ1: ENTRY CONFIRMATION STRENGTH — Wins vs Losses")
report.append("=" * 70)

metrics = [
    ("m15_ema_distance_r", "M15 EMA distance (R)", "Price distance from EMA21 in ATR multiples"),
    ("m15_3bar_momentum_r", "3-bar momentum (R)", "Sum of last 3 price changes in ATR multiples"),
    ("m15_atr_ratio", "ATR ratio", "Current ATR / 20-bar avg ATR"),
    ("m15_5min_realized_vol", "Realized vol (5-min)", "Current bar (high-low)/close"),
]
for col, label, desc in metrics:
    w_vals = np.array([t[col] for t in wins], dtype=float)
    l_vals = np.array([t[col] for t in losses], dtype=float)
    report.append(f"\n  {label} — {desc}")
    report.append(f"    {'Wins (N=' + str(len(w_vals)) + ')':>20s}    {'Losses (N=' + str(len(l_vals)) + ')':>20s}")
    report.append(f"    Mean:  {np.mean(w_vals):+12.4f}        Mean:  {np.mean(l_vals):+12.4f}")
    report.append(f"    Std:   {np.std(w_vals):12.4f}        Std:   {np.std(l_vals):12.4f}")
    report.append(f"    Median:{np.median(w_vals):+12.4f}        Median:{np.median(l_vals):+12.4f}")
    p25w, p75w = np.percentile(w_vals, [25, 75])
    p25l, p75l = np.percentile(l_vals, [25, 75])
    report.append(f"    [25th, 75th]: [{p25w:+.4f}, {p75w:+.4f}]    [25th, 75th]: [{p25l:+.4f}, {p75l:+.4f}]")

# M15 confidence and bias
for col, label in [("m15_entry_confidence", "M15 Entry Confidence"), ("m15_direction_bias", "M15 Direction Bias")]:
    w_vals = np.array([t[col] for t in wins], dtype=float)
    l_vals = np.array([t[col] for t in losses], dtype=float)
    report.append(f"\n  {label}")
    report.append(f"    {'Wins':>20s}    {'Losses':>20s}")
    report.append(f"    Mean:  {np.mean(w_vals):12.4f}        Mean:  {np.mean(l_vals):12.4f}")
    report.append(f"    Median:{np.median(w_vals):12.4f}        Median:{np.median(l_vals):12.4f}")

# Confirmation method breakdown
for method in ["nn_model", "ema_rule"]:
    m_trades = [t for t in trades if t["confirmation_method"] == method]
    m_wins = [t for t in m_trades if t["pnl_r"] > 0]
    m_wr = len(m_wins) / len(m_trades) * 100 if m_trades else 0
    report.append(f"\n  {method}: {len(m_trades)} trades, WR={m_wr:.1f}%, AvgPnL={np.mean([t['pnl_r'] for t in m_trades]):+.4f}R" if m_trades else f"\n  {method}: 0 trades")

# ── Q2: Time of Day ──
report.append(f"\n\nQ2: TIME OF DAY DISTRIBUTION (UTC)")
report.append("=" * 70)
hourly = defaultdict(list)
for t in trades:
    hourly[t["entry_hour_utc"]].append(t["pnl_r"])
report.append(f"  {'Hour':>5s}  {'Trades':>7s}  {'Wins':>6s}  {'Losses':>7s}  {'WinRate':>8s}  {'AvgPnLR':>9s}  {'SumPnLR':>9s}")
for hr in sorted(hourly):
    vals = hourly[hr]
    wins_h = sum(1 for v in vals if v > 0)
    losses_h = len(vals) - wins_h
    wr_h = wins_h / len(vals) * 100
    report.append(f"  {hr:5d}  {len(vals):7d}  {wins_h:6d}  {losses_h:7d}  {wr_h:7.1f}%  {np.mean(vals):+9.4f}R  {sum(vals):+9.2f}R")

# ── Q3: 3-Bar Momentum Filter Simulation ──
report.append(f"\n\nQ3: 3-BAR MOMENTUM FILTER SIMULATION (require > 0.2R)")
report.append("=" * 70)

dz_trades = [t for t in trades if 0.25 <= t["mfe_peak"] < 0.50]
filtered_dz = [t for t in dz_trades if t["m15_3bar_momentum_r"] <= 0.2]
filtered_in = [t for t in trades if t["m15_3bar_momentum_r"] > 0.2]
filtered_wins = [t for t in filtered_in if t["pnl_r"] > 0]
filtered_losses = [t for t in filtered_in if t["pnl_r"] <= 0]
f_tg = sum(t["pnl_r"] for t in filtered_wins)
f_tl = abs(sum(t["pnl_r"] for t in filtered_losses))
f_pf = f_tg / max(f_tl, 0.001)
f_wr = len(filtered_wins) / len(filtered_in) * 100 if filtered_in else 0
f_pnl = sum(t["pnl_dollar"] for t in filtered_in)

# Counterfactual: what if we skip trades with momentum <= 0.2 instead of taking them?
skipped = [t for t in trades if t["m15_3bar_momentum_r"] <= 0.2]
kept = [t for t in trades if t["m15_3bar_momentum_r"] > 0.2]
k_wins = [t for t in kept if t["pnl_r"] > 0]
k_losses = [t for t in kept if t["pnl_r"] <= 0]
k_tg = sum(t["pnl_r"] for t in k_wins)
k_tl = abs(sum(t["pnl_r"] for t in k_losses))
k_pf = k_tg / max(k_tl, 0.001)
k_wr = len(k_wins) / len(kept) * 100 if kept else 0
k_pnl = sum(t["pnl_dollar"] for t in kept)
k_avg_win = np.mean([t["pnl_r"] for t in k_wins]) if k_wins else 0
k_avg_loss = np.mean([t["pnl_r"] for t in k_losses]) if k_losses else 0

report.append(f"  Original (all trades):     N={n:4d}  WR={wr:.1f}%  PF={pf:.2f}  PnL=${total_pnl:,.0f}")
report.append(f"  Filter (mom > 0.2R only):  N={len(kept):4d}  WR={k_wr:.1f}%  PF={k_pf:.2f}  PnL=${k_pnl:,.0f}")
report.append(f"  Trades skipped (mom <= 0.2R): {len(skipped)} ({len(skipped)/n*100:.1f}%)")
report.append(f"    Skipped avg PnL: {np.mean([t['pnl_r'] for t in skipped]):+.4f}R  (would have been these results)")

report.append(f"\n  Death-zone trades (0.25-0.50R MFE): {len(dz_trades)}")
report.append(f"    Of these, filtered (mom <= 0.2R):  {len(filtered_dz)} ({len(filtered_dz)/max(len(dz_trades),1)*100:.1f}%)")
report.append(f"    Death-zone avg PnL of filtered:    {np.mean([t['pnl_r'] for t in filtered_dz]):+.4f}R" if filtered_dz else "    N/A")

report.append(f"\n  Momentum threshold sensitivity:")
for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    kept_t = [t for t in trades if t["m15_3bar_momentum_r"] > thresh]
    kept_w = [t for t in kept_t if t["pnl_r"] > 0]
    kept_l = [t for t in kept_t if t["pnl_r"] <= 0]
    ktg = sum(t["pnl_r"] for t in kept_w)
    ktl = abs(sum(t["pnl_r"] for t in kept_l))
    kpf = ktg / max(ktl, 0.001)
    kwr = len(kept_w) / len(kept_t) * 100 if kept_t else 0
    kpnl = sum(t["pnl_dollar"] for t in kept_t)
    skipped_n = n - len(kept_t)
    report.append(f"    > {thresh:.2f}R: N={len(kept_t):4d}  WR={kwr:.1f}%  PF={kpf:.2f}  PnL=${kpnl:,.0f}  skipped={skipped_n} ({skipped_n/n*100:.0f}%)")

# ── Q4: MAE in First 3 Bars (0.25-0.50R bucket) ──
report.append(f"\n\nQ4: MAE IN FIRST 3 BARS — 0.25-0.50R MFE BUCKET")
report.append("=" * 70)
if dz_trades:
    maes_3 = [t["first_3bar_mae"] for t in dz_trades]
    report.append(f"  Count: {len(dz_trades)} trades in this bucket")
    report.append(f"  Avg MAE in bars 0-3:  {np.mean(maes_3):.4f}R")
    report.append(f"  Median MAE:           {np.median(maes_3):.4f}R")
    report.append(f"  Min MAE:              {min(maes_3):.4f}R")
    report.append(f"  Max MAE:              {max(maes_3):.4f}R")
    p25, p75 = np.percentile(maes_3, [25, 75])
    report.append(f"  25th/75th pctl:       [{p25:.4f}R, {p75:.4f}R]")
    # Direction breakdown
    for d in ["LONG", "SHORT"]:
        d_trades = [t for t in dz_trades if t["direction"] == d]
        if d_trades:
            d_maes = [t["first_3bar_mae"] for t in d_trades]
            report.append(f"  {d}: {len(d_trades)} trades, avg first-3 MAE={np.mean(d_maes):.4f}R")
    # Win vs loss
    dz_wins = [t for t in dz_trades if t["pnl_r"] > 0]
    dz_losses = [t for t in dz_trades if t["pnl_r"] <= 0]
    report.append(f"  Wins in bucket:   {len(dz_wins)}, avg first-3 MAE={np.mean([t['first_3bar_mae'] for t in dz_wins]):.4f}R" if dz_wins else "")
    report.append(f"  Losses in bucket: {len(dz_losses)}, avg first-3 MAE={np.mean([t['first_3bar_mae'] for t in dz_losses]):.4f}R" if dz_losses else "")
    # High MAE > 0.2R flag
    high_mae = [t for t in dz_trades if t["first_3bar_mae"] < -0.2]
    report.append(f"  Trades with MAE < -0.2R in first 3 bars: {len(high_mae)} ({len(high_mae)/len(dz_trades)*100:.1f}%)")
    if high_mae:
        report.append(f"    Avg exit for these: {np.mean([t['pnl_r'] for t in high_mae]):+.4f}R")

# ── Q5: Win Rate by MFE Peak Bar ──
report.append(f"\n\nQ5: WIN RATE BY MFE PEAK BAR")
report.append("=" * 70)
peak_bins = [(1, 2, "bar 1-2"), (3, 4, "bar 3-4"), (5, 10, "bar 5-10"), (11, 999, "bar 11+")]
report.append(f"  {'Peak Bar':>12s}  {'Trades':>7s}  {'Wins':>6s}  {'Losses':>7s}  {'WinRate':>8s}  {'AvgMFE':>8s}  {'AvgPnL':>8s}  {'AvgExit':>8s}")
for lo, hi, label in peak_bins:
    bucket = [t for t in trades if lo <= t["peak_bar"] <= hi]
    if not bucket:
        continue
    b_wins = [t for t in bucket if t["pnl_r"] > 0]
    b_losses = [t for t in bucket if t["pnl_r"] <= 0]
    b_wr = len(b_wins) / len(bucket) * 100
    avg_mfe = np.mean([t["mfe_peak"] for t in bucket])
    avg_pnl = np.mean([t["pnl_r"] for t in bucket])
    avg_exit_method = max(set(t["exit_reason"] for t in bucket), key=lambda x: sum(1 for t in bucket if t["exit_reason"] == x))
    report.append(f"  {label:>12s}  {len(bucket):7d}  {len(b_wins):6d}  {len(b_losses):7d}  {b_wr:7.1f}%  {avg_mfe:+7.3f}R  {avg_pnl:+7.3f}R  {avg_exit_method:>8s}")

# Also show for losses only
loss_peak_bins = [(1, 2, "Losses: peak bar 1-2"), (3, 4, "Losses: peak bar 3-4"), (5, 999, "Losses: peak bar 5+")]
report.append(f"\n  Losses by peak bar:")
for lo, hi, label in loss_peak_bins:
    lb = [t for t in losses if lo <= t["peak_bar"] <= hi]
    if lb:
        report.append(f"    {label}: {len(lb)}/{len(losses)} ({len(lb)/max(len(losses),1)*100:.1f}%), avg MFE={np.mean([t['mfe_peak'] for t in lb]):.3f}R, avg exit={np.mean([t['pnl_r'] for t in lb]):.3f}R")

# ── Q6: Correlation with H1 Trend ──
report.append(f"\n\nQ6: CORRELATION WITH HIGHER TIMEFRAME TREND")
report.append("=" * 70)

with_t = [t for t in trades if t["with_h1_trend"]]
against_t = [t for t in trades if not t["with_h1_trend"]]
wt_wins = [t for t in with_t if t["pnl_r"] > 0]
at_wins = [t for t in against_t if t["pnl_r"] > 0]
wt_wr = len(wt_wins) / len(with_t) * 100 if with_t else 0
at_wr = len(at_wins) / len(against_t) * 100 if against_t else 0
wt_pf_num = sum(t["pnl_r"] for t in wt_wins)
wt_pf_den = abs(sum(t["pnl_r"] for t in with_t if t["pnl_r"] <= 0))
at_pf_num = sum(t["pnl_r"] for t in at_wins)
at_pf_den = abs(sum(t["pnl_r"] for t in against_t if t["pnl_r"] <= 0))
wt_pf = wt_pf_num / max(wt_pf_den, 0.001)
at_pf = at_pf_num / max(at_pf_den, 0.001)

report.append(f"  All trades:")
report.append(f"    With H1 trend:    N={len(with_t):4d}  WR={wt_wr:.1f}%  PF={wt_pf:.2f}  AvgPnL={np.mean([t['pnl_r'] for t in with_t]):+.4f}R  SumPnL=${sum(t['pnl_dollar'] for t in with_t):,.0f}")
report.append(f"    Against H1 trend: N={len(against_t):4d}  WR={at_wr:.1f}%  PF={at_pf:.2f}  AvgPnL={np.mean([t['pnl_r'] for t in against_t]):+.4f}R  SumPnL=${sum(t['pnl_dollar'] for t in against_t):,.0f}")

report.append(f"\n  0.25-0.50R MFE bucket ({len(dz_trades)} trades):")
dz_with = [t for t in dz_trades if t["with_h1_trend"]]
dz_against = [t for t in dz_trades if not t["with_h1_trend"]]
report.append(f"    With H1 trend:    {len(dz_with)} ({len(dz_with)/max(len(dz_trades),1)*100:.1f}%) — avg PnL={np.mean([t['pnl_r'] for t in dz_with]):+.4f}R" if dz_with else "    With H1 trend: 0")
report.append(f"    Against H1 trend: {len(dz_against)} ({len(dz_against)/max(len(dz_trades),1)*100:.1f}%) — avg PnL={np.mean([t['pnl_r'] for t in dz_against]):+.4f}R" if dz_against else "    Against H1 trend: 0")

# Also by regime
report.append(f"\n  By H1 regime:")
for regime in sorted(set(t["h1_regime"] for t in trades)):
    r_trades = [t for t in trades if t["h1_regime"] == regime]
    r_wins = [t for t in r_trades if t["pnl_r"] > 0]
    r_wr = len(r_wins) / len(r_trades) * 100 if r_trades else 0
    r_pnl_d = sum(t["pnl_dollar"] for t in r_trades)
    report.append(f"    {regime:>15s}: {len(r_trades):4d} trades, WR={r_wr:.1f}%, AvgPnL={np.mean([t['pnl_r'] for t in r_trades]):+.4f}R, PnL=${r_pnl_d:,.0f}")

# ── Summary ──
report.append(f"\n\n{'='*70}")
report.append("SUMMARY")
report.append("=" * 70)
report.append(f"  Total trades: {n}  |  Wins: {len(wins)}  |  Losses: {len(losses)}")
report.append(f"  WR: {wr:.1f}%  |  PF: {pf:.2f}  |  PnL: ${total_pnl:,.0f}")
report.append(f"  Avg Win: {avg_win:+.3f}R  |  Avg Loss: {avg_loss:+.3f}R  |  Expectancy: {np.mean([t['pnl_r'] for t in trades]):+.4f}R/trade")

# ── Write report ──
report_path = os.path.join(log_dir, "btc_trade_analysis_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report))

print(f"\nReport written to {report_path}")
