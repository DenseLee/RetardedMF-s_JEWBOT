"""Test hour+trend fixes + measure noise duration + cooldown gate."""
import sys, os, json, numpy as np, pandas as pd, torch
from collections import defaultdict, Counter
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
encoder.load_state_dict(ckpt["encoder_state_dict"]); classifier.load_state_dict(ckpt["classifier_state_dict"])

m15_model = CNNGRUM15(
    n_features=config.n_features, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
mc = torch.load(os.path.join(config.model_dir, "btc_m15_model.pt"), map_location=device, weights_only=False)
m15_model.load_state_dict(mc["model_state_dict"])

engine = BTCFeatureEngine(); gate = EntryGate()

h1f = pd.read_csv(os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv"))
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)
ft = pd.Timestamp("2026-01-01", tz="UTC"); et = pd.Timestamp("2026-05-06", tz="UTC")
h1f = h1f[(h1f["timestamp"] >= ft) & (h1f["timestamp"] < et)].reset_index(drop=True)
m15f = m15f[(m15f["timestamp"] >= ft) & (m15f["timestamp"] < et)].reset_index(drop=True)

BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}


def run_backtest(hour_filter=False, trend_filter=False, cooldown_bars=0, label=""):
    tm = TradeManager(initial_sl=config.initial_sl, hard_tp=config.hard_tp,
                      breakeven_trigger=config.breakeven_trigger,
                      trail_trigger=config.trail_trigger,
                      trail_dist=config.trail_dist, trail_dist_s=config.trail_dist_s,
                      regime_tighten=config.regime_tighten, max_hold=config.max_hold_bars,
                      mae_guard_retrace=config.mae_guard_retrace)
    executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
    h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
    lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []
    entry_regime = ""; entry_conf = 0.0

    # Cooldown gate state
    cooldown_dir = 0          # 0=none, 1=block longs, -1=block shorts
    cooldown_remaining = 0     # bars left in cooldown
    last_exit_noise = False
    noise_durations = []       # track noise trade bars_held
    stats = {"blocked_hour": 0, "blocked_trend": 0, "blocked_cooldown": 0}

    for i in range(max(config.seq_len_m15, 20), len(m15f)):
        ts = m15f["timestamp"].iloc[i]; price = m15f["close"].iloc[i]
        executor._current_price = price
        today = ts.date()
        if ld and today != ld: pnl_d = 0.0; sb = bal
        ld = today

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        else:
            cooldown_dir = 0

        h1s = h1f[h1f["timestamp"] <= ts]
        m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
        if len(h1s) < config.seq_len_h1: continue

        hl = h1s["timestamp"].max()
        if hl != lh:
            lh = hl; h1_feats = engine.compute(h1s)
            seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0).to(device)
            for _, row in h1s.iloc[-14:].iterrows(): rd.update(row["high"], row["low"], row["close"])
            rr = classify_regime(encoder, classifier, t, rd, model_confidence_threshold=config.min_regime_confidence)
            g = gate.evaluate(rr["regime"], rr["confidence"], rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])

            if g.entry_signal:
                if trend_filter:
                    h1_closes = h1s["close"].values
                    if len(h1_closes) >= 23:
                        h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                        h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                        with_trend = ((g.direction == 1 and h1_slope > 0) or (g.direction == -1 and h1_slope < 0))
                        if not with_trend: stats["blocked_trend"] += 1; h1_sig = None; listen = False; continue

                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price; entry_regime = rr["regime"]; entry_conf = g.confidence
            else:
                h1_sig = None; listen = False

        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None; s2 = tm.state; sd2 = 1.0 * s2.entry_atr
            mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
            mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
            ab.append({"bar": len(ab), "mfe": float(mfe_now), "mae": float(mae_now), "phase": s2.phase.name, "price": float(price)})

            if tm.check_sl_hit(lo, hi): epx = tm.exit_price_at_sl(); er = "sl_hit"
            elif tm.check_tp_hit(lo, hi): epx = tm.exit_price_at_tp(); er = "tp_hit"
            else:
                a = tm.update(price, hi, lo, h1_atr)
                if a.action_type == TradeActionType.CLOSE: epx = price; er = a.reason

            if epx:
                pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
                pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
                bal += pnl_dollar; pnl_d += pnl_dollar
                mfe_peak = max(b["mfe"] for b in ab) if ab else 0.0

                # Track noise: loss with MFE < 0.25R
                is_noise = pnl_r <= 0 and mfe_peak <= 0.25
                if is_noise: noise_durations.append(len(ab))
                if cooldown_bars > 0 and is_noise:
                    cooldown_dir = pos  # block same direction
                    cooldown_remaining = cooldown_bars
                    last_exit_noise = True

                trades.append({"pnl_r": round(pnl_r, 4), "pnl_dollar": round(pnl_dollar, 2),
                               "mfe_peak": round(mfe_peak, 4), "bars_held": len(ab),
                               "exit_reason": er, "direction": "LONG" if pos == 1 else "SHORT"})
                pos = 0; tm.state = None; ab = []
            continue

        if not listen: continue

        bl += 1
        if bl > config.max_listen_bars: listen = False; h1_sig = None; continue

        # Hour filter
        if hour_filter and ts.hour in BLOCKED_HOURS:
            stats["blocked_hour"] += 1; continue

        # Cooldown gate: block same-direction entries after noise exit
        if cooldown_bars > 0 and cooldown_dir != 0 and h1_sig == cooldown_dir:
            stats["blocked_cooldown"] += 1; continue

        # M15 confirmation
        m15_feats = engine.compute(m15s); confirmed = False
        sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad(): mo = m15_model(tt2)
        m15_conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
        m15_bias = mo["direction_bias"].item() if hasattr(mo["direction_bias"], "item") else float(mo["direction_bias"])
        if m15_conf >= config.min_entry_confidence:
            if (h1_sig == 1 and m15_bias > 0) or (h1_sig == -1 and m15_bias < 0): confirmed = True
        if not confirmed:
            mc2 = m15s["close"].values; ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]: confirmed = True
            elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]: confirmed = True
        if not confirmed: continue
        if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss: continue

        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots)
        executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    wins = [t for t in trades if t["pnl_r"] > 0]; losses = [t for t in trades if t["pnl_r"] <= 0]
    n = len(trades); wr = len(wins)/n*100 if n else 0
    tg = sum(t["pnl_r"] for t in wins); tl = abs(sum(t["pnl_r"] for t in losses))
    pf = tg/max(tl,0.001); total_pnl = sum(t["pnl_dollar"] for t in trades)
    noise = [t for t in losses if t["mfe_peak"] <= 0.25]
    micro = [t for t in wins if t["pnl_r"] <= 0.25]
    good = [t for t in wins if t["pnl_r"] > 0.50]

    return {"label": label, "trades": n, "wins": len(wins), "losses": len(losses),
            "wr": wr, "pf": pf, "pnl": total_pnl,
            "avg_win": np.mean([t["pnl_r"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl_r"] for t in losses]) if losses else 0,
            "noise_n": len(noise), "noise_avg_bars": np.mean(noise_durations) if noise_durations else 0,
            "noise_median_bars": np.median(noise_durations) if noise_durations else 0,
            "noise_pct": len(noise)/n*100 if n else 0,
            "micro_n": len(micro), "good_n": len(good),
            "exit_reasons": dict(Counter(t["exit_reason"] for t in trades)),
            "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Baselines + measure noise duration
# ═══════════════════════════════════════════════════════════════════
results = []

print("BASELINE...")
r_base = run_backtest(label="BASELINE")
results.append(r_base)

print("HOUR + TREND...")
r_ht = run_backtest(hour_filter=True, trend_filter=True, label="HOUR+TREND")
results.append(r_ht)

# ── Noise duration analysis ──
print("\n" + "=" * 70)
print("NOISE TRADE DURATION ANALYSIS")
print("=" * 70)

# Run a detailed pass to collect noise durations with per-bar data
tm = TradeManager(initial_sl=config.initial_sl, hard_tp=config.hard_tp,
                  breakeven_trigger=config.breakeven_trigger, trail_trigger=config.trail_trigger,
                  trail_dist=config.trail_dist, trail_dist_s=config.trail_dist_s,
                  regime_tighten=config.regime_tighten, max_hold=config.max_hold_bars,
                  mae_guard_retrace=config.mae_guard_retrace)
executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
bal = 10000.0; pnl_d = 0.0; ld = None; trades_d = []; sb = 10000.0
h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []
entry_regime = ""; entry_conf = 0.0
noise_details = []

for i in range(max(config.seq_len_m15, 20), len(m15f)):
    ts = m15f["timestamp"].iloc[i]; price = m15f["close"].iloc[i]
    executor._current_price = price
    today = ts.date()
    if ld and today != ld: pnl_d = 0.0; sb = bal
    ld = today
    h1s = h1f[h1f["timestamp"] <= ts]; m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
    if len(h1s) < config.seq_len_h1: continue
    hl = h1s["timestamp"].max()
    if hl != lh:
        lh = hl; h1_feats = engine.compute(h1s)
        seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
        t = torch.from_numpy(seq).unsqueeze(0).to(device)
        for _, row in h1s.iloc[-14:].iterrows(): rd.update(row["high"], row["low"], row["close"])
        rr = classify_regime(encoder, classifier, t, rd, model_confidence_threshold=config.min_regime_confidence)
        g = gate.evaluate(rr["regime"], rr["confidence"], rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])
        if g.entry_signal:
            h1_closes = h1s["close"].values
            if len(h1_closes) >= 23:
                h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                with_trend = ((g.direction == 1 and h1_slope > 0) or (g.direction == -1 and h1_slope < 0))
                if not with_trend: h1_sig = None; listen = False; continue
            if ts.hour in BLOCKED_HOURS: h1_sig = None; listen = False; continue
            h1_sig = g.direction; listen = True; bl = 0; h1_atr = h1_feats[-1, 6] * price
            entry_regime = rr["regime"]; entry_conf = g.confidence
        else: h1_sig = None; listen = False

    if pos != 0 and tm.state is not None:
        hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
        epx = None; er = None; s2 = tm.state; sd2 = 1.0 * s2.entry_atr
        mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
        mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
        ab.append({"bar": len(ab), "mfe": float(mfe_now), "mae": float(mae_now), "phase": s2.phase.name})
        if tm.check_sl_hit(lo, hi): epx = tm.exit_price_at_sl(); er = "sl_hit"
        elif tm.check_tp_hit(lo, hi): epx = tm.exit_price_at_tp(); er = "tp_hit"
        else:
            a = tm.update(price, hi, lo, h1_atr)
            if a.action_type == TradeActionType.CLOSE: epx = price; er = a.reason

        if epx:
            pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
            pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
            bal += pnl_dollar; pnl_d += pnl_dollar
            mfe_peak = max(b["mfe"] for b in ab) if ab else 0.0
            is_noise = pnl_r <= 0 and mfe_peak <= 0.25
            if is_noise:
                noise_details.append({"bars_held": len(ab), "pnl_r": round(pnl_r, 4),
                                      "mfe_peak": round(mfe_peak, 4), "exit_reason": er,
                                      "direction": "LONG" if pos == 1 else "SHORT",
                                      "mfe_bar0": ab[0]["mfe"] if ab else 0,
                                      "mfe_bar1": ab[1]["mfe"] if len(ab) > 1 else 0,
                                      "mfe_bar2": ab[2]["mfe"] if len(ab) > 2 else 0})
            pos = 0; tm.state = None; ab = []
        continue
    if not listen: continue
    bl += 1
    if bl > config.max_listen_bars: listen = False; h1_sig = None; continue
    m15_feats = engine.compute(m15s); confirmed = False
    sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
    tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
    with torch.no_grad(): mo = m15_model(tt2)
    m15_conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
    m15_bias = mo["direction_bias"].item() if hasattr(mo["direction_bias"], "item") else float(mo["direction_bias"])
    if m15_conf >= config.min_entry_confidence:
        if (h1_sig == 1 and m15_bias > 0) or (h1_sig == -1 and m15_bias < 0): confirmed = True
    if not confirmed:
        mc2 = m15s["close"].values; ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
        if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]: confirmed = True
        elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]: confirmed = True
    if not confirmed: continue
    if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss: continue
    listen = False
    lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
    tm.enter(h1_sig, price, h1_atr, lots)
    executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
    pos = h1_sig

# Noise duration stats
nd = [d["bars_held"] for d in noise_details]
print(f"  Noise trades (with hour+trend): {len(noise_details)}")
print(f"  Avg bars held:                   {np.mean(nd):.1f}")
print(f"  Median bars held:                {np.median(nd):.0f}")
print(f"  Std:                             {np.std(nd):.1f}")
print(f"  Min / Max:                       {min(nd)} / {max(nd)}")
print(f"\n  Duration distribution (bars):")
for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10), (10, 15), (15, 99)]:
    cnt = sum(1 for d in nd if lo <= d < hi)
    if cnt > 0: print(f"    {lo:2d}-{hi:2d} bars: {cnt:3d} ({cnt/len(nd)*100:.0f}%)")

mean_noise_bars = int(round(np.mean(nd))) if nd else 0
median_noise_bars = int(round(np.median(nd))) if nd else 0

print(f"\n  Avg noise duration: {mean_noise_bars} bars (rounded)")
print(f"  Median noise duration: {median_noise_bars} bars (rounded)")
print(f"\n  Recommendation: cooldown = {mean_noise_bars} bars after noise exit")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Test cooldown gate at mean noise duration
# ═══════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"COOLDOWN GATE TEST (block same direction for {mean_noise_bars} bars after noise exit)")
print("=" * 70)

print(f"\nHOUR+TREND+COOLDOWN({mean_noise_bars})...")
r_cd = run_backtest(hour_filter=True, trend_filter=True, cooldown_bars=mean_noise_bars,
                    label=f"HT+CD{mean_noise_bars}")
results.append(r_cd)

# Also test median and mean±2
for cd_bars in [median_noise_bars, mean_noise_bars + 2, mean_noise_bars - 2]:
    if cd_bars > 0 and cd_bars != mean_noise_bars:
        print(f"HOUR+TREND+COOLDOWN({cd_bars})...")
        r = run_backtest(hour_filter=True, trend_filter=True, cooldown_bars=cd_bars,
                        label=f"HT+CD{cd_bars}")
        results.append(r)

# ═══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
print(f"\n\n{'='*100}")
print("FULL RESULTS")
print("=" * 100)
HDR = f"  {'Method':<24s} {'Trds':>5s} {'WR':>6s} {'PF':>6s} {'PnL':>10s} {'AvgW':>7s} {'AvgL':>7s} {'Noise':>6s} {'Noise%':>7s} {'NoiseBars':>10s}"
print(HDR)
print("  " + "-" * 95)
for r in results:
    print(f"  {r['label']:<24s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"${r['pnl']:>9,.0f} {r['avg_win']:+6.3f}R {r['avg_loss']:+6.3f}R "
          f"{r['noise_n']:5d} {r['noise_pct']:6.1f}% {r['noise_avg_bars']:9.1f}b")

print(f"\n  Exit reasons:")
for r in results:
    parts = [f"{k}:{v} ({v/r['trades']*100:.0f}%)" for k,v in r["exit_reasons"].items()]
    print(f"  {r['label']:<24s} {', '.join(parts)}")

print(f"\n  Filter stats:")
for r in results:
    if r["label"] == "BASELINE": continue
    s = r["stats"]; parts = []
    if s.get("blocked_hour", 0) > 0: parts.append(f"hours: {s['blocked_hour']}")
    if s.get("blocked_trend", 0) > 0: parts.append(f"trend: {s['blocked_trend']}")
    if s.get("blocked_cooldown", 0) > 0: parts.append(f"cooldown: {s['blocked_cooldown']}")
    print(f"  {r['label']:<24s} {', '.join(parts)}")

# ═══════════════════════════════════════════════════════════════════
# Noise MFE trajectory (with hour+trend active)
# ═══════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("NOISE MFE TRAJECTORY (with hour+trend filters active)")
print("=" * 70)
for bar in range(min(8, max(nd) if nd else 8)):
    mfes = [d[f"mfe_bar{bar}"] for d in noise_details if f"mfe_bar{bar}" in d]
    if mfes:
        print(f"  bar {bar}: avg MFE={np.mean(mfes):+.3f}R  median={np.median(mfes):+.3f}R  "
              f"%neg={sum(1 for m in mfes if m < 0)/len(mfes)*100:.0f}%")

# How many noise trades are negative by bar?
print(f"\n  % of noise trades with negative MFE at each bar:")
for bar in range(min(8, max(nd) if nd else 8)):
    mfes = [d[f"mfe_bar{bar}"] for d in noise_details if f"mfe_bar{bar}" in d]
    if mfes:
        pct_neg = sum(1 for m in mfes if m < 0) / len(mfes) * 100
        pct_neg_05 = sum(1 for m in mfes if m < -0.05) / len(mfes) * 100
        print(f"    bar {bar}: <0={pct_neg:.0f}%  <-0.05R={pct_neg_05:.0f}%  "
              f"n={len(mfes)}")
