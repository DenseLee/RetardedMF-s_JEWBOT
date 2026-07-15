"""Test hour filter + post-entry trajectory rule (bar-2 MFE vs bar-0 MFE)."""
import sys, os, json, numpy as np, pandas as pd, torch
from collections import Counter
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

# ── Filters ──
BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}
TRAJECTORY_CHECK_BAR = 2       # check at bar 2
TRAJECTORY_MIN_MFE = 0.20      # MFE must be below this AND declining to exit


def run_backtest(hour_filter=False, trajectory_rule=False, method_a=False, label="BASELINE"):
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
    entry_regime = ""; entry_conf = 0.0

    stats = {"blocked_hour": 0, "blocked_trend": 0, "trajectory_exits": 0}

    for i in range(max(config.seq_len_m15, 20), len(m15f)):
        ts = m15f["timestamp"].iloc[i]
        price = m15f["close"].iloc[i]
        executor._current_price = price

        today = ts.date()
        if ld and today != ld:
            pnl_d = 0.0; sb = bal
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
                # Method A: trend alignment
                if method_a:
                    h1_closes = h1s["close"].values
                    if len(h1_closes) >= 23:
                        h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                        h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                        with_trend = ((g.direction == 1 and h1_slope > 0) or
                                      (g.direction == -1 and h1_slope < 0))
                        if not with_trend:
                            stats["blocked_trend"] += 1
                            h1_sig = None; listen = False; continue

                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price
                entry_regime = rr["regime"]; entry_conf = g.confidence
            else:
                h1_sig = None; listen = False

        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None
            s2 = tm.state; sd2 = 1.0 * s2.entry_atr
            mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
            mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
            current_bar_idx = len(ab)
            ab.append({"bar": current_bar_idx, "mfe": float(mfe_now), "mae": float(mae_now),
                       "phase": s2.phase.name, "price": float(price)})

            # ── Trajectory rule: at bar 2, check if MFE is declining from bar 0 ──
            trajectory_exit = False
            if trajectory_rule and current_bar_idx == TRAJECTORY_CHECK_BAR:
                mfe_bar0 = ab[0]["mfe"] if len(ab) > 0 else 0
                mfe_bar2 = mfe_now
                # Exit if: MFE at bar 2 is LOWER than MFE at bar 0 (declining)
                # AND MFE at bar 2 is below the minimum threshold (not enough profit)
                if mfe_bar2 < mfe_bar0 and mfe_bar2 < TRAJECTORY_MIN_MFE:
                    epx = price
                    er = "trajectory"
                    trajectory_exit = True
                    stats["trajectory_exits"] += 1

            if not trajectory_exit:
                if tm.check_sl_hit(lo, hi):
                    epx = tm.exit_price_at_sl(); er = "sl_hit"
                elif tm.check_tp_hit(lo, hi):
                    epx = tm.exit_price_at_tp(); er = "tp_hit"
                else:
                    a = tm.update(price, hi, lo, h1_atr)
                    if a.action_type == TradeActionType.CLOSE:
                        epx = price; er = a.reason

            if epx:
                pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
                pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
                bal += pnl_dollar; pnl_d += pnl_dollar
                mfe_peak = max(b["mfe"] for b in ab) if ab else 0.0
                trades.append({
                    "pnl_dollar": round(pnl_dollar, 2), "pnl_r": round(pnl_r, 4),
                    "mfe_peak": round(mfe_peak, 4), "bars_held": len(ab),
                    "exit_reason": er,
                })
                pos = 0; tm.state = None; ab = []
            continue

        if not listen:
            continue

        bl += 1
        if bl > config.max_listen_bars:
            listen = False; h1_sig = None; continue

        # ── Hour filter ──
        if hour_filter and ts.hour in BLOCKED_HOURS:
            stats["blocked_hour"] += 1
            continue

        # M15 confirmation (NN + EMA rule — baseline logic)
        m15_feats = engine.compute(m15s)
        confirmed = False
        sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad():
            mo = m15_model(tt2)
        m15_conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
        m15_bias = mo["direction_bias"].item() if hasattr(mo["direction_bias"], "item") else float(mo["direction_bias"])

        if m15_conf >= config.min_entry_confidence:
            if (h1_sig == 1 and m15_bias > 0) or (h1_sig == -1 and m15_bias < 0):
                confirmed = True
        if not confirmed:
            mc2 = m15s["close"].values
            ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]:
                confirmed = True
            elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]:
                confirmed = True

        if not confirmed:
            continue

        if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss:
            continue

        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots)
        executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    wins = [t for t in trades if t["pnl_r"] > 0]
    losses = [t for t in trades if t["pnl_r"] <= 0]
    n = len(trades)
    wr = len(wins)/n*100 if n else 0
    tg = sum(t["pnl_r"] for t in wins); tl = abs(sum(t["pnl_r"] for t in losses))
    pf = tg/max(tl,0.001)
    total_pnl = sum(t["pnl_dollar"] for t in trades)

    # Micro-wins vs death-zone counts
    micro = len([t for t in wins if t["pnl_r"] <= 0.25])
    death = len([t for t in losses if t["mfe_peak"] >= 0.25])
    noise = len([t for t in losses if t["mfe_peak"] <= 0.25])
    good = len([t for t in wins if t["pnl_r"] > 0.50])

    return {
        "label": label, "trades": n, "wins": len(wins), "losses": len(losses),
        "wr": wr, "pf": pf, "pnl": total_pnl,
        "avg_win": np.mean([t["pnl_r"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_r"] for t in losses]) if losses else 0,
        "avg_mfe_win": np.mean([t["mfe_peak"] for t in wins]) if wins else 0,
        "avg_mfe_loss": np.mean([t["mfe_peak"] for t in losses]) if losses else 0,
        "exit_reasons": dict(Counter(t["exit_reason"] for t in trades)),
        "expectancy": np.mean([t["pnl_r"] for t in trades]) if trades else 0,
        "micro": micro, "death": death, "noise": noise, "good": good,
        "stats": stats,
    }


# ═══════════════════════════════════════════════════════════════════
results = []

print("BASELINE...")
results.append(run_backtest(label="BASELINE"))

print("Hour filter only...")
results.append(run_backtest(hour_filter=True, label="HOUR_FILTER"))

print("Trajectory rule only...")
results.append(run_backtest(trajectory_rule=True, label="TRAJECTORY"))

print("Hour + Trajectory...")
results.append(run_backtest(hour_filter=True, trajectory_rule=True, label="HOUR+TRAJ"))

print("Hour + Trajectory + Trend (full)...")
results.append(run_backtest(hour_filter=True, trajectory_rule=True, method_a=True, label="FULL (H+T+A)"))

# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("RESULTS — Hour Filter + Trajectory Rule")
print("=" * 100)
HDR = f"  {'Method':<22s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Expect':>8s} {'Micro':>6s} {'Death':>6s} {'Noise':>6s} {'Good':>6s}"
print(HDR)
print("  " + "-" * 98)
base = results[0]
for r in results:
    delta_pf = r["pf"] - base["pf"]
    delta_pnl = r["pnl"] - base["pnl"]
    print(f"  {r['label']:<22s} {r['trades']:7d} {r['wr']:6.1f}% {r['pf']:6.2f} "
          f"${r['pnl']:>9,.0f} {r['avg_win']:+7.3f}R {r['avg_loss']:+7.3f}R {r['expectancy']:+7.4f}R "
          f"{r['micro']:6d} {r['death']:6d} {r['noise']:6d} {r['good']:6d}")

print(f"\n  Exit reasons:")
for r in results:
    parts = [f"{k}:{v} ({v/r['trades']*100:.0f}%)" for k,v in r["exit_reasons"].items()]
    print(f"  {r['label']:<22s} {', '.join(parts)}")

print(f"\n  Filter stats:")
for r in results[1:]:
    s = r["stats"]
    parts = []
    if s.get("blocked_hour", 0) > 0: parts.append(f"hours blocked: {s['blocked_hour']}")
    if s.get("blocked_trend", 0) > 0: parts.append(f"trend blocked: {s['blocked_trend']}")
    if s.get("trajectory_exits", 0) > 0: parts.append(f"trajectory exits: {s['trajectory_exits']} ({s['trajectory_exits']/r['trades']*100:.0f}% of trades)")
    print(f"  {r['label']:<22s} {', '.join(parts)}")

# ═══════════════════════════════════════════════════════════════════
# Deep dive: trajectory exits — what did we cut?
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TRAJECTORY RULE DEEP DIVE")
print("=" * 70)

# Run a single backtest with trajectory rule to capture per-trade details
r = run_backtest(trajectory_rule=True, hour_filter=False, method_a=False, label="traj_detail")

# We need per-bar data to analyze trajectory exits. Run a detailed pass.
print("(Re-running with detailed per-bar capture for trajectory analysis...)")

# Quick detailed run
tm = TradeManager(initial_sl=config.initial_sl, hard_tp=config.hard_tp,
                  breakeven_trigger=config.breakeven_trigger,
                  trail_trigger=config.trail_trigger,
                  trail_dist=config.trail_dist, trail_dist_s=config.trail_dist_s,
                  regime_tighten=config.regime_tighten,
                  max_hold=config.max_hold_bars, mae_guard_retrace=config.mae_guard_retrace)
executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)

bal = 10000.0; pnl_d = 0.0; ld = None; trades_d = []; sb = 10000.0
h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []
entry_regime = ""; entry_conf = 0.0
traj_exits = []; traj_saved = []

for i in range(max(config.seq_len_m15, 20), len(m15f)):
    ts = m15f["timestamp"].iloc[i]
    price = m15f["close"].iloc[i]
    executor._current_price = price
    today = ts.date()
    if ld and today != ld: pnl_d = 0.0; sb = bal
    ld = today
    h1s = h1f[h1f["timestamp"] <= ts]
    m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
    if len(h1s) < config.seq_len_h1: continue

    hl = h1s["timestamp"].max()
    if hl != lh:
        lh = hl
        h1_feats = engine.compute(h1s)
        seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
        t = torch.from_numpy(seq).unsqueeze(0).to(device)
        for _, row in h1s.iloc[-14:].iterrows(): rd.update(row["high"], row["low"], row["close"])
        rr = classify_regime(encoder, classifier, t, rd, model_confidence_threshold=config.min_regime_confidence)
        g = gate.evaluate(rr["regime"], rr["confidence"], rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])
        if g.entry_signal: h1_sig = g.direction; listen = True; bl = 0; h1_atr = h1_feats[-1, 6] * price; entry_regime = rr["regime"]; entry_conf = g.confidence
        else: h1_sig = None; listen = False

    if pos != 0 and tm.state is not None:
        hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
        epx = None; er = None
        s2 = tm.state; sd2 = 1.0 * s2.entry_atr
        mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
        mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
        current_bar_idx = len(ab)
        ab.append({"bar": current_bar_idx, "mfe": float(mfe_now), "mae": float(mae_now), "phase": s2.phase.name, "price": float(price)})

        traj_exit = False
        if current_bar_idx == TRAJECTORY_CHECK_BAR:
            mfe_bar0 = ab[0]["mfe"]
            mfe_bar2 = mfe_now
            if mfe_bar2 < mfe_bar0 and mfe_bar2 < TRAJECTORY_MIN_MFE:
                # This trade WOULD be exited by trajectory rule.
                # Let's also track what WOULD have happened if we didn't exit
                traj_exit = True

        if traj_exit:
            # Don't actually exit — let it run to see the counterfactual
            pass

        if tm.check_sl_hit(lo, hi):
            epx = tm.exit_price_at_sl(); er = "sl_hit"
        elif tm.check_tp_hit(lo, hi):
            epx = tm.exit_price_at_tp(); er = "tp_hit"
        else:
            a = tm.update(price, hi, lo, h1_atr)
            if a.action_type == TradeActionType.CLOSE:
                epx = price; er = a.reason

        if epx:
            pnl_r = (epx - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - epx) / sd2
            pnl_dollar = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
            bal += pnl_dollar; pnl_d += pnl_dollar
            mfe_peak = max(b["mfe"] for b in ab) if ab else 0.0
            entry_data = {"pnl_r": round(pnl_r, 4), "mfe_peak": round(mfe_peak, 4),
                          "bars_held": len(ab), "exit_reason": er,
                          "mfe_bar0": ab[0]["mfe"] if ab else 0,
                          "mfe_bar2": ab[min(2, len(ab)-1)]["mfe"] if len(ab) > 2 else 0}
            if traj_exit:
                entry_data["pnl_if_exited"] = round(mfe_now, 4)  # approximate
                traj_exits.append(entry_data)
            elif current_bar_idx >= TRAJECTORY_CHECK_BAR and ab[0]["mfe"] < TRAJECTORY_MIN_MFE:
                # Trade that qualified for check but passed (MFE rising)
                traj_saved.append(entry_data)

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
        mc2 = m15s["close"].values
        ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
        if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]: confirmed = True
        elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]: confirmed = True
    if not confirmed: continue
    if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss: continue
    listen = False
    lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
    tm.enter(h1_sig, price, h1_atr, lots)
    executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
    pos = h1_sig

if traj_exits:
    avg_actual = np.mean([t["pnl_r"] for t in traj_exits])
    avg_if_exited = np.mean([t["pnl_if_exited"] for t in traj_exits])
    wins_if_held = sum(1 for t in traj_exits if t["pnl_r"] > 0)
    print(f"\n  Trajectory rule would have exited {len(traj_exits)} trades at bar 2")
    print(f"  Avg actual outcome (if held):  {avg_actual:+.3f}R")
    print(f"  Avg MFE at bar 2 (if exited):  {avg_if_exited:+.3f}R")
    print(f"  Would have been wins if held:   {wins_if_held}/{len(traj_exits)} ({wins_if_held/len(traj_exits)*100:.0f}%)")
    print(f"  Exiting saves:                  {len(traj_exits) - wins_if_held} losses prevented")
    print(f"  Exiting costs:                  {wins_if_held} wins lost")
    print(f"  Net PnL impact:                 R saved = {(avg_if_exited - avg_actual):+.3f}R per trade")
    # Bar-0 vs Bar-2 MFE comparison
    print(f"  Avg MFE bar 0:  {np.mean([t['mfe_bar0'] for t in traj_exits]):+.3f}R")
    print(f"  Avg MFE bar 2:  {np.mean([t['mfe_bar2'] for t in traj_exits]):+.3f}R")
    print(f"  Exit reasons if held: {dict(Counter(t['exit_reason'] for t in traj_exits))}")

if traj_saved:
    print(f"\n  Trades that PASSED trajectory check (MFE rising at bar 2): {len(traj_saved)}")
    print(f"  Avg outcome:     {np.mean([t['pnl_r'] for t in traj_saved]):+.3f}R")
    print(f"  Avg MFE peak:    {np.mean([t['mfe_peak'] for t in traj_saved]):+.3f}R")
    wins_saved = sum(1 for t in traj_saved if t["pnl_r"] > 0)
    print(f"  Win rate:        {wins_saved/len(traj_saved)*100:.0f}%")
    print(f"  Avg MFE bar 0:   {np.mean([t['mfe_bar0'] for t in traj_saved]):+.3f}R")
    print(f"  Avg MFE bar 2:   {np.mean([t['mfe_bar2'] for t in traj_saved]):+.3f}R")
    print(f"  (These are the trades the rule CORRECTLY leaves alone)")
