"""Test PF improvement methods from trade report analysis. Runs baseline + 4 variants."""
import sys, os, numpy as np, pandas as pd, torch
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


def run_backtest(method_a=False, method_b=False, method_c=False, early_mae_threshold=-0.55,
                 label="BASELINE"):
    """Run replay loop with optional filters. Returns dict of metrics."""
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

    stats = {"blocked_trend": 0, "blocked_bars": 0, "early_mae_exits": 0}

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

            # Method A: block against-H1-trend entries
            if g.entry_signal and method_a:
                h1_closes = h1s["close"].values
                if len(h1_closes) >= 23:
                    h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                    h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                else:
                    h1_slope = 0.0
                with_trend = (g.direction == 1 and h1_slope > 0) or (g.direction == -1 and h1_slope < 0)
                if not with_trend:
                    stats["blocked_trend"] += 1
                    h1_sig = None; listen = False
                else:
                    h1_sig = g.direction; listen = True; bl = 0
                    h1_atr = h1_feats[-1, 6] * price
                    entry_regime = rr["regime"]; entry_conf = g.confidence
            elif g.entry_signal:
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
            ab.append({"bar": len(ab), "mfe": float(mfe_now), "mae": float(mae_now),
                       "phase": s2.phase.name, "price": float(price)})

            # Method C: early MAE exit
            if method_c and len(ab) <= 3 and mae_now < early_mae_threshold:
                epx = price; er = "early_mae"
                stats["early_mae_exits"] += 1
            elif tm.check_sl_hit(lo, hi):
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
                    "regime": entry_regime if pos == h1_sig else "unknown",
                })
                pos = 0; tm.state = None; ab = []
            continue

        if not listen:
            continue

        bl += 1
        if bl > config.max_listen_bars:
            listen = False; h1_sig = None; continue

        # M15 confirmation (NN model + EMA fallback — BASELINE logic)
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

        # Method B: require bars_listened >= 3 for confirmation
        if method_b and bl < 3:
            if confirmed:
                stats["blocked_bars"] += 1
            continue

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
    wr = len(wins) / n * 100 if n else 0
    tg = sum(t["pnl_r"] for t in wins)
    tl = abs(sum(t["pnl_r"] for t in losses))
    pf = tg / max(tl, 0.001)
    total_pnl = sum(t["pnl_dollar"] for t in trades)

    return {
        "label": label,
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr": wr, "pf": pf, "pnl": total_pnl,
        "avg_win": np.mean([t["pnl_r"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_r"] for t in losses]) if losses else 0,
        "avg_mfe_win": np.mean([t["mfe_peak"] for t in wins]) if wins else 0,
        "avg_mfe_loss": np.mean([t["mfe_peak"] for t in losses]) if losses else 0,
        "exit_reasons": dict(Counter(t["exit_reason"] for t in trades)),
        "expectancy": np.mean([t["pnl_r"] for t in trades]) if trades else 0,
        "stats": stats,
    }


# ═══════════════════════════════════════════════════════════════════
# RUN ALL VARIANTS
# ═══════════════════════════════════════════════════════════════════
results = []

print("Running BASELINE...")
results.append(run_backtest(label="BASELINE"))

print("Running Method A (block against-H1-trend)...")
results.append(run_backtest(method_a=True, label="A: Block against-trend"))

print("Running Method B (bars_listened >= 3)...")
results.append(run_backtest(method_b=True, label="B: Min 3-bar listen"))

print("Running Method C (early MAE -0.55R)...")
results.append(run_backtest(method_c=True, early_mae_threshold=-0.55, label="C: Early MAE -0.55R"))

print("Running Method D (A + B + C combined)...")
results.append(run_backtest(method_a=True, method_b=True, method_c=True,
                            early_mae_threshold=-0.55, label="D: A+B+C combined"))

# ═══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("PF IMPROVEMENT METHODS — COMPARISON")
print("=" * 100)

HDR = f"  {'Method':<28s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Expect':>8s}"
print(HDR)
print("  " + "-" * 96)

base = results[0]
for r in results:
    delta_pf = r["pf"] - base["pf"]
    delta_pnl = r["pnl"] - base["pnl"]
    print(f"  {r['label']:<28s} {r['trades']:7d} {r['wr']:6.1f}% {r['pf']:6.2f} "
          f"${r['pnl']:>9,.0f} {r['avg_win']:+7.3f}R {r['avg_loss']:+7.3f}R {r['expectancy']:+7.4f}R "
          f"({'PF ' + ('+' if delta_pf > 0 else '') + f'{delta_pf:.2f}'} | "
          f"{'PnL ' + ('+' if delta_pnl > 0 else '') + f'${delta_pnl:,.0f}'})")

print(f"\n  Exit reason breakdown:")
for r in results:
    parts = []
    for reason in ["sl_hit", "Time stop", "tp_hit", "early_mae"]:
        if reason in r["exit_reasons"]:
            pct = r["exit_reasons"][reason] / r["trades"] * 100
            parts.append(f"{reason}: {r['exit_reasons'][reason]} ({pct:.0f}%)")
    print(f"  {r['label']:<28s} {', '.join(parts)}")

print(f"\n  Win size distribution (0-0.25R / 0.25-0.50R / 0.50-1R / 1-2R / 2-3R / full TP):")
for r in results:
    all_trades = []
    # We need to recompute win sizes — store them in results
    pass

print(f"\n  Filter stats:")
for r in results[1:]:
    s = r["stats"]
    stat_parts = []
    if "blocked_trend" in s and s["blocked_trend"] > 0:
        stat_parts.append(f"trend-blocked: {s['blocked_trend']}")
    if "blocked_bars" in s and s["blocked_bars"] > 0:
        stat_parts.append(f"bars-blocked: {s['blocked_bars']}")
    if "early_mae_exits" in s and s["early_mae_exits"] > 0:
        stat_parts.append(f"early_mae: {s['early_mae_exits']} ({s['early_mae_exits']/r['trades']*100:.0f}% of trades)")
    if stat_parts:
        print(f"  {r['label']:<28s} {', '.join(stat_parts)}")
