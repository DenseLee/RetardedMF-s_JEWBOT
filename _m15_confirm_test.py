"""Phase 1: Enhanced rule-based M15 confirmation — volume, momentum, regime-conditional EMA."""
import sys, os, numpy as np, pandas as pd, torch
from collections import Counter
sys.path.insert(0, ".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from models.cnn_gru_m15 import CNNGRUM15
from execution.mt5_executor_btc import DryRunExecutor

config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = CNNLSTMEncoder(n_features=config.n_features, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
    bidirectional=True).to(device).eval()
classifier = RegimeClassifier(embedding_dim=config.embedding_dim, n_classes=config.regime_classes).to(device).eval()
ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder.pt"), map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"]); classifier.load_state_dict(ckpt["classifier_state_dict"])

engine = BTCFeatureEngine(); gate = EntryGate()

# Load M15 NN model for BASELINE comparison only
m15_model = CNNGRUM15(n_features=config.n_features, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
mc = torch.load(os.path.join(config.model_dir, "btc_m15_model.pt"), map_location=device, weights_only=False)
m15_model.load_state_dict(mc["model_state_dict"])

h1f = pd.read_csv(os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv"))
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)
ft = pd.Timestamp("2026-01-01", tz="UTC"); et = pd.Timestamp("2026-05-06", tz="UTC")
h1f = h1f[(h1f["timestamp"] >= ft) & (h1f["timestamp"] < et)].reset_index(drop=True)
m15f = m15f[(m15f["timestamp"] >= ft) & (m15f["timestamp"] < et)].reset_index(drop=True)

BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}


def m15_confirm_enhanced(m15_df, h1_signal, regime, use_volume=False, use_momentum=False,
                         use_regime_ema=False):
    """Enhanced rule-based M15 confirmation. No NN model."""
    closes = m15_df["close"].values
    ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().values

    # ── Base check: EMA proximity + 2-bar direction (original fallback) ──
    base_confirmed = False
    if h1_signal == 1:
        near_ema = closes[-1] <= ema21[-1] * 1.01
        turning_up = closes[-1] > closes[-2]
        base_confirmed = near_ema and turning_up
    else:
        near_ema = closes[-1] >= ema21[-1] * 0.99
        turning_down = closes[-1] < closes[-2]
        base_confirmed = near_ema and turning_down

    if not base_confirmed:
        return False

    # ── A) Volume confirmation ──
    if use_volume:
        vol = m15_df["volume"].values
        vol_sma20 = pd.Series(vol).rolling(20).mean().values
        if len(vol_sma20) >= 1 and vol_sma20[-1] > 0:
            vol_ratio = vol[-1] / vol_sma20[-1]
            if vol_ratio < 1.0:  # below-average volume → wick, skip
                return False

    # ── B) Strengthened momentum: 2 of last 3 bars close in signal direction ──
    if use_momentum:
        if len(closes) < 4:
            return False
        bars_ok = 0
        for j in range(1, 4):  # check bars -1, -2, -3 vs their previous
            if h1_signal == 1 and closes[-j] > closes[-(j+1)]:
                bars_ok += 1
            elif h1_signal == -1 and closes[-j] < closes[-(j+1)]:
                bars_ok += 1
        if bars_ok < 2:
            return False

    # ── C) Regime-conditional EMA ──
    if use_regime_ema:
        if regime in ("TREND_UP",):
            # Long in uptrend: must be above EMA21 (not just "near")
            if h1_signal == 1 and closes[-1] <= ema21[-1]:
                return False
            # Short in uptrend: counter-trend, use normal rule
        elif regime in ("TREND_DOWN",):
            # Short in downtrend: must be below EMA21
            if h1_signal == -1 and closes[-1] >= ema21[-1]:
                return False
        # RANGE/TRANSITION: keep base rule as-is

    return True


def run_backtest(use_volume=False, use_momentum=False, use_regime_ema=False, use_nn=False, label=""):
    tm = TradeManager(initial_sl=config.initial_sl, hard_tp=config.hard_tp,
                      breakeven_trigger=config.breakeven_trigger,
                      trail_trigger=config.trail_trigger, trail_dist=config.trail_dist,
                      trail_dist_s=config.trail_dist_s, regime_tighten=config.regime_tighten,
                      max_hold=config.max_hold_bars, mae_guard_retrace=config.mae_guard_retrace)
    executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
    h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
    lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []
    entry_regime = ""; entry_conf = 0.0
    confirm_counts = Counter()

    for i in range(max(config.seq_len_m15, 20), len(m15f)):
        ts = m15f["timestamp"].iloc[i]; price = m15f["close"].iloc[i]
        executor._current_price = price
        today = ts.date()
        if ld and today != ld: pnl_d = 0.0; sb = bal
        ld = today
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
                h1_closes = h1s["close"].values
                if len(h1_closes) >= 23:
                    h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                    h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                    with_trend = ((g.direction == 1 and h1_slope > 0) or (g.direction == -1 and h1_slope < 0))
                    if not with_trend: h1_sig = None; listen = False; continue
                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price
                entry_regime = rr["regime"]; entry_conf = g.confidence
            else:
                h1_sig = None; listen = False

        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None; s2 = tm.state; sd2 = 1.0 * s2.entry_atr
            mfe_now = (hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2
            mae_now = (lo - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - hi) / sd2
            ab.append({"bar": len(ab), "mfe": float(mfe_now), "mae": float(mae_now)})
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
                trades.append({"pnl_r": round(pnl_r, 4), "pnl_dollar": round(pnl_dollar, 2),
                               "mfe_peak": round(mfe_peak, 4), "bars_held": len(ab), "exit_reason": er})
                pos = 0; tm.state = None; ab = []
            continue

        if not listen: continue
        bl += 1
        if bl > config.max_listen_bars: listen = False; h1_sig = None; continue
        if ts.hour in BLOCKED_HOURS: continue

        # ── M15 confirmation ──
        confirmed = False
        m15_feats = engine.compute(m15s)

        # Stage 1: NN model (if enabled — BASELINE only)
        if use_nn:
            sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
            tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
            with torch.no_grad():
                mo = m15_model(tt2)
            m15_conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
            m15_bias = mo["direction_bias"].item() if hasattr(mo["direction_bias"], "item") else float(mo["direction_bias"])
            if m15_conf >= config.min_entry_confidence:
                if (h1_sig == 1 and m15_bias > 0) or (h1_sig == -1 and m15_bias < 0):
                    confirmed = True

        # Stage 2: Enhanced rule-based confirmation
        if not confirmed:
            confirmed = m15_confirm_enhanced(m15s, h1_sig, entry_regime,
                                             use_volume=use_volume,
                                             use_momentum=use_momentum,
                                             use_regime_ema=use_regime_ema)
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
    noise_n = len([t for t in losses if t["mfe_peak"] <= 0.25])
    micro_n = len([t for t in wins if t["pnl_r"] <= 0.25])
    good_n = len([t for t in wins if t["pnl_r"] > 0.50])

    return {"label": label, "trades": n, "wins": len(wins), "losses": len(losses),
            "wr": wr, "pf": pf, "pnl": total_pnl,
            "avg_win": np.mean([t["pnl_r"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl_r"] for t in losses]) if losses else 0,
            "noise_n": noise_n, "noise_pct": noise_n/n*100 if n else 0,
            "micro_n": micro_n, "good_n": good_n,
            "exit_reasons": dict(Counter(t["exit_reason"] for t in trades))}


# ═══════════════════════════════════════════════════════════════════
print("BASELINE (HOUR+TREND + NN+EMA)...")
results = [run_backtest(use_nn=True, label="BASELINE (NN+EMA)")]

print("EMA-only (no NN, original EMA rule)...")
results.append(run_backtest(label="EMA_ONLY"))

print("EMA + VOLUME...")
results.append(run_backtest(use_volume=True, label="EMA+VOL"))

print("EMA + MOMENTUM (2-of-3)...")
results.append(run_backtest(use_momentum=True, label="EMA+MOM"))

print("EMA + REGIME_EMA...")
results.append(run_backtest(use_regime_ema=True, label="EMA+REG"))

print("EMA + VOL + MOM...")
results.append(run_backtest(use_volume=True, use_momentum=True, label="EMA+VOL+MOM"))

print("EMA + VOL + REG...")
results.append(run_backtest(use_volume=True, use_regime_ema=True, label="EMA+VOL+REG"))

print("EMA + MOM + REG...")
results.append(run_backtest(use_momentum=True, use_regime_ema=True, label="EMA+MOM+REG"))

print("ALL 3 (EMA + VOL + MOM + REG)...")
results.append(run_backtest(use_volume=True, use_momentum=True, use_regime_ema=True, label="ALL_3"))

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("ENHANCED M15 CONFIRMATION RESULTS")
print("=" * 100)
HDR = f"  {'Method':<22s} {'Trds':>5s} {'WR':>6s} {'PF':>6s} {'PnL':>10s} {'AvgW':>7s} {'AvgL':>7s} {'Noise':>6s} {'Noise%':>7s} {'Micro':>6s} {'Good':>6s}"
print(HDR)
print("  " + "-" * 96)
for r in results:
    print(f"  {r['label']:<22s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"${r['pnl']:>9,.0f} {r['avg_win']:+6.3f}R {r['avg_loss']:+6.3f}R "
          f"{r['noise_n']:5d} {r['noise_pct']:6.1f}% {r['micro_n']:5d} {r['good_n']:5d}")

print(f"\n  Exit reasons:")
for r in results:
    parts = [f"{k}:{v}" for k,v in r["exit_reasons"].items()]
    print(f"  {r['label']:<22s} {', '.join(parts)}")

# Highlight best
best = max(results, key=lambda r: r["pnl"])
print(f"\n  BEST: {best['label']} — ${best['pnl']:,.0f} at PF {best['pf']:.2f}")
