"""Parallel grid search: trade manager params optimized for v2 M15 model."""
import sys, os, numpy as np, pandas as pd, torch, multiprocessing as mp
from itertools import product
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only for parallel workers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config = BTCConfig()
device = torch.device("cpu")
BLOCKED_HOURS = {2, 11, 18, 19, 21, 22, 23}

# ── Load models once globally for each worker ──
_encoder = None; _classifier = None; _m15_v2 = None; _engine = None; _gate = None
_h1f = None; _m15f = None


def init_worker():
    global _encoder, _classifier, _m15_v2, _engine, _gate, _h1f, _m15f
    _encoder = CNNLSTMEncoder(n_features=config.n_features, seq_len=config.seq_len_h1,
        cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
        embedding_dim=config.embedding_dim, regime_classes=config.regime_classes,
        bidirectional=True).to(device).eval()
    _classifier = RegimeClassifier(embedding_dim=config.embedding_dim,
                                   n_classes=config.regime_classes).to(device).eval()
    ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder.pt"),
                      map_location=device, weights_only=False)
    _encoder.load_state_dict(ckpt["encoder_state_dict"])
    _classifier.load_state_dict(ckpt["classifier_state_dict"])
    _m15_v2 = CNNGRUM15(n_features=config.n_features, seq_len=config.seq_len_m15,
        cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
    mc2 = torch.load(os.path.join(config.model_dir, "btc_m15_v2.pt"),
                     map_location=device, weights_only=False)
    _m15_v2.load_state_dict(mc2["model_state_dict"], strict=False)
    _engine = BTCFeatureEngine()
    _gate = EntryGate()
    _h1f = pd.read_csv(os.path.join(config.data_dir,
        "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv"))
    _h1f["timestamp"] = pd.to_datetime(_h1f["timestamp"], utc=True)
    _m15f = pd.read_csv(os.path.join(config.data_dir,
        "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv"))
    _m15f["timestamp"] = pd.to_datetime(_m15f["timestamp"], utc=True)
    ft = pd.Timestamp("2026-01-01", tz="UTC"); et = pd.Timestamp("2026-05-06", tz="UTC")
    _h1f = _h1f[(_h1f["timestamp"] >= ft) & (_h1f["timestamp"] < et)].reset_index(drop=True)
    _m15f = _m15f[(_m15f["timestamp"] >= ft) & (_m15f["timestamp"] < et)].reset_index(drop=True)


def run_one(params):
    """Run a single backtest with given trade manager params. Returns (params, metrics)."""
    be, tt, td, mh, sl = params
    tm = TradeManager(initial_sl=sl, hard_tp=3.0, breakeven_trigger=be,
                      trail_trigger=tt, trail_dist=td, trail_dist_s=td * 0.67,
                      regime_tighten=0.40, max_hold=mh, mae_guard_retrace=2.5)
    executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; sb = 10000.0
    h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
    lh = None; h1_atr = 0.0; lots = 0.0; pos = 0; ab = []
    entry_regime = ""; entry_conf = 0.0

    n_m15 = len(_m15f)
    for i in range(max(config.seq_len_m15, 20), n_m15):
        ts = _m15f["timestamp"].iloc[i]; price = _m15f["close"].iloc[i]
        executor._current_price = price
        today = ts.date()
        if ld and today != ld: pnl_d = 0.0; sb = bal
        ld = today
        h1s = _h1f[_h1f["timestamp"] <= ts]
        m15s = _m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
        if len(h1s) < config.seq_len_h1: continue

        hl = h1s["timestamp"].max()
        if hl != lh:
            lh = hl; h1_feats = _engine.compute(h1s)
            seq = _engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0).to(device)
            for _, row in h1s.iloc[-14:].iterrows(): rd.update(row["high"], row["low"], row["close"])
            rr = classify_regime(_encoder, _classifier, t, rd,
                                 model_confidence_threshold=config.min_regime_confidence)
            g = _gate.evaluate(rr["regime"], rr["confidence"],
                              rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])
            if g.entry_signal:
                h1_closes = h1s["close"].values
                if len(h1_closes) >= 23:
                    h1_ema22 = pd.Series(h1_closes).ewm(span=22, adjust=False).mean().values
                    h1_slope = (h1_ema22[-1] - h1_ema22[-2]) / max(abs(float(h1_ema22[-2])), 1e-12)
                    with_trend = ((g.direction == 1 and h1_slope > 0) or
                                  (g.direction == -1 and h1_slope < 0))
                    if not with_trend: h1_sig = None; listen = False; continue
                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price
                entry_regime = rr["regime"]; entry_conf = g.confidence
            else:
                h1_sig = None; listen = False

        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None; s2 = tm.state; sd2 = 1.0 * s2.entry_atr
            ab.append({"bar": len(ab), "mfe": float((hi - s2.entry_price) / sd2 if pos == 1 else (s2.entry_price - lo) / sd2)})
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
                trades.append({"pnl_r": pnl_r, "pnl_dollar": pnl_dollar, "mfe_peak": mfe_peak,
                               "bars_held": len(ab), "exit_reason": er})
                pos = 0; tm.state = None; ab = []
            continue

        if not listen: continue
        bl += 1
        if bl > config.max_listen_bars: listen = False; h1_sig = None; continue
        if ts.hour in BLOCKED_HOURS: continue

        m15_feats = _engine.compute(m15s); confirmed = False
        sm = _engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt2 = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad(): mo = _m15_v2(tt2)
        conf = mo["entry_confidence"].item() if hasattr(mo["entry_confidence"], "item") else float(mo["entry_confidence"])
        if conf >= 0.5: confirmed = True

        if not confirmed:
            mc2 = m15s["close"].values; ema21 = pd.Series(mc2).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc2[-1] <= ema21[-1] * 1.01 and mc2[-1] > mc2[-2]: confirmed = True
            elif h1_sig == -1 and mc2[-1] >= ema21[-1] * 0.99 and mc2[-1] < mc2[-2]: confirmed = True

        if not confirmed: continue
        if abs(pnl_d) / max(sb, 1) >= config.max_daily_loss: continue

        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, 0.02, tm.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots)
        executor.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    wins = [t for t in trades if t["pnl_r"] > 0]; losses = [t for t in trades if t["pnl_r"] <= 0]
    n = len(trades); wr = len(wins) / n * 100 if n else 0
    tg = sum(t["pnl_r"] for t in wins); tl = abs(sum(t["pnl_r"] for t in losses))
    pf = tg / max(tl, 0.001); total_pnl = sum(t["pnl_dollar"] for t in trades)
    dd = 0.0  # simplified — no per-bar balance tracking in grid search
    tp_pct = sum(1 for t in trades if t["exit_reason"] == "tp_hit") / n * 100 if n else 0
    avg_win = np.mean([t["pnl_r"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_r"] for t in losses]) if losses else 0
    score = total_pnl / max(abs(dd), 1)

    return {"be": be, "tt": tt, "td": td, "mh": mh, "sl": sl,
            "trades": n, "wr": wr, "pf": pf, "pnl": total_pnl,
            "avg_win": avg_win, "avg_loss": avg_loss, "tp_pct": tp_pct,
            "score": score}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    be_vals = [0.50, 0.75, 1.00]
    tt_vals = [2.0, 2.5, 3.0]
    td_vals = [0.75, 1.0, 1.25]
    mh_vals = [18, 21, 24]
    sl_vals = [1.0, 1.25]

    combos = list(product(be_vals, tt_vals, td_vals, mh_vals, sl_vals))
    print(f"Grid search: {len(combos)} combos (BE×TT×TD×MH×SL)")
    print(f"BE={be_vals}, TT={tt_vals}, TD={td_vals}, MH={mh_vals}, SL={sl_vals}")

    n_workers = min(4, mp.cpu_count())
    print(f"Workers: {n_workers}")

    with mp.Pool(n_workers, initializer=init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_one, combos)):
            results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(combos)}...")

    results.sort(key=lambda r: r["pnl"], reverse=True)

    print(f"\n{'='*100}")
    print("TOP 15 CONFIGS (by PnL)")
    print("=" * 100)
    print(f"  {'BE':>5s} {'TT':>5s} {'TD':>5s} {'MH':>5s} {'SL':>5s} {'Trds':>6s} {'WR':>7s} {'PF':>7s} {'PnL':>10s} {'AvgW':>8s} {'AvgL':>8s} {'TP%':>6s}")
    print("  " + "-" * 90)
    for r in results[:15]:
        print(f"  {r['be']:5.2f} {r['tt']:5.1f} {r['td']:5.2f} {r['mh']:5d} {r['sl']:5.2f} "
              f"{r['trades']:6d} {r['wr']:6.1f}% {r['pf']:6.2f} ${r['pnl']:>9,.0f} "
              f"{r['avg_win']:+7.3f}R {r['avg_loss']:+7.3f}R {r['tp_pct']:5.1f}%")

    best = results[0]
    print(f"\n  BEST: BE={best['be']:.2f} TT={best['tt']:.1f} TD={best['td']:.2f} "
          f"MH={best['mh']} SL={best['sl']:.2f}")
    print(f"         {best['trades']} trades, WR={best['wr']:.1f}%, PF={best['pf']:.2f}, "
          f"PnL=${best['pnl']:,.0f}")

    # Save all results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.log_dir, "grid_search_v2_results.csv"), index=False)
    print(f"\nAll results saved to logs/grid_search_v2_results.csv")
