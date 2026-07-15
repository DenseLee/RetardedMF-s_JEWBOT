"""
Parallel grid search over trade manager parameters.
Uses multiprocessing to run backtests concurrently.
Each worker loads its own model copy.
"""
import sys, os, time, itertools, json
import numpy as np, pandas as pd, torch
import multiprocessing as mp
from functools import partial
from multiprocessing import cpu_count

# Force CPU in workers to avoid CUDA OOM with multiple processes
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Grid ──
PARAM_GRID = {
    "breakeven_trigger": [0.5, 0.75, 1.0],
    "trail_trigger":     [2.0, 2.5],
    "trail_dist":        [0.75, 1.0, 1.5],
    "max_hold":          [18, 24],
}
ALL_COMBOS = list(itertools.product(
    PARAM_GRID["breakeven_trigger"],
    PARAM_GRID["trail_trigger"],
    PARAM_GRID["trail_dist"],
    PARAM_GRID["max_hold"],
))
print(f"Total combos: {len(ALL_COMBOS)}")
print(f"Workers: {min(cpu_count(), 4)}")

# ── Pre-load raw data (shared across workers) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_btc import BTCConfig
cfg = BTCConfig()
h1f = pd.read_csv(cfg.data_dir + "/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(cfg.data_dir + "/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)

# Filter to 2026 YTD
from_ts = pd.Timestamp("2026-01-01", tz="UTC")
h1f = h1f[h1f["timestamp"] >= from_ts].reset_index(drop=True)
m15f = m15f[m15f["timestamp"] >= from_ts].reset_index(drop=True)


def run_one(params_tuple):
    """Run a single backtest. Called by worker process."""
    be, tt, td, mh = params_tuple
    params = {"breakeven_trigger": be, "trail_trigger": tt, "trail_dist": td, "max_hold": mh}
    label = f"BE={be:.2f}_TT={tt:.1f}_TD={td:.2f}_MH={mh}"

    # Imports inside worker (fresh process)
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
    device = torch.device("cpu")  # CPU per worker to avoid CUDA contention

    # Load models
    encoder = CNNLSTMEncoder(n_features=17, seq_len=config.seq_len_h1,
        cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
        embedding_dim=config.embedding_dim, regime_classes=4,
        bidirectional=True).to(device).eval()
    classifier = RegimeClassifier(embedding_dim=128, n_classes=4).to(device).eval()
    ckpt = torch.load(config.model_dir + "/btc_h1_encoder.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    m15_model = CNNGRUM15(n_features=17, seq_len=config.seq_len_m15,
        cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
    m15_ckpt = torch.load(config.model_dir + "/btc_m15_model.pt", map_location=device, weights_only=False)
    m15_model.load_state_dict(m15_ckpt["model_state_dict"])

    engine = BTCFeatureEngine(); gate = EntryGate()
    tm = TradeManager(initial_sl=1.0, hard_tp=config.hard_tp,
        breakeven_trigger=params["breakeven_trigger"],
        trail_trigger=params["trail_trigger"],
        trail_dist=params["trail_dist"],
        trail_dist_s=params["trail_dist"] * 0.67,
        regime_tighten=config.regime_tighten,
        max_hold=params["max_hold"],
        mae_guard_retrace=config.mae_guard_retrace)
    exec_dr = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)

    # State
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; start_bal = 10000.0
    h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
    last_h1 = None; h1_atr = 0.0; lots = 0.0; pos = 0

    for i in range(max(config.seq_len_m15, 20), len(m15f)):
        ts = m15f["timestamp"].iloc[i]; price = m15f["close"].iloc[i]
        exec_dr._current_price = price
        today = ts.date()
        if ld and today != ld: pnl_d = 0.0; start_bal = bal
        ld = today
        h1s = h1f[h1f["timestamp"] <= ts]
        m15s = m15f.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
        if len(h1s) < config.seq_len_h1: continue

        hl = h1s["timestamp"].max()
        if hl != last_h1:
            last_h1 = hl; h1_feats = engine.compute(h1s)
            seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0).to(device)
            for _, row in h1s.iloc[-14:].iterrows():
                rd.update(row["high"], row["low"], row["close"])
            rr = classify_regime(encoder, classifier, t, rd,
                                 model_confidence_threshold=config.min_regime_confidence)
            g = gate.evaluate(rr["regime"], rr["confidence"],
                              rr.get("atr_percentile", 0.5), bb_position=h1_feats[-1, 4])
            if g.entry_signal:
                h1_sig = g.direction; listen = True; bl = 0
                h1_atr = h1_feats[-1, 6] * price
            else:
                h1_sig = None; listen = False

        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None
            if tm.check_sl_hit(lo, hi):
                epx = tm.exit_price_at_sl(); er = "sl_hit"
            elif tm.check_tp_hit(lo, hi):
                epx = tm.exit_price_at_tp(); er = "tp_hit"
            else:
                a = tm.update(price, hi, lo, h1_atr)
                if a.action_type == TradeActionType.CLOSE: epx = price; er = a.reason
            if epx:
                s_ = tm.state; sl_d = 1.0 * s_.entry_atr
                pnl_r = (epx - s_.entry_price) / sl_d if pos == 1 else (s_.entry_price - epx) / sl_d
                pnl = (epx - s_.entry_price) * lots if pos == 1 else (s_.entry_price - epx) * lots
                bal += pnl; pnl_d += pnl
                trades.append({"pnl_dollar": pnl, "pnl_r": pnl_r, "mfe_r": s_.mfe_r,
                               "exit_reason": er, "bars_held": s_.bars_held})
                pos = 0; tm.state = None
            continue

        if not listen: continue
        bl += 1
        if bl > config.max_listen_bars: listen = False; h1_sig = None; continue

        m15_feats = engine.compute(m15s); confirmed = False
        sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad(): mo = m15_model(tt)
        if mo["entry_confidence"].item() >= config.min_entry_confidence:
            bias = mo["direction_bias"].item()
            if (h1_sig == 1 and bias > 0) or (h1_sig == -1 and bias < 0): confirmed = True
        if not confirmed:
            mc = m15s["close"].values
            ema21 = pd.Series(mc).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc[-1] <= ema21[-1] * 1.01 and mc[-1] > mc[-2]: confirmed = True
            elif h1_sig == -1 and mc[-1] >= ema21[-1] * 0.99 and mc[-1] < mc[-2]: confirmed = True
        if not confirmed: continue
        if abs(pnl_d) / max(start_bal, 1) >= config.max_daily_loss: continue
        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots)
        exec_dr.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    if not trades:
        return {"label": label, "trades": 0, "wr": 0, "pnl": 0, "pf": 0, "dd": 0,
                "avg_r": 0, "ret": 0, "score": -999, "avg_win_r": 0, "tp_pct": 0,
                **params}

    n = len(trades); wins = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    wr = len(wins) / n * 100; tp = sum(t["pnl_dollar"] for t in trades)
    avg_r = sum(t["pnl_r"] for t in trades) / n
    avg_win_r = np.mean([t["pnl_r"] for t in wins]) if wins else 0
    avg_loss_r = np.mean([t["pnl_r"] for t in losses]) if losses else 0
    tg = sum(t["pnl_dollar"] for t in wins)
    tl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = tg / tl if tl > 0 else float("inf")
    tp_hits = sum(1 for t in trades if t["exit_reason"] == "tp_hit")
    tp_pct = tp_hits / n * 100

    cum = np.cumsum([0] + [t["pnl_dollar"] for t in trades])
    eq = 10000 + cum; peak = np.maximum.accumulate(eq)
    dd = float(np.max(np.where(peak > 0, (peak - eq) / peak * 100, 0)))
    ret = (10000 + tp) / 10000 - 1
    score = tp / max(dd, 0.5)

    return {"label": label, "trades": n, "wr": wr, "pnl": tp, "pf": pf, "dd": dd,
            "avg_r": avg_r, "ret": ret, "score": score,
            "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r, "tp_pct": tp_pct,
            **params}


if __name__ == "__main__":
    n_workers = min(cpu_count(), 4)
    print(f"Running {len(ALL_COMBOS)} combos with {n_workers} parallel workers...")
    t_start = time.time()

    mp.set_start_method("spawn", force=True)
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_one, ALL_COMBOS)

    elapsed = (time.time() - t_start) / 60
    print(f"Done in {elapsed:.1f} min")

    df = pd.DataFrame(results).sort_values("score", ascending=False)

    out_path = os.path.join(cfg.log_dir, "parallel_search_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # ── Top 10 ──
    print(f"\n{'='*100}")
    print(f"TOP 10 CONFIGS (score = PnL / max(DD%, 0.5%))")
    print(f"{'='*100}")
    print(f"{'Rank':>4s} {'BE':>5s} {'TrTrig':>7s} {'TrDist':>7s} {'Hold':>5s}  "
          f"{'Trd':>5s} {'WR':>6s} {'PnL':>8s} {'Ret%':>7s} {'PF':>5s} "
          f"{'DD%':>6s} {'AvgR':>6s} {'WinR':>6s} {'LossR':>6s} {'TP%':>5s} {'Score':>7s}")
    print(f"{'─'*4} {'─'*5} {'─'*7} {'─'*7} {'─'*5}  "
          f"{'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*5} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*7}")

    for rank, (_, row) in enumerate(df.head(10).iterrows()):
        print(f"{rank+1:4d} {row['breakeven_trigger']:5.2f} {row['trail_trigger']:7.1f} "
              f"{row['trail_dist']:7.2f} {int(row['max_hold']):5d}  "
              f"{int(row['trades']):5d} {row['wr']:5.1f}% ${row['pnl']:7.0f} "
              f"{row['ret']*100:6.1f}% {row['pf']:5.2f} "
              f"{row['dd']:5.1f}% {row['avg_r']:+6.3f} {row['avg_win_r']:+6.3f} "
              f"{row['avg_loss_r']:+6.3f} {row['tp_pct']:4.1f}% {row['score']:7.1f}")

    # Compare to old best
    print(f"\n{'─'*100}")
    print(f"OLD BEST (BE=0.25, TT=1.5, TD=0.50, MH=18): PnL=$7,819 WR=67.9% DD=21.5% PF=1.15 TP=1.6%")
