"""
Trade diagnostic: why is PF low despite high WR?
Runs best config on target quarters, analyzes per-trade exit details.
"""
import sys, numpy as np, pandas as pd, torch
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST = {"initial_sl": 1.0, "max_hold": 18, "trail_dist": 0.50,
        "breakeven_trigger": 0.25, "mae_guard_retrace": 2.5}

encoder = CNNLSTMEncoder(n_features=17, seq_len=config.seq_len_h1,
    cnn_channels=config.cnn_channels, lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers, dropout=config.lstm_dropout,
    embedding_dim=config.embedding_dim, regime_classes=4,
    bidirectional=True).to(device).eval()
classifier = RegimeClassifier(embedding_dim=128, n_classes=4).to(device).eval()
ckpt = torch.load(config.model_dir + "/btc_h1_encoder.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"]); classifier.load_state_dict(ckpt["classifier_state_dict"])
m15_model = CNNGRUM15(n_features=17, seq_len=config.seq_len_m15,
    cnn_channels=config.gru_cnn_channels, gru_hidden=config.gru_hidden,
    gru_layers=config.gru_layers, dropout=config.gru_dropout).to(device).eval()
m15_ckpt = torch.load(config.model_dir + "/btc_m15_model.pt", map_location=device, weights_only=False)
m15_model.load_state_dict(m15_ckpt["model_state_dict"])

engine = BTCFeatureEngine(); gate = EntryGate()
h1f = pd.read_csv(config.data_dir + "/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(config.data_dir + "/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)

TARGETS = [
    ("2024-Q2 (worst)", "2024-04-01", "2024-06-30"),
    ("2025-Q3 (low PF)", "2025-07-01", "2025-09-30"),
    ("2024-Q4 (best)", "2024-10-01", "2024-12-31"),
]


def run_with_details(label, start, end):
    """Run backtest and return per-trade list + aggregates."""
    s = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
    h1 = h1f[(h1f["timestamp"] >= s) & (h1f["timestamp"] < e)].reset_index(drop=True)
    m15 = m15f[(m15f["timestamp"] >= s) & (m15f["timestamp"] < e)].reset_index(drop=True)
    if len(m15) < 100: return None

    tm = TradeManager(initial_sl=BEST["initial_sl"], hard_tp=config.hard_tp,
        breakeven_trigger=BEST["breakeven_trigger"], trail_trigger=config.trail_trigger,
        trail_dist=BEST["trail_dist"], trail_dist_s=BEST["trail_dist"] * 0.67,
        regime_tighten=config.regime_tighten, max_hold=BEST["max_hold"],
        mae_guard_retrace=BEST["mae_guard_retrace"])
    exec = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; start_bal = 10000.0
    h1_sig = None; listen = False; bl = 0; rd = RuleBasedRegimeDetector()
    last_h1 = None; h1_atr = 0.0; lots = 0.0; pos = 0

    for i in range(max(config.seq_len_m15, 20), len(m15)):
        ts = m15["timestamp"].iloc[i]; price = m15["close"].iloc[i]
        exec._current_price = price
        today = ts.date()
        if ld and today != ld: pnl_d = 0.0; start_bal = bal
        ld = today
        h1s = h1[h1["timestamp"] <= ts]
        m15s = m15.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
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
            epx = None; er = None; orig_phase = tm.state.phase.name
            if tm.check_sl_hit(lo, hi):
                epx = tm.exit_price_at_sl(); er = "sl_hit"
            elif tm.check_tp_hit(lo, hi):
                epx = tm.exit_price_at_tp(); er = "tp_hit"
            else:
                a = tm.update(price, hi, lo, h1_atr)
                if a.action_type == TradeActionType.CLOSE:
                    epx = price; er = a.reason
            if epx:
                s_ = tm.state
                pnl = (epx - s_.entry_price) * lots if pos == 1 else (s_.entry_price - epx) * lots
                bal += pnl; pnl_d += pnl
                # pnl_r computed from entry_atr * initial_sl
                sl_dist = BEST["initial_sl"] * s_.entry_atr
                pnl_r = (epx - s_.entry_price) / sl_dist if pos == 1 else (s_.entry_price - epx) / sl_dist
                trades.append({
                    "pnl_dollar": pnl, "pnl_r": round(pnl_r, 4),
                    "mfe_r": round(s_.mfe_r, 4), "mae_r": round(s_.mae_r, 4),
                    "bars_held": s_.bars_held, "exit_reason": er,
                    "direction": "LONG" if pos == 1 else "SHORT",
                    "phase_at_exit": orig_phase,
                    "entry_atr": s_.entry_atr, "entry_price": s_.entry_price,
                })
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
            if (h1_sig == 1 and bias > 0) or (h1_sig == -1 and bias < 0):
                confirmed = True
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
        exec.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    return trades


def analyze(label, trades):
    """Print detailed trade analysis."""
    if not trades: return
    n = len(trades)
    wins = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]

    print(f"\n{'='*70}")
    print(f"  {label}  —  {n} trades, {len(wins)} wins ({len(wins)/n*100:.1f}%), "
          f"${sum(t['pnl_dollar'] for t in trades):+,.0f}")
    print(f"{'='*70}")

    # ── Win size distribution ──
    print(f"\n  WIN R-MULTIPLE DISTRIBUTION ({len(wins)} wins):")
    r_buckets = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99)]
    for lo, hi in r_buckets:
        count = sum(1 for t in wins if lo <= t["pnl_r"] < hi)
        pct = count / n * 100
        bar = "#" * max(1, int(pct * 2))
        print(f"    {lo:5.2f} - {hi:5.2f}R: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Loss size distribution ──
    print(f"\n  LOSS R-MULTIPLE DISTRIBUTION ({len(losses)} losses):")
    l_buckets = [(-0.25, 0), (-0.5, -0.25), (-1.0, -0.5), (-1.5, -1.0), (-99, -1.5)]
    for lo, hi in l_buckets:
        count = sum(1 for t in losses if lo <= t["pnl_r"] < hi)
        pct = count / n * 100
        bar = "#" * max(1, int(pct))
        print(f"    {lo:+6.2f} to {hi:+6.2f}R: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Exit reason breakdown ──
    print(f"\n  EXIT REASON BREAKDOWN:")
    by_reason = defaultdict(lambda: {"count": 0, "wins": 0, "total_r": 0.0, "avg_r": 0.0,
                                      "avg_mfe": 0.0, "avg_bars": 0.0})
    for t in trades:
        r = by_reason[t["exit_reason"]]
        r["count"] += 1
        if t["pnl_dollar"] > 0: r["wins"] += 1
        r["total_r"] += t["pnl_r"]
        r["avg_mfe"] += t["mfe_r"]
        r["avg_bars"] += t["bars_held"]

    print(f"    {'Reason':>25s} {'Count':>6s} {'WR':>6s} {'AvgR':>7s} {'TotalR':>7s} "
          f"{'AvgMFE':>7s} {'AvgBars':>7s}")
    print(f"    {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for reason, r in sorted(by_reason.items(), key=lambda x: -x[1]["count"]):
        r["avg_r"] = r["total_r"] / r["count"]
        r["avg_mfe"] /= r["count"]
        r["avg_bars"] /= r["count"]
        wr = r["wins"] / r["count"] * 100
        print(f"    {reason:>25s} {r['count']:6d} {wr:5.1f}% {r['avg_r']:+6.3f} "
              f"{r['total_r']:+6.1f}R {r['avg_mfe']:+6.2f}R {r['avg_bars']:6.1f}")

    # ── TP vs early exit for wins ──
    win_reasons = Counter(t["exit_reason"] for t in wins)
    print(f"\n  WIN EXIT TYPES:")
    for reason, count in win_reasons.most_common():
        avg_r = np.mean([t["pnl_r"] for t in wins if t["exit_reason"] == reason])
        avg_mfe = np.mean([t["mfe_r"] for t in wins if t["exit_reason"] == reason])
        capture = avg_r / max(avg_mfe, 0.01) * 100
        print(f"    {reason:>25s}: {count:4d} wins, avg +{avg_r:.2f}R, "
              f"MFE={avg_mfe:.2f}R, capture={capture:.0f}%")

    # ── Loss exit types ──
    loss_reasons = Counter(t["exit_reason"] for t in losses)
    print(f"\n  LOSS EXIT TYPES:")
    for reason, count in loss_reasons.most_common():
        avg_r = np.mean([t["pnl_r"] for t in losses if t["exit_reason"] == reason])
        avg_mfe = np.mean([t["mfe_r"] for t in losses if t["exit_reason"] == reason])
        avg_mae = np.mean([t["mae_r"] for t in losses if t["exit_reason"] == reason])
        print(f"    {reason:>25s}: {count:4d} losses, avg {avg_r:+.3f}R, "
              f"MFE={avg_mfe:+.2f}R, |MAE|={abs(avg_mae):.2f}R")

    # ── MFE capture ratio for wins ──
    captures = []
    for t in wins:
        if t["mfe_r"] > 0.01:
            captures.append(t["pnl_r"] / t["mfe_r"])
    if captures:
        print(f"\n  MFE CAPTURE (wins): mean={np.mean(captures)*100:.0f}% "
              f"median={np.median(captures)*100:.0f}%")

    # ── Directional accuracy ──
    # "Did the trade go in our favor at any point?"
    ever_profitable = sum(1 for t in trades if t["mfe_r"] > 0.1)
    print(f"\n  DIRECTIONAL ACCURACY: {ever_profitable}/{n} ({ever_profitable/n*100:.1f}%) "
          f"trades had MFE > 0.1R")
    never_profitable = sum(1 for t in trades if t["mfe_r"] <= 0.01)
    print(f"  NEVER PROFITABLE: {never_profitable}/{n} ({never_profitable/n*100:.1f}%) "
          f"trades had MFE <= 0.01R")


# ── Run all targets ──
for label, start, end in TARGETS:
    trades = run_with_details(label, start, end)
    if trades:
        analyze(label, trades)
