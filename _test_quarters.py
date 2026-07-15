"""Run best trade manager config on each quarter 2024 → YTD."""
import sys, time, numpy as np, pandas as pd, torch
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
print(f"Device: {device}")

BEST = {"initial_sl": 1.0, "max_hold": 18, "trail_dist": 0.50,
        "breakeven_trigger": 0.25, "mae_guard_retrace": 2.5}

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
print("Models loaded")

engine = BTCFeatureEngine(); gate = EntryGate()
h1f = pd.read_csv(config.data_dir + "/(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
h1f["timestamp"] = pd.to_datetime(h1f["timestamp"], utc=True)
m15f = pd.read_csv(config.data_dir + "/(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
m15f["timestamp"] = pd.to_datetime(m15f["timestamp"], utc=True)

QUARTERS = [
    ("2024-Q1", "2024-01-01", "2024-03-31"),
    ("2024-Q2", "2024-04-01", "2024-06-30"),
    ("2024-Q3", "2024-07-01", "2024-09-30"),
    ("2024-Q4", "2024-10-01", "2024-12-31"),
    ("2025-Q1", "2025-01-01", "2025-03-31"),
    ("2025-Q2", "2025-04-01", "2025-06-30"),
    ("2025-Q3", "2025-07-01", "2025-09-30"),
    ("2025-Q4", "2025-10-01", "2025-12-31"),
    ("2026-Q1", "2026-01-01", "2026-03-31"),
    ("2026-Q2", "2026-04-01", "2026-05-06"),
]


def run_period(label, start, end):
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

        # Position management
        if pos != 0 and tm.state is not None:
            hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
            epx = None; er = None
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
                trades.append({"pnl_dollar": pnl, "pnl_r": s_.unrealized_pnl_r,
                               "mfe_r": s_.mfe_r, "mae_r": s_.mae_r,
                               "bars_held": s_.bars_held, "exit_reason": er})
                pos = 0; tm.state = None
            continue

        if not listen: continue
        bl += 1
        if bl > config.max_listen_bars: listen = False; h1_sig = None; continue

        # M15 confirmation
        m15_feats = engine.compute(m15s); confirmed = False
        sm = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tt = torch.from_numpy(sm).unsqueeze(0).to(device)
        with torch.no_grad():
            mo = m15_model(tt)
        if mo["entry_confidence"].item() >= config.min_entry_confidence:
            bias = mo["direction_bias"].item()
            if (h1_sig == 1 and bias > 0) or (h1_sig == -1 and bias < 0):
                confirmed = True
        if not confirmed:
            mc = m15s["close"].values
            ema21 = pd.Series(mc).ewm(span=21, adjust=False).mean().values
            if h1_sig == 1 and mc[-1] <= ema21[-1] * 1.01 and mc[-1] > mc[-2]:
                confirmed = True
            elif h1_sig == -1 and mc[-1] >= ema21[-1] * 0.99 and mc[-1] < mc[-2]:
                confirmed = True
        if not confirmed: continue

        if abs(pnl_d) / max(start_bal, 1) >= config.max_daily_loss: continue
        listen = False
        lots = tm.compute_position_size(bal, h1_atr, price, config.risk_pct, tm.initial_sl)
        tm.enter(h1_sig, price, h1_atr, lots)
        exec.open_position(h1_sig, lots, tm.state.current_sl, tm.state.current_tp)
        pos = h1_sig

    if not trades:
        return {"label": label, "trades": 0, "wr": 0, "pnl": 0, "pf": 0,
                "dd": 0, "avg_r": 0, "ret": 0, "bars_h1": len(h1)}

    n = len(trades); wins = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    wr = len(wins) / n * 100
    tp = sum(t["pnl_dollar"] for t in trades)
    avg_r = sum(t["pnl_r"] for t in trades) / n
    tg = sum(t["pnl_dollar"] for t in wins)
    tl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = tg / tl if tl > 0 else float("inf")
    cum = np.cumsum([0] + [t["pnl_dollar"] for t in trades])
    eq = 10000 + cum; peak = np.maximum.accumulate(eq)
    dd = float(np.max(np.where(peak > 0, (peak - eq) / peak * 100, 0)))
    ret = (10000 + tp) / 10000 - 1
    return {"label": label, "trades": n, "wr": wr, "pnl": tp, "pf": pf,
            "dd": dd, "avg_r": avg_r, "ret": ret, "bars_h1": len(h1)}


print(f"\n{'Quarter':>10s} {'Trades':>6s} {'WR':>6s} {'PnL':>9s} {'Ret%':>7s} {'PF':>5s} {'DD%':>6s} {'AvgR':>6s} {'H1bars':>7s}")
print("-" * 75)
all_trades = 0; total_pnl = 0.0
for label, start, end in QUARTERS:
    r = run_period(label, start, end)
    if r is None: continue
    all_trades += r["trades"]; total_pnl += r["pnl"]
    print(f"{r['label']:>10s} {r['trades']:6d} {r['wr']:5.1f}% ${r['pnl']:8.0f} {r['ret']*100:6.1f}% {r['pf']:5.2f} {r['dd']:5.1f}% {r['avg_r']:+6.3f} {r['bars_h1']:7d}")

print("-" * 75)
print(f"{'ALL':>10s} {all_trades:6d} {'':>6s} ${total_pnl:8.0f}")
