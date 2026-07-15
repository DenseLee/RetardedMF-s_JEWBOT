"""
Grid search over trade manager parameters.
Reuses trained models, runs 2026 YTD backtest for each config.
Saves results to logs/grid_search_results.csv.
"""
import os, sys, json, time, itertools, logging
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from execution.mt5_executor_btc import DryRunExecutor

logging.basicConfig(level=logging.WARNING)  # suppress per-trade logs from engine

# ── Parameter grid ──
PARAM_GRID = {
    "initial_sl":       [1.0],              # fix for first pass
    "max_hold":         [12, 18, 24],       # most impactful
    "trail_dist":       [0.5, 0.75, 1.0],   # wide vs tight trail
    "breakeven_trigger": [0.25, 0.5],       # early vs standard breakeven
    "mae_guard_retrace": [1.5, 2.5],        # tight vs loose MAE guard
}
# 3×3×2×2 = 36 combos, ~1.5 min each = ~54 min total
ALL_COMBOS = list(itertools.product(
    PARAM_GRID["initial_sl"],
    PARAM_GRID["max_hold"],
    PARAM_GRID["trail_dist"],
    PARAM_GRID["breakeven_trigger"],
    PARAM_GRID["mae_guard_retrace"],
))
print(f"Total combos: {len(ALL_COMBOS)} (pass 1: 36 combos)")

# ── Load models once ──
config = BTCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

encoder = CNNLSTMEncoder(
    n_features=17, seq_len=config.seq_len_h1, cnn_channels=config.cnn_channels,
    lstm_hidden=config.lstm_hidden, lstm_layers=config.lstm_layers,
    dropout=config.lstm_dropout, embedding_dim=config.embedding_dim,
    regime_classes=4, bidirectional=True).to(device).eval()

classifier = RegimeClassifier(embedding_dim=128, n_classes=4).to(device).eval()

ckpt = torch.load(os.path.join(config.model_dir, "btc_h1_encoder.pt"),
                  map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])
print(f"H1 encoder loaded (val_acc={ckpt.get('val_acc','?')}%)")

m15_model = CNNGRUM15(
    n_features=17, seq_len=config.seq_len_m15, cnn_channels=config.gru_cnn_channels,
    gru_hidden=config.gru_hidden, gru_layers=config.gru_layers,
    dropout=config.gru_dropout).to(device).eval()
m15_ckpt = torch.load(os.path.join(config.model_dir, "btc_m15_model.pt"),
                      map_location=device, weights_only=False)
m15_model.load_state_dict(m15_ckpt["model_state_dict"])
print("M15 model loaded")

# ── Pre-load data ──
engine = BTCFeatureEngine()
rule_detector = RuleBasedRegimeDetector()
entry_gate = EntryGate()

h1_path = os.path.join(config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
m15_path = os.path.join(config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")
h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
m15 = pd.read_csv(m15_path); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)

from_ts = pd.Timestamp("2026-01-01", tz="UTC")
h1 = h1[h1["timestamp"] >= from_ts].reset_index(drop=True)
m15 = m15[m15["timestamp"] >= from_ts].reset_index(drop=True)
print(f"Data: {len(h1)} H1, {len(m15)} M15 bars")
print(f"Range: {m15['timestamp'].min()} → {m15['timestamp'].max()}")


# ── Single backtest runner ──
def run_backtest(tm_params: dict, risk_pct=0.02) -> dict:
    """Run YTD backtest with given TradeManager parameters. Returns metrics dict."""
    tm = TradeManager(
        initial_sl=tm_params["initial_sl"],
        hard_tp=config.hard_tp,
        breakeven_trigger=tm_params["breakeven_trigger"],
        trail_trigger=config.trail_trigger,
        trail_dist=tm_params["trail_dist"],
        trail_dist_s=tm_params["trail_dist"] * 0.67,  # shorts ~2/3 of long
        regime_tighten=config.regime_tighten,
        max_hold=tm_params["max_hold"],
        mae_guard_retrace=tm_params["mae_guard_retrace"],
    )

    executor = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)
    balance = 10000.0; starting_balance = 10000.0
    position = 0; position_ticket = None; position_lots = 0.0
    daily_pnl = 0.0; last_day = None
    trades = []

    # Listening state
    h1_signal = None; h1_confidence = 0.0; h1_atr = 0.0
    listening = False; bars_listened = 0; max_listen = 8

    # Rule detector (per-run copy)
    rd = RuleBasedRegimeDetector()

    m15_start = max(config.seq_len_m15, 20)
    last_h1_key = None

    for i in range(m15_start, len(m15)):
        ts = m15["timestamp"].iloc[i]
        price = m15["close"].iloc[i]
        executor._current_price = price

        # Daily reset
        today = ts.date()
        if last_day and today != last_day:
            daily_pnl = 0.0; starting_balance = balance
        last_day = today

        # H1 slice
        h1_slice = h1[h1["timestamp"] <= ts]
        m15_slice = m15.iloc[max(0, i - config.seq_len_m15 * 4):i + 1]
        if len(h1_slice) < config.seq_len_h1:
            continue

        # H1 bar close
        h1_latest = h1_slice["timestamp"].max()
        if h1_latest != last_h1_key:
            last_h1_key = h1_latest
            h1_feats = engine.compute(h1_slice)
            seq = engine.compute_sequence(h1_feats, len(h1_feats) - 1, config.seq_len_h1)
            t = torch.from_numpy(seq).unsqueeze(0).to(device)

            # Update rule detector
            for _, row in h1_slice.iloc[-14:].iterrows():
                rd.update(row["high"], row["low"], row["close"])

            regime_result = classify_regime(encoder, classifier, t, rd,
                                            model_confidence_threshold=config.min_regime_confidence)
            gate = entry_gate.evaluate(
                regime_result["regime"], regime_result["confidence"],
                regime_result.get("atr_percentile", 0.5),
                bb_position=h1_feats[-1, 4])

            if gate.entry_signal:
                h1_signal = gate.direction
                h1_confidence = gate.confidence
                h1_atr = h1_feats[-1, 6] * price
                listening = True
                bars_listened = 0
            else:
                h1_signal = None; listening = False

        # ── Position management ──
        if position != 0 and tm.state is not None:
            hi = m15_slice["high"].iloc[-1]; lo = m15_slice["low"].iloc[-1]
            exit_price = None; exit_reason = None
            if tm.check_sl_hit(lo, hi):
                exit_price = tm.exit_price_at_sl(); exit_reason = "sl_hit"
            elif tm.check_tp_hit(lo, hi):
                exit_price = tm.exit_price_at_tp(); exit_reason = "tp_hit"
            else:
                action = tm.update(price, hi, lo, h1_atr)
                if action.action_type == TradeActionType.CLOSE:
                    exit_price = price; exit_reason = action.reason

            if exit_price is not None:
                # Compute PnL
                s = tm.state
                if position == 1:
                    pnl = (exit_price - s.entry_price) * position_lots * 1.0  # contract_size=1
                else:
                    pnl = (s.entry_price - exit_price) * position_lots * 1.0
                balance += pnl; daily_pnl += pnl
                trades.append({"pnl_dollar": pnl, "pnl_r": s.unrealized_pnl_r,
                               "mfe_r": s.mfe_r, "mae_r": s.mae_r,
                               "bars_held": s.bars_held, "exit_reason": exit_reason,
                               "direction": "LONG" if position == 1 else "SHORT"})
                position = 0; tm.state = None
            continue

        # ── M15 listener ──
        if not listening: continue
        bars_listened += 1
        if bars_listened > max_listen:
            listening = False; h1_signal = None; continue

        # M15 confirmation
        m15_feats = engine.compute(m15_slice)
        confirmed = False
        # NN model
        seq_m = engine.compute_sequence(m15_feats, len(m15_feats) - 1, config.seq_len_m15)
        tm_t = torch.from_numpy(seq_m).unsqueeze(0).to(device)
        with torch.no_grad():
            m15_out = m15_model(tm_t)
        if m15_out["entry_confidence"].item() >= config.min_entry_confidence:
            bias = m15_out["direction_bias"].item()
            if (h1_signal == 1 and bias > 0) or (h1_signal == -1 and bias < 0):
                confirmed = True
        # Rule fallback
        if not confirmed:
            mc = m15_slice["close"].values
            ema21 = pd.Series(mc).ewm(span=21, adjust=False).mean().values
            if h1_signal == 1 and mc[-1] <= ema21[-1] * 1.01 and mc[-1] > mc[-2]:
                confirmed = True
            elif h1_signal == -1 and mc[-1] >= ema21[-1] * 0.99 and mc[-1] < mc[-2]:
                confirmed = True

        if not confirmed: continue

        # Enter
        if not (abs(daily_pnl) / max(starting_balance, 1) < config.max_daily_loss):
            continue

        listening = False
        lots = tm.compute_position_size(balance, h1_atr, price, risk_pct, tm.initial_sl)
        tm.enter(h1_signal, price, h1_atr, lots)
        executor.open_position(h1_signal, lots, tm.state.current_sl, tm.state.current_tp)
        position = h1_signal; position_lots = lots

    # ── Compute metrics ──
    if not trades:
        return {**tm_params, "trades": 0, "win_rate": 0, "total_pnl": 0,
                "pf": 0, "max_dd": 0, "avg_r": 0, "return_pct": 0, "score": -999}

    n = len(trades)
    wins = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    wr = len(wins) / n * 100
    total_pnl = sum(t["pnl_dollar"] for t in trades)
    avg_r = sum(t["pnl_r"] for t in trades) / n
    total_gain = sum(t["pnl_dollar"] for t in wins)
    total_loss = abs(sum(t["pnl_dollar"] for t in losses))
    pf = total_gain / total_loss if total_loss > 0 else float("inf")

    # Drawdown
    cum = np.cumsum([0] + [t["pnl_dollar"] for t in trades])
    eq = 10000 + cum
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (peak - eq) / peak * 100, 0)
    max_dd = float(np.max(dd))

    final_balance = 10000 + total_pnl
    return_pct = (final_balance / 10000 - 1) * 100

    # Score: penalize drawdown heavily, reward PnL
    score = total_pnl / max(max_dd, 0.5)  # risk-adjusted return

    return {**tm_params, "trades": n, "win_rate": wr, "total_pnl": total_pnl,
            "pf": pf, "max_dd": max_dd, "avg_r": avg_r,
            "return_pct": return_pct, "score": score, "final_balance": final_balance}


# ── Run grid search ──
print(f"\nRunning {len(ALL_COMBOS)} configs...")
results = []
t_start = time.time()
for idx, (sl, mh, td, be, mg) in enumerate(ALL_COMBOS):
    params = {"initial_sl": sl, "max_hold": mh, "trail_dist": td,
              "breakeven_trigger": be, "mae_guard_retrace": mg}
    r = run_backtest(params)
    results.append(r)
    elapsed = (time.time() - t_start) / 60
    eta = elapsed / (idx + 1) * (len(ALL_COMBOS) - idx - 1) if idx > 0 else 0
    print(f"  [{idx+1}/{len(ALL_COMBOS)}] {elapsed:.1f}m ETA {eta:.1f}m  "
          f"SL={sl:.2f} H={int(mh)} T={td:.2f} BE={be:.2f} MAE={mg:.1f}  "
          f"PnL=${r['total_pnl']:+.0f} WR={r['win_rate']:.1f}% DD={r['max_dd']:.1f}% PF={r['pf']:.2f}")

# ── Save & report ──
df = pd.DataFrame(results)
df = df.sort_values("score", ascending=False)
out_path = os.path.join(config.log_dir, "grid_search_results.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} results to {out_path}")

print(f"\n{'='*80}")
print(f"TOP 10 CONFIGS (by risk-adjusted score = PnL / max(DD%, 0.5%))")
print(f"{'='*80}")
print(f"{'Rank':>4s} {'SL':>5s} {'Hold':>5s} {'Trail':>6s} {'BE':>5s} {'MAE':>5s}  "
      f"{'Trades':>6s} {'WR':>6s} {'PnL':>8s} {'Ret%':>7s} {'PF':>5s} {'DD%':>6s} {'AvgR':>6s} {'Score':>7s}")
print(f"{'─'*4} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5}  "
      f"{'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*5} {'─'*6} {'─'*6} {'─'*7}")
for rank, (_, row) in enumerate(df.head(10).iterrows()):
    print(f"{rank+1:4d} {row['initial_sl']:5.2f} {int(row['max_hold']):5d} {row['trail_dist']:6.2f} "
          f"{row['breakeven_trigger']:5.2f} {row['mae_guard_retrace']:5.1f}  "
          f"{int(row['trades']):6d} {row['win_rate']:5.1f}% ${row['total_pnl']:7.0f} "
          f"{row['return_pct']:6.1f}% {row['pf']:5.2f} {row['max_dd']:5.1f}% "
          f"{row['avg_r']:+6.3f} {row['score']:7.1f}")

# Print current default for comparison
print(f"\n{'─'*80}")
default = next((r for r in results if r["initial_sl"] == 1.0 and r["max_hold"] == 12
                and r["trail_dist"] == 0.75 and r["breakeven_trigger"] == 0.5
                and r["mae_guard_retrace"] == 1.5), None)
if default:
    print(f"CURRENT DEFAULT: SL=1.0 Hold=12 Trail=0.75 BE=0.5 MAE=1.5  →  "
          f"PnL=${default['total_pnl']:+.0f} WR={default['win_rate']:.1f}% "
          f"DD={default['max_dd']:.1f}% PF={default['pf']:.2f}")
