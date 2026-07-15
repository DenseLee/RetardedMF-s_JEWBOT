"""
Deep exit analysis: answer specific questions about win/loss structure.
Runs best config (BE=0.50, TT=2.5, TD=0.75, MH=18) on 2026 YTD.
"""
import sys, numpy as np, pandas as pd, torch
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

config = BTCConfig(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST = {"initial_sl": 1.0, "max_hold": 18, "trail_dist": 0.75,
        "breakeven_trigger": 0.50, "mae_guard_retrace": 2.5, "trail_trigger": 2.5}

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

s = pd.Timestamp("2026-01-01", tz="UTC"); e = pd.Timestamp("2026-05-06", tz="UTC")
h1 = h1f[(h1f["timestamp"] >= s) & (h1f["timestamp"] < e)].reset_index(drop=True)
m15 = m15f[(m15f["timestamp"] >= s) & (m15f["timestamp"] < e)].reset_index(drop=True)

# ── Run backtest capturing full bar-level data per trade ──
tm = TradeManager(initial_sl=BEST["initial_sl"], hard_tp=config.hard_tp,
    breakeven_trigger=BEST["breakeven_trigger"], trail_trigger=BEST["trail_trigger"],
    trail_dist=BEST["trail_dist"], trail_dist_s=BEST["trail_dist"] * 0.67,
    regime_tighten=config.regime_tighten, max_hold=BEST["max_hold"],
    mae_guard_retrace=BEST["mae_guard_retrace"])
exec_dr = DryRunExecutor(symbol=config.symbol, initial_balance=10000.0)

bal = 10000.0; pnl_d = 0.0; ld = None; trades = []; start_bal = 10000.0
h1_sig = None; listen = False; bl2 = 0; rd = RuleBasedRegimeDetector()
last_h1 = None; h1_atr = 0.0; lots = 0.0; pos = 0
active_trade_bars = []  # track bar-level MFE during active trade

for i in range(max(config.seq_len_m15, 20), len(m15)):
    ts = m15["timestamp"].iloc[i]; price = m15["close"].iloc[i]
    exec_dr._current_price = price
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
            h1_sig = g.direction; listen = True; bl2 = 0
            h1_atr = h1_feats[-1, 6] * price
        else:
            h1_sig = None; listen = False

    if pos != 0 and tm.state is not None:
        hi = m15s["high"].iloc[-1]; lo = m15s["low"].iloc[-1]
        # Track bar-level MFE
        s2 = tm.state
        sl_d = BEST["initial_sl"] * s2.entry_atr
        mfe_now = (hi - s2.entry_price) / sl_d if pos == 1 else (s2.entry_price - lo) / sl_d
        mae_now = (lo - s2.entry_price) / sl_d if pos == 1 else (s2.entry_price - hi) / sl_d
        active_trade_bars.append({
            "bar": len(active_trade_bars), "price": price, "mfe": mfe_now, "mae": mae_now,
            "hi": hi, "lo": lo, "sl": s2.current_sl, "tp": s2.current_tp,
        })

        epx = None; er = None
        if tm.check_sl_hit(lo, hi):
            epx = tm.exit_price_at_sl(); er = "sl_hit"
        elif tm.check_tp_hit(lo, hi):
            epx = tm.exit_price_at_tp(); er = "tp_hit"
        else:
            a = tm.update(price, hi, lo, h1_atr)
            if a.action_type == TradeActionType.CLOSE: epx = price; er = a.reason

        if epx:
            pnl_r = (epx - s2.entry_price) / sl_d if pos == 1 else (s2.entry_price - epx) / sl_d
            pnl = (epx - s2.entry_price) * lots if pos == 1 else (s2.entry_price - epx) * lots
            bal += pnl; pnl_d += pnl

            # Capture MFE trajectory during trade
            mfe_peak = max(b["mfe"] for b in active_trade_bars) if active_trade_bars else 0
            mfe_at_exit = active_trade_bars[-1]["mfe"] if active_trade_bars else 0
            mfe_peak_bar = np.argmax([b["mfe"] for b in active_trade_bars]) if active_trade_bars else 0
            total_bars = len(active_trade_bars)
            mfe_after_peak = mfe_peak - mfe_at_exit  # how much MFE was lost after peak
            # When did MFE peak relative to exit?
            peak_pct = mfe_peak_bar / max(total_bars, 1) * 100  # % through trade when peak hit

            # How much more time was available?
            h1_idx = h1s.index[-1]  # current H1 bar index
            remaining_h1 = len(h1) - h1_idx - 1  # bars left in dataset
            would_win_more_time = mfe_peak >= 0.5 and remaining_h1 >= 6  # had room

            # Would wider SL have saved it?
            would_win_wider_sl = False
            if er == "sl_hit" and pnl_r < 0:
                # Check if price recovered after hitting SL
                # Look ahead in M15 data
                look_ahead = min(20, len(m15) - i - 1)
                future_m15 = m15.iloc[i+1:i+1+look_ahead]
                if pos == 1:
                    # Did price go above entry+0.5R after hitting SL?
                    recovery_price = s2.entry_price + 0.5 * sl_d
                    if any(future_m15["high"] >= recovery_price):
                        would_win_wider_sl = True
                else:
                    recovery_price = s2.entry_price - 0.5 * sl_d
                    if any(future_m15["low"] <= recovery_price):
                        would_win_wider_sl = True

            # Late exit from reversal? (had MFE > 0.5R but closed as loss)
            late_exit_reversal = (pnl_r < 0 and mfe_peak >= 0.5)

            trades.append({
                "pnl_dollar": pnl, "pnl_r": round(pnl_r, 4),
                "mfe_peak": round(mfe_peak, 4), "mfe_at_exit": round(mfe_at_exit, 4),
                "mae_peak": round(min(b["mae"] for b in active_trade_bars), 4) if active_trade_bars else 0,
                "bars_held": total_bars, "exit_reason": er,
                "direction": "LONG" if pos == 1 else "SHORT",
                "mfe_after_peak": round(mfe_after_peak, 4),
                "peak_pct": round(peak_pct, 1),
                "would_win_more_time": would_win_more_time,
                "would_win_wider_sl": would_win_wider_sl,
                "late_exit_reversal": late_exit_reversal,
            })
            pos = 0; tm.state = None; active_trade_bars = []
        continue

    if not listen: continue
    bl2 += 1
    if bl2 > config.max_listen_bars: listen = False; h1_sig = None; continue

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

# ── ANALYSIS ──
print(f"\n{'='*70}")
print(f"  EXIT ANALYSIS — Best Config (BE=0.50, TT=2.5, TD=0.75, MH=18)")
print(f"  {len(trades)} trades on 2026 YTD")
print(f"{'='*70}")

n = len(trades)
wins = [t for t in trades if t["pnl_r"] > 0]
losses = [t for t in trades if t["pnl_r"] <= 0]
tp_wins = [t for t in trades if t["exit_reason"] == "tp_hit"]
time_stops = [t for t in trades if t["exit_reason"] == "Time stop"]
sl_hits = [t for t in trades if t["exit_reason"] == "sl_hit"]

# ── Q1: Are wins timing out before reaching max profit? ──
print(f"\n{'─'*70}")
print(f"  Q1: ARE WINS EXITING BEFORE REACHING MAX PROFIT?")
print(f"{'─'*70}")
# Wins that exited via time stop: did MFE keep climbing after exit?
win_time_stops = [t for t in time_stops if t["pnl_r"] > 0]
win_sl_hits = [t for t in sl_hits if t["pnl_r"] > 0]

print(f"  Win via Time Stop: {len(win_time_stops)} trades, avg +{np.mean([t['pnl_r'] for t in win_time_stops]):.3f}R" if win_time_stops else "  No win time stops")
print(f"  Win via SL Hit:   {len(win_sl_hits)} trades, avg +{np.mean([t['pnl_r'] for t in win_sl_hits]):.3f}R" if win_sl_hits else "  No win SL hits")
print(f"  Win via TP Hit:   {len(tp_wins)} trades, avg +{np.mean([t['pnl_r'] for t in tp_wins]):.3f}R" if tp_wins else "  No TP hits")

if win_time_stops:
    mfe_at_exit = np.mean([t["mfe_at_exit"] for t in win_time_stops])
    mfe_peak = np.mean([t["mfe_peak"] for t in win_time_stops])
    peak_pct = np.mean([t["peak_pct"] for t in win_time_stops])
    print(f"\n  Win Time Stops: MFE peak={mfe_peak:.2f}R, MFE at exit={mfe_at_exit:.2f}R")
    print(f"  Peak occurred at {peak_pct:.0f}% through trade (then retraced)")
    # How many could have been bigger wins with more time?
    could_be_bigger = [t for t in win_time_stops if t["pnl_r"] < t["mfe_peak"] * 0.5]
    print(f"  {len(could_be_bigger)}/{len(win_time_stops)} captured <50% of MFE peak")

if win_sl_hits:
    mfe_at_exit2 = np.mean([t["mfe_at_exit"] for t in win_sl_hits])
    mfe_peak2 = np.mean([t["mfe_peak"] for t in win_sl_hits])
    print(f"\n  Win SL Hits (trailing/breakeven): MFE peak={mfe_peak2:.2f}R, MFE at exit={mfe_at_exit2:.2f}R")
    captured_half = [t for t in win_sl_hits if t["pnl_r"] < t["mfe_peak"] * 0.5]
    print(f"  {len(captured_half)}/{len(win_sl_hits)} captured <50% of MFE peak before SL hit")

# ── Q2: Big losses — SL vs time stop? ──
print(f"\n{'─'*70}")
print(f"  Q2: HOW OFTEN ARE BIG LOSSES DUE TO SL vs TIME STOP?")
print(f"{'─'*70}")
big_losses = [t for t in losses if t["pnl_r"] < -0.5]
big_loss_sl = [t for t in big_losses if t["exit_reason"] == "sl_hit"]
big_loss_time = [t for t in big_losses if t["exit_reason"] == "Time stop"]

print(f"  Big losses (< -0.5R): {len(big_losses)}/{len(losses)} ({len(big_losses)/max(len(losses),1)*100:.1f}% of losses)")
print(f"    Via SL hit:     {len(big_loss_sl)} trades, avg {np.mean([t['pnl_r'] for t in big_loss_sl]):.3f}R" if big_loss_sl else "")
print(f"    Via Time stop:  {len(big_loss_time)} trades, avg {np.mean([t['pnl_r'] for t in big_loss_time]):.3f}R" if big_loss_time else "")

# Loss distribution by exit
print(f"\n  ALL LOSSES by exit reason:")
for reason in ["sl_hit", "Time stop", "MAE guard"]:
    subset = [t for t in losses if reason in t["exit_reason"]]
    if subset:
        print(f"    {reason:>15s}: {len(subset):4d}, avg {np.mean([t['pnl_r'] for t in subset]):+.3f}R, "
              f"avg bars={np.mean([t['bars_held'] for t in subset]):.1f}")

# ── Q3: Losses that could have been wins with more time or wider SL ──
print(f"\n{'─'*70}")
print(f"  Q3: HOW MANY LOSSES COULD HAVE BEEN PROFITABLE?")
print(f"{'─'*70}")
could_win_more_time = [t for t in losses if t["would_win_more_time"]]
could_win_wider_sl = [t for t in losses if t["would_win_wider_sl"]]
print(f"  Losses that had MFE >= 0.5R + 6 bars remaining: {len(could_win_more_time)}/{len(losses)} ({len(could_win_more_time)/max(len(losses),1)*100:.1f}%)")
print(f"  Losses where price recovered >0.5R after SL hit: {len(could_win_wider_sl)}/{len(losses)} ({len(could_win_wider_sl)/max(len(losses),1)*100:.1f}%)")

if could_win_more_time:
    avg_mfe = np.mean([t["mfe_peak"] for t in could_win_more_time])
    print(f"  → These had avg MFE of +{avg_mfe:.2f}R before turning to loss")
    print(f"  → Avg exit: {np.mean([t['pnl_r'] for t in could_win_more_time]):.3f}R after {np.mean([t['bars_held'] for t in could_win_more_time]):.0f} bars")

# ── Q4: Late exits from reversals (had MFE > 0.5R, ended as loss) ──
print(f"\n{'─'*70}")
print(f"  Q4: LATE EXITS FROM REVERSALS (MFE > 0.5R → LOSS)")
print(f"{'─'*70}")
reversal_losses = [t for t in trades if t["late_exit_reversal"]]
print(f"  Reversal losses (MFE >= 0.5R, ended loss): {len(reversal_losses)}/{n} ({len(reversal_losses)/n*100:.1f}% of all trades)")

if reversal_losses:
    avg_mfe = np.mean([t["mfe_peak"] for t in reversal_losses])
    avg_pnl = np.mean([t["pnl_r"] for t in reversal_losses])
    avg_bars = np.mean([t["bars_held"] for t in reversal_losses])
    avg_peak_pct = np.mean([t["peak_pct"] for t in reversal_losses])
    print(f"  Avg MFE peak: +{avg_mfe:.2f}R  →  Avg exit: {avg_pnl:+.3f}R")
    print(f"  Avg bars held: {avg_bars:.0f}, peak at {avg_peak_pct:.0f}% through trade")
    print(f"  Avg MFE lost after peak: {np.mean([t['mfe_after_peak'] for t in reversal_losses]):.2f}R")

    by_exit = defaultdict(list)
    for t in reversal_losses:
        by_exit[t["exit_reason"]].append(t["pnl_r"])
    print(f"\n  Reversal losses by exit reason:")
    for reason, r_list in sorted(by_exit.items()):
        print(f"    {reason:>25s}: {len(r_list):4d} trades, avg {np.mean(r_list):+.3f}R")

    # These are the most painful: had profit, gave it all back
    worst_reversals = [t for t in reversal_losses if t["mfe_peak"] >= 1.5]
    print(f"\n  MOST PAINFUL (MFE >= 1.5R → loss): {len(worst_reversals)} trades")
    if worst_reversals:
        print(f"    Avg MFE: +{np.mean([t['mfe_peak'] for t in worst_reversals]):.2f}R → "
              f"exit: {np.mean([t['pnl_r'] for t in worst_reversals]):+.3f}R")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Total trades: {n}")
print(f"  Wins: {len(wins)} ({len(wins)/n*100:.1f}%) avg +{np.mean([t['pnl_r'] for t in wins]):.3f}R")
print(f"  Losses: {len(losses)} ({len(losses)/n*100:.1f}%) avg {np.mean([t['pnl_r'] for t in losses]):.3f}R")
print(f"  TP hits: {len(tp_wins)} ({len(tp_wins)/n*100:.1f}%)")
print(f"  Time stops: {len(time_stops)} (win={len(win_time_stops)}, loss={len(time_stops)-len(win_time_stops)})")
print(f"  Reversal losses: {len(reversal_losses)} ({len(reversal_losses)/n*100:.1f}%)")
print(f"  Wide-SL savable: {len(could_win_wider_sl)} ({len(could_win_wider_sl)/max(len(losses),1)*100:.1f}% of losses)")
