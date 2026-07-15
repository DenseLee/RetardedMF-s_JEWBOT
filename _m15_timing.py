"""
M15 Timing Audit: Given H1 direction, does M15 model pick the best entry bar?

Three strategies compared per listening window:
  1. FIRST    — enter on the first M15 bar (earliest possible)
  2. EMA      — enter on first bar where EMA turning rule confirms
  3. M15_BEST — enter on bar with highest M15 confidence (model's pick)
  4. ORACLE   — enter on bar with best actual outcome (hindsight benchmark)

Also: correlation between M15 confidence rank and oracle R rank within window.
"""
import sys, os, json
import numpy as np, pandas as pd, torch
from datetime import datetime, timedelta

sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_gru_m15 import CNNGRUM15
from benchmark.oracle_m15 import M15OracleLabeler

cfg = BTCConfig()
device = torch.device("cpu")
fe = BTCFeatureEngine()

# ── Load M15 model ──
m15_path = os.path.join(cfg.model_dir, "btc_m15_v2.pt")
if not os.path.exists(m15_path):
    m15_path = os.path.join(cfg.model_dir, "btc_m15_model.pt")
m15_model = CNNGRUM15(
    n_features=cfg.n_features, seq_len=cfg.seq_len_m15,
    cnn_channels=cfg.gru_cnn_channels, gru_hidden=cfg.gru_hidden,
    gru_layers=cfg.gru_layers, dropout=cfg.gru_dropout).to(device).eval()
ckpt = torch.load(m15_path, map_location=device, weights_only=False)
sd = ckpt["model_state_dict"]
if "conv1.0.weight" in sd:
    remapped = {}
    bs = {"conv1":0,"conv2":4,"conv3":8}
    for ok, v in sd.items():
        pfx = ok.split(".")[0]
        if pfx in bs:
            rest = ok.split(".",1)[1]; si = int(rest.split(".")[0])
            nk = f"cnn.{bs[pfx]+si}.{rest.split('.',1)[1]}"
        elif ok.startswith("entry_head."): nk = ok.replace("entry_head.","entry_conf.",1)
        else: nk = ok
        remapped[nk] = v
    sd = remapped
m15_model.load_state_dict(sd, strict=False)

# ── Load oracle ──
print("Generating oracle labels...")
lab = M15OracleLabeler(max_hold_m15=72)
oracle_labels = lab.label("2026-05-26", "2026-05-27", use_m1=True)
oracle_by_ts = {}
for ol in oracle_labels:
    oracle_by_ts[ol.timestamp] = ol

# ── Fetch M15 data ──
import MetaTrader5 as mt5
mt5.initialize()
m15r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15,
    datetime(2026,5,25,0,0), datetime(2026,5,27,0,0))
m15 = pd.DataFrame(m15r).rename(columns={'time':'timestamp','tick_volume':'volume'})
m15['timestamp'] = pd.to_datetime(m15['timestamp'], unit='s', utc=True).dt.tz_localize(None)
m15 = m15.sort_values('timestamp').reset_index(drop=True)
mt5.shutdown()

# ── Load listening windows ──
log_path = os.path.join(cfg.log_dir, "h1_eval_BTCBot.jsonl")
h1_evals = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except: continue
        if "atr" in rec and "gate" in rec: h1_evals.append(rec)

listening_windows = []
for ev in h1_evals:
    if ev["gate"]["signal"]:
        ts_str = ev["ts"]
        for fmt in ["%Y-%m-%d %H:%M:%S+00:00","%Y-%m-%dT%H:%M:%S+00:00",
                    "%Y-%m-%d %H:%M:%S.%f+00:00","%Y-%m-%dT%H:%M:%S.%f+00:00"]:
            try:
                bar_dt = datetime.strptime(ts_str, fmt).replace(tzinfo=None)
                break
            except: continue
        else: continue
        listening_windows.append({
            "h1_bar": bar_dt, "direction": ev["gate"]["direction"],
            "window_start": bar_dt,
            "window_end": bar_dt + timedelta(hours=3),  # max 8 bars = 2h
        })

# ── Evaluate every M15 bar in every window ──
window_data = {}  # window_idx → list of (ts, m15_conf, m15_dir_bias, ema_turn, oracle_r)

for m15_i in range(cfg.seq_len_m15, len(m15)):
    ts = m15['timestamp'].iloc[m15_i]
    price = float(m15['close'].iloc[m15_i])

    for wi, w in enumerate(listening_windows):
        if not (w['window_start'] <= ts <= w['window_end']):
            continue

        # M15 model inference
        window_m15 = m15.iloc[m15_i - cfg.seq_len_m15 + 1:m15_i + 1]
        if len(window_m15) < cfg.seq_len_m15: continue
        m15_feats = fe.compute(window_m15)
        seq = fe.compute_sequence(m15_feats, len(m15_feats)-1, cfg.seq_len_m15)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            out = m15_model(tensor)
            m15_conf = out["entry_confidence"].item()
            m15_dir_bias = out["direction_bias"].item()

        # EMA turning rule
        closes = m15['close'].values[:m15_i+1]
        if len(closes) >= 3:
            if w['direction'] == 1: ema_turn = closes[-1] > closes[-2]
            else: ema_turn = closes[-1] < closes[-2]
        else:
            ema_turn = False

        # Oracle R for this direction
        ts_key = ts.strftime("%Y-%m-%d %H:%M:%S")
        ol = oracle_by_ts.get(ts_key)
        oracle_r = (ol.long_r if w['direction'] == 1 else ol.short_r) if ol else 0.0

        if wi not in window_data:
            window_data[wi] = []
        window_data[wi].append({
            "ts": ts, "price": price,
            "m15_conf": round(m15_conf, 4),
            "m15_dir_bias": round(m15_dir_bias, 4),
            "ema_turn": ema_turn,
            "oracle_r": round(oracle_r, 4),
        })
        break  # each M15 bar belongs to at most one window

# ── Compare strategies per window ──
print(f"\n{'='*80}")
print("M15 TIMING AUDIT: Per listening window, which strategy picks the best bar?")
print("=" * 80)

strategies = {
    "FIRST": lambda bars: bars[0],
    "EMA": lambda bars: next((b for b in bars if b["ema_turn"]), None),
    "M15_HIGHEST_CONF": lambda bars: max(bars, key=lambda b: b["m15_conf"]),
    "M15_LOWEST_CONF": lambda bars: min(bars, key=lambda b: b["m15_conf"]),
    "ORACLE_BEST": lambda bars: max(bars, key=lambda b: b["oracle_r"]),
    "ORACLE_WORST": lambda bars: min(bars, key=lambda b: b["oracle_r"]),
}

results = {name: [] for name in strategies}

for wi in sorted(window_data.keys()):
    bars = window_data[wi]
    if len(bars) < 2: continue  # need at least 2 bars for comparison
    w = listening_windows[wi]

    print(f"\nWindow {wi}: H1 {str(w['h1_bar'])[:19]} dir={'LONG' if w['direction']==1 else 'SHORT'} "
          f"({len(bars)} M15 bars)")

    # Print per-bar detail
    best_oracle = max(b["oracle_r"] for b in bars)
    for b in bars:
        marker = ""
        if b["ema_turn"]: marker += " [EMA]"
        if b["oracle_r"] == best_oracle: marker += " ★BEST★"
        print(f"  {str(b['ts'])[:19]}  conf={b['m15_conf']:.4f}  dir_bias={b['m15_dir_bias']:+.4f}  "
              f"oracle={b['oracle_r']:+.4f}R{marker}")

    for name, pick_fn in strategies.items():
        picked = pick_fn(bars)
        if picked is None:
            results[name].append(None)
            continue
        results[name].append({
            "oracle_r": picked["oracle_r"],
            "m15_conf": picked["m15_conf"],
            "is_best": picked["oracle_r"] == best_oracle,
            "rank": sorted(bars, key=lambda b: b["oracle_r"], reverse=True).index(picked) + 1,
        })

# ── Aggregate ──
print(f"\n{'='*80}")
print("AGGREGATE RESULTS")
print("=" * 80)

print(f"\n{'Strategy':<20s} {'Windows':>8s} {'Avg OracleR':>12s} {'Pct of Best':>12s} {'Avg Rank':>10s} {'Hit Best':>10s}")
print("-" * 80)

for name in ["FIRST", "EMA", "M15_HIGHEST_CONF", "M15_LOWEST_CONF", "ORACLE_BEST", "ORACLE_WORST"]:
    vals = [r for r in results[name] if r is not None]
    if not vals: continue
    avg_r = np.mean([v["oracle_r"] for v in vals])
    avg_best_r = np.mean([max(window_data[wi][0]["oracle_r"] for _ in [0]) or
                          max(b["oracle_r"] for b in window_data[wi])
                          for wi in sorted(window_data.keys()) if len(window_data[wi]) >= 2])
    # Recalculate properly
    best_rs = []
    for wi in sorted(window_data.keys()):
        bars = window_data[wi]
        if len(bars) >= 2:
            best_rs.append(max(b["oracle_r"] for b in bars))
    avg_best = np.mean(best_rs) if best_rs else 0
    pct_of_best = (avg_r / avg_best * 100) if avg_best > 0 else 0
    avg_rank = np.mean([v["rank"] for v in vals])
    hit_best = sum(1 for v in vals if v["is_best"])

    print(f"{name:<20s} {len(vals):>8d} {avg_r:>+12.4f}R {pct_of_best:>11.1f}% {avg_rank:>10.2f} {hit_best:>8d}/{len(vals)}")

# ── Rank correlation: within each window, does M15 conf rank match oracle R rank? ──
print(f"\n{'='*80}")
print("WITHIN-WINDOW RANK CORRELATION")
print("=" * 80)

all_conf_ranks = []
all_oracle_ranks = []
for wi in sorted(window_data.keys()):
    bars = window_data[wi]
    if len(bars) < 3: continue
    # Rank by M15 confidence (higher = rank 1) and by oracle R
    conf_ranks = [sorted(bars, key=lambda b: b["m15_conf"], reverse=True).index(b)+1 for b in bars]
    oracle_ranks = [sorted(bars, key=lambda b: b["oracle_r"], reverse=True).index(b)+1 for b in bars]
    all_conf_ranks.extend(conf_ranks)
    all_oracle_ranks.extend(oracle_ranks)

if len(all_conf_ranks) > 2:
    from scipy.stats import spearmanr
    rho, pval = spearmanr(all_conf_ranks, all_oracle_ranks)
    print(f"  Spearman rank correlation: rho={rho:+.4f} (p={pval:.4f})")
    print(f"  N = {len(all_conf_ranks)} M15 bars across {sum(1 for wi in window_data if len(window_data[wi])>=3)} windows")
    if abs(rho) < 0.2:
        print(f"  → No meaningful relationship between M15 confidence and actual outcome rank")
    elif rho > 0:
        print(f"  → M15 confidence weakly predicts better outcomes (higher conf = higher oracle R)")
    else:
        print(f"  → M15 confidence is NEGATIVELY correlated with outcomes (higher conf = LOWER oracle R)")

# ── EMA vs M15 head-to-head ──
print(f"\n{'='*80}")
print("EMA vs M15 HEAD-TO-HEAD")
print("=" * 80)

ema_wins = 0
m15_wins = 0
ties = 0
for wi in sorted(window_data.keys()):
    bars = window_data[wi]
    if len(bars) < 2: continue
    ema_pick = next((b for b in bars if b["ema_turn"]), None)
    m15_pick = max(bars, key=lambda b: b["m15_conf"])
    if ema_pick is None: continue
    if ema_pick["oracle_r"] > m15_pick["oracle_r"]:
        ema_wins += 1
    elif m15_pick["oracle_r"] > ema_pick["oracle_r"]:
        m15_wins += 1
    else:
        ties += 1

print(f"  EMA wins:       {ema_wins}")
print(f"  M15 conf wins:  {m15_wins}")
print(f"  Ties:           {ties}")
if ema_wins + m15_wins > 0:
    print(f"  EMA win rate:   {ema_wins/(ema_wins+m15_wins)*100:.1f}%")

# ── The real question: if we enter on the M15_HIGHEST_CONF bar, what's the PnL? ──
print(f"\n{'='*80}")
print("REAL-WORLD IMPACT: If bot used M15 confidence instead of EMA rule")
print("=" * 80)

# Reconstruct what the bot actually did vs what M15 would have done
# Bot entered on first EMA turn in each window
print("\nBot's actual entries (EMA rule):")
for wi in sorted(window_data.keys()):
    bars = window_data[wi]
    w = listening_windows[wi]
    ema_pick = next((b for b in bars if b["ema_turn"]), None)
    m15_pick = max(bars, key=lambda b: b["m15_conf"])
    if ema_pick is None: continue
    print(f"  H1 {str(w['h1_bar'])[:19]}: EMA→{str(ema_pick['ts'])[:19]} ({ema_pick['oracle_r']:+.3f}R)  "
          f"M15→{str(m15_pick['ts'])[:19]} ({m15_pick['oracle_r']:+.3f}R)  "
          f"diff={m15_pick['oracle_r']-ema_pick['oracle_r']:+.3f}R")

print("\nNote: Oracle R is the MAX available for the H1 direction (SHORT).")
print("H1 direction was wrong today (should have been LONG). So both strategies lose.")
print("The question is: which loses LESS by entering at a better price?")
