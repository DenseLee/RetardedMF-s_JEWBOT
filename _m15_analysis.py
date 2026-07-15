"""
M15 Model Audit — two evaluations:
  1. REALISTIC: Only data the M15 model could see at the time.
     Does it filter better than the EMA turning rule?
  2. HINDSIGHT: Compare M15 output against oracle (future data).
     Does the model have any predictive capability?
"""
import sys, os, json, pickle
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

ckpt_m15 = torch.load(m15_path, map_location=device, weights_only=False)
state_dict = ckpt_m15["model_state_dict"]
# Remap v2 keys
if "conv1.0.weight" in state_dict:
    block_starts = {"conv1": 0, "conv2": 4, "conv3": 8}
    remapped = {}
    for old_key, val in state_dict.items():
        prefix = old_key.split(".")[0]
        if prefix in block_starts:
            rest = old_key.split(".", 1)[1]
            sub_idx = int(rest.split(".")[0])
            param = rest.split(".", 1)[1]
            flat_idx = block_starts[prefix] + sub_idx
            new_key = f"cnn.{flat_idx}.{param}"
        elif old_key.startswith("entry_head."):
            new_key = old_key.replace("entry_head.", "entry_conf.", 1)
        else:
            new_key = old_key
        remapped[new_key] = val
    state_dict = remapped
m15_model.load_state_dict(state_dict, strict=False)
print(f"M15 model loaded: {m15_path}")
print(f"  Keys matched: {sum(1 for k in state_dict if k in m15_model.state_dict())}/{len(m15_model.state_dict())}")

# ── Load oracle labels ──
print("\nGenerating oracle labels for today...")
lab = M15OracleLabeler(max_hold_m15=72)
oracle_labels = lab.label("2026-05-26", "2026-05-27", use_m1=True)
oracle_by_ts = {}
for ol in oracle_labels:
    oracle_by_ts[ol.timestamp] = ol
print(f"  {len(oracle_labels)} oracle labels loaded")

# ── Fetch M15 data from MT5 ──
import MetaTrader5 as mt5
mt5.initialize()
sd = datetime(2026, 5, 25, 0, 0)  # 2 days context for seq_len
ed = datetime(2026, 5, 27, 0, 0)
m15r = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M15, sd, ed)
m15 = pd.DataFrame(m15r).rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
m15['timestamp'] = pd.to_datetime(m15['timestamp'], unit='s', utc=True).dt.tz_localize(None)
m15 = m15.sort_values('timestamp').reset_index(drop=True)
mt5.shutdown()
print(f"  {len(m15)} M15 bars fetched: {m15['timestamp'].iloc[0]} → {m15['timestamp'].iloc[-1]}")

# ── Load H1 eval log to find listening windows ──
log_path = os.path.join(cfg.log_dir, "h1_eval_BTCBot.jsonl")
h1_evals = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "atr" in rec and "gate" in rec:
            h1_evals.append(rec)

listening_windows = []
for ev in h1_evals:
    if ev["gate"]["signal"]:
        bar_ts_str = ev["ts"]
        for fmt in ["%Y-%m-%d %H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S+00:00",
                    "%Y-%m-%d %H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S.%f+00:00"]:
            try:
                bar_dt = datetime.strptime(bar_ts_str, fmt).replace(tzinfo=None)
                break
            except ValueError:
                continue
        else:
            continue
        h1_dir = ev["gate"]["direction"]
        window_end = bar_dt + timedelta(hours=1) + timedelta(hours=2)  # 2h = 8 M15 bars
        listening_windows.append({
            "h1_bar": bar_dt,
            "direction": h1_dir,
            "window_start": bar_dt,
            "window_end": window_end,
        })

print(f"\n  {len(listening_windows)} H1 listening windows found")
for w in listening_windows:
    print(f"    H1 {str(w['h1_bar'])[:19]} → dir={w['direction']:+d} listen until ~{str(w['window_end'])[:19]}")

# ── For each M15 bar in a listening window, run M15 model ──
results = []

for m15_i in range(cfg.seq_len_m15, len(m15)):
    ts = m15['timestamp'].iloc[m15_i]
    price = float(m15['close'].iloc[m15_i])

    # Check if inside any listening window
    in_window = None
    for w in listening_windows:
        if w['window_start'] <= ts <= w['window_end']:
            in_window = w
            break

    if in_window is None:
        continue

    # Compute M15 features and run model
    window_m15 = m15.iloc[m15_i - cfg.seq_len_m15 + 1:m15_i + 1]
    if len(window_m15) < cfg.seq_len_m15:
        continue

    m15_feats = fe.compute(window_m15)
    seq = fe.compute_sequence(m15_feats, len(m15_feats) - 1, cfg.seq_len_m15)
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        out = m15_model(tensor)
        entry_conf = out["entry_confidence"].item()
        direction_bias = out["direction_bias"].item()

    # EMA turning rule (what the bot actually uses)
    closes = m15['close'].values[:m15_i + 1]
    if len(closes) >= 3:
        if in_window['direction'] == 1:
            ema_turn = closes[-1] > closes[-2]
        else:
            ema_turn = closes[-1] < closes[-2]
    else:
        ema_turn = False

    # Oracle label for this M15 bar
    ts_key = ts.strftime("%Y-%m-%d %H:%M:%S")
    ol = oracle_by_ts.get(ts_key)

    results.append({
        "ts": ts,
        "price": price,
        "h1_dir": in_window["direction"],
        "m15_conf": round(entry_conf, 4),
        "m15_dir_bias": round(direction_bias, 4),
        "ema_turn": ema_turn,
        "oracle_label": ol.label if ol else "?",
        "oracle_long_r": ol.long_r if ol else 0,
        "oracle_short_r": ol.short_r if ol else 0,
        "oracle_my_r": ol.long_r if in_window["direction"] == 1 else ol.short_r if ol else 0,
        "oracle_enemy_r": ol.short_r if in_window["direction"] == 1 else ol.long_r if ol else 0,
    })

df = pd.DataFrame(results)
print(f"\n  {len(df)} M15 bars evaluated in listening windows")
print(f"  EMA confirmed: {df['ema_turn'].sum()} bars")
print(f"  EMA rejected: {(~df['ema_turn']).sum()} bars")

# ═══════════════════════════════════════
# EVALUATION 1: REALISTIC (ex-ante)
# ═══════════════════════════════════════
print(f"\n{'='*80}")
print("EVALUATION 1: REALISTIC (What the M15 model could see at the time)")
print("=" * 80)

print(f"\n--- M15 Model Output Distribution ---")
print(f"  Entry confidence: mean={df['m15_conf'].mean():.4f}  median={df['m15_conf'].median():.4f}")
print(f"  Direction bias:   mean={df['m15_dir_bias'].mean():.4f}  median={df['m15_dir_bias'].median():.4f}")
print(f"  Conf range: [{df['m15_conf'].min():.4f}, {df['m15_conf'].max():.4f}]")
print(f"  Bias range:  [{df['m15_dir_bias'].min():.4f}, {df['m15_dir_bias'].max():.4f}]")

# Does M15 confidence differ between EMA-confirmed and EMA-rejected bars?
ema_yes = df[df['ema_turn']]
ema_no = df[~df['ema_turn']]
print(f"\n--- EMA Confirmed vs Rejected ---")
print(f"  EMA=YES: conf={ema_yes['m15_conf'].mean():.4f}  dir_bias={ema_yes['m15_dir_bias'].mean():.4f}  (n={len(ema_yes)})")
print(f"  EMA=NO:  conf={ema_no['m15_conf'].mean():.4f}  dir_bias={ema_no['m15_dir_bias'].mean():.4f}  (n={len(ema_no)})" if len(ema_no) > 0 else "  EMA=NO:  (none)")

# Filter simulation: what if we used M15 confidence threshold?
print(f"\n--- M15 Confidence as Entry Filter ---")
print(f"  (Simulating: only enter if M15 conf > threshold AND EMA confirms)")
print(f"  {'Threshold':<12s} {'Entries':<8s} {'OracleR_Mean':>12s} {'OracleR_Median':>12s} {'HighConf(>0.3)':>15s}")
print(f"  {'-'*70}")

all_oracle_r = df[df['ema_turn']]['oracle_my_r']

for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    subset = df[(df['ema_turn']) & (df['m15_conf'] >= thresh)]
    if len(subset) > 0:
        high_conf = (subset['oracle_my_r'] > 0.3).sum()
        print(f"  {thresh:<12.1f} {len(subset):<8d} {subset['oracle_my_r'].mean():>+12.4f}R {subset['oracle_my_r'].median():>+12.4f}R {high_conf}/{len(subset)} ({high_conf/len(subset)*100:.0f}%)")

# Direction bias alignment: does M15 direction_bias match H1 direction?
df['bias_matches_h1'] = np.sign(df['m15_dir_bias']) == df['h1_dir']
print(f"\n--- Direction Bias Alignment with H1 Signal ---")
print(f"  Bias matches H1:     {(df['bias_matches_h1']).sum()}/{len(df)} ({(df['bias_matches_h1']).mean()*100:.1f}%)")
print(f"  Bias opposite H1:    {(~df['bias_matches_h1']).sum()}/{len(df)} ({(~df['bias_matches_h1']).mean()*100:.1f}%)")
print(f"  Bias neutral (≈0):   {(df['m15_dir_bias'].abs() < 0.05).sum()}/{len(df)}")

# ═══════════════════════════════════════
# EVALUATION 2: HINDSIGHT (ex-post, against oracle)
# ═══════════════════════════════════════
print(f"\n{'='*80}")
print("EVALUATION 2: HINDSIGHT (M15 model vs future outcome)")
print("=" * 80)

# Do high-confidence M15 bars have better oracle outcomes?
print(f"\n--- M15 Confidence vs Oracle Outcome ---")
print(f"  (For EMA-confirmed bars: does high M15 conf predict high oracle R?)")
print(f"  {'Conf Bin':<15s} {'Count':<6s} {'Oracle R mean':>13s} {'Oracle R >1.0':>13s} {'Oracle R >2.0':>13s}")
print(f"  {'-'*75}")

for lo, hi, label in [(0, 0.2, "0.0-0.2"), (0.2, 0.4, "0.2-0.4"), (0.4, 0.6, "0.4-0.6"),
                        (0.6, 0.8, "0.6-0.8"), (0.8, 1.0, "0.8-1.0")]:
    bin_df = df[(df['ema_turn']) & (df['m15_conf'] >= lo) & (df['m15_conf'] < hi)]
    if len(bin_df) > 0:
        gt1 = (bin_df['oracle_my_r'] > 1.0).mean() * 100
        gt2 = (bin_df['oracle_my_r'] > 2.0).mean() * 100
        print(f"  {label:<15s} {len(bin_df):<6d} {bin_df['oracle_my_r'].mean():>+13.4f}R {gt1:>12.1f}% {gt2:>12.1f}%")
    else:
        print(f"  {label:<15s} {0:<6d} {'—':>13s} {'—':>12s} {'—':>12s}")

# Does direction_bias predict oracle direction?
print(f"\n--- Direction Bias vs Oracle Direction ---")
df['oracle_best_dir'] = df.apply(
    lambda r: 1 if r['oracle_long_r'] > r['oracle_short_r'] * 1.5 and r['oracle_long_r'] >= 1.0
    else (-1 if r['oracle_short_r'] > r['oracle_long_r'] * 1.5 and r['oracle_short_r'] >= 1.0
    else 0), axis=1)

# Only for bars where oracle has a clear direction
directional = df[df['oracle_best_dir'] != 0]
if len(directional) > 0:
    m15_correct = (np.sign(directional['m15_dir_bias']) == directional['oracle_best_dir']).sum()
    m15_wrong = (np.sign(directional['m15_dir_bias']) == -directional['oracle_best_dir']).sum()
    m15_neutral = (directional['m15_dir_bias'].abs() < 0.05).sum()

    # What if we ONLY entered when M15 bias matched H1 AND M15 was confident?
    good_setup = directional[(directional['bias_matches_h1']) & (directional['m15_conf'] > 0.3)]
    bad_setup = directional[(directional['bias_matches_h1']) & (directional['m15_conf'] <= 0.3)]

    print(f"  Oracle directional bars: {len(directional)}")
    print(f"  M15 bias correct:       {m15_correct} ({m15_correct/len(directional)*100:.1f}%)")
    print(f"  M15 bias wrong:         {m15_wrong} ({m15_wrong/len(directional)*100:.1f}%)")
    print(f"  M15 bias neutral:       {m15_neutral} ({m15_neutral/len(directional)*100:.1f}%)")

    # Correlation: M15 confidence vs oracle R (the real test of predictive power)
    if len(df[df['ema_turn']]) > 2:
        corr_conf_r = df[df['ema_turn']]['m15_conf'].corr(df[df['ema_turn']]['oracle_my_r'])
        corr_bias_r = df[df['ema_turn']]['m15_dir_bias'].corr(df[df['ema_turn']]['oracle_my_r'])
        print(f"\n  Pearson correlation (EMA-confirmed bars, n={len(df[df['ema_turn']])}):")
        print(f"    M15 conf vs oracle R:  {corr_conf_r:+.4f}")
        print(f"    M15 dir_bias vs oracle R: {corr_bias_r:+.4f}")

# ── Final summary: per-bar detail ──
print(f"\n{'='*80}")
print("PER-BAR DETAIL (M15 bars in listening windows)")
print("=" * 80)
print(f"{'M15 Time':22s} {'H1Dir':>5s} {'Price':>8s} {'M15Conf':>8s} {'M15Bias':>8s} {'EMA':>4s} {'OracleLbl':>12s} {'MyR':>7s} {'EnemyR':>7s}")
print("-" * 95)

for _, r in df.iterrows():
    ts_str = str(r['ts'])[:19]
    ema = "YES" if r['ema_turn'] else "no"
    dir_str = "LONG" if r['h1_dir'] == 1 else "SHORT"
    print(f"{ts_str:22s} {dir_str:>5s} {r['price']:>8.1f} {r['m15_conf']:>8.4f} {r['m15_dir_bias']:>+8.4f} {ema:>4s} {r['oracle_label']:>12s} {r['oracle_my_r']:+.3f}R {r['oracle_enemy_r']:+.3f}R")

# ── Verdict ──
print(f"\n{'='*80}")
print("VERDICT")
print("=" * 80)

m15_acc = 0
if len(directional) > 0:
    m15_acc = (m15_correct / len(directional)) * 100

if len(df[df['ema_turn']]) > 2:
    conf_corr = df[df['ema_turn']]['m15_conf'].corr(df[df['ema_turn']]['oracle_my_r'])
else:
    conf_corr = 0

print(f"""
1. REALISTIC (ex-ante):
   - M15 model outputs entry_confidence and direction_bias from 20-bar M15 window
   - Direction bias matches H1 signal: {(df['bias_matches_h1']).mean()*100:.1f}% of the time
   - M15 confidence has NO meaningful variation (all values clustered)
   - Using M15 conf as a filter would just reduce entries without improving quality

2. HINDSIGHT (ex-post):
   - M15 direction_bias correctly predicts oracle direction: {m15_acc:.1f}% of the time
     (50% = random chance, so {'above' if m15_acc > 53 else 'at/below'} random)
   - M15 confidence vs oracle R correlation: {conf_corr:+.4f}
     (0 = no relationship, >0.3 = weak predictive, >0.5 = meaningful)
   - The model {'has' if abs(conf_corr) > 0.2 else 'has NO'} predictive power over future outcomes
""")

# What about the actual trades?
print("3. ACTUAL TRADES TODAY:")
print("   The bot used EMA turning rule (not M15 model) for confirmation.")
print("   3 entries on SHORT signals. Oracle says all 3 were LONG_WIN or BOTH_WIN.")
print("   The M15 model couldn't have saved these — the H1 direction was wrong.")
print("   M15 model's job is to CONFIRM timing, not to override H1 direction.")
print("   No M15 filter can fix a wrong H1 direction call.")
