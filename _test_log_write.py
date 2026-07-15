"""Test if _log_h1_eval can write to the log file."""
import sys, os, json
sys.path.insert(0, "D:/FiananceBot/BTC_BOT")
from config_btc import BTCConfig
from models.entry_gate import GateDecision

cfg = BTCConfig()
log_path = os.path.join(cfg.log_dir, "h1_eval_test.jsonl")

# Simulate exact same data as _log_h1_eval
import numpy as np

ts = "2026-05-26 17:00:00+00:00"
price = 77172.6
atr = 325.39
model_regime = "TREND_DOWN"
model_conf = 0.9614
model_probs = [0.0021, 0.9614, 0.0043, 0.0322]
model_raw_logits = [-11.9, 12.6, -9.1, -1.0]
model_used = True
rule_regime = "TREND_DOWN"
rule_conf = 0.2828
rule_ema_slope = -0.000848
rule_atr_pct = 0.5
conflict_fired = False
final_regime_result = {"regime": "TREND_DOWN", "confidence": 0.9614, "source": "model"}
gate = GateDecision(entry_signal=True, direction=-1, confidence=0.9614,
                    model_used="trend", reason="trend signal")
h4_blocked = False
h4_regime = ""

entry = {
    "ts": str(ts),
    "price": round(price, 2),
    "atr": round(atr, 2),
    "model": {
        "regime": model_regime,
        "confidence": round(model_conf, 4),
        "used_for_final": model_used,
        "class_probs": [round(p, 4) for p in model_probs],
        "raw_logits": [round(l, 1) for l in model_raw_logits],
    },
    "rule": {
        "regime": rule_regime,
        "confidence": round(rule_conf, 4),
        "ema_slope": round(rule_ema_slope, 6),
        "atr_percentile": round(rule_atr_pct, 4),
    },
    "conflict": {"fired": conflict_fired},
    "final": {
        "regime": final_regime_result["regime"],
        "confidence": round(final_regime_result["confidence"], 4),
        "source": final_regime_result.get("source", "?"),
    },
    "gate": {
        "signal": gate.entry_signal,
        "direction": gate.direction,
        "confidence": round(gate.confidence, 4),
        "model_used": gate.model_used,
        "reason": gate.reason,
    },
    "h4": {"blocked": h4_blocked, "regime": h4_regime},
}

try:
    json_str = json.dumps(entry)
    print(f"JSON serialization OK: {len(json_str)} chars")
    with open(log_path, 'a') as f:
        f.write(json_str + "\n")
    print(f"Write OK to {log_path}")
    print(f"Content: {json_str[:200]}...")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# Also test with NaN
print("\n--- Testing with NaN ---")
model_probs_nan = [float('nan'), 0.9614, 0.0043, 0.0322]
entry_nan = dict(entry)
entry_nan["model"]["class_probs"] = model_probs_nan
try:
    json_str = json.dumps(entry_nan)
    print("JSON with NaN: OK (unexpected)")
except Exception as e:
    print(f"JSON with NaN: FAILED - {type(e).__name__}: {e}")

# Test with inf
print("\n--- Testing with Inf ---")
model_raw_logits_inf = [float('inf'), 12.6, -9.1, -1.0]
entry_inf = dict(entry)
entry_inf["model"]["raw_logits"] = model_raw_logits_inf
try:
    json_str = json.dumps(entry_inf)
    print("JSON with Inf: OK (unexpected)")
except Exception as e:
    print(f"JSON with Inf: FAILED - {type(e).__name__}: {e}")

# Test with numpy float
print("\n--- Testing with numpy types ---")
entry_np = dict(entry)
entry_np["model"]["confidence"] = np.float32(0.9614)
try:
    json_str = json.dumps(entry_np)
    print("JSON with numpy float32: OK (unexpected)")
except Exception as e:
    print(f"JSON with numpy float32: FAILED - {type(e).__name__}: {e}")
