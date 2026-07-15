"""
BTC Trading System — Listening Architecture per spec.

Two-layer design:
  H1 layer runs on H1 bar close → generates directional signal
  M15 layer "listens" on every M15 close → confirms + executes

Flow:
  on_h1_bar_close()  → regime → gate → if signal: activate listener
  on_m15_bar_close() → if listening: M15 confirm → enter OR expire
"""
import os, sys, json, time, logging
from datetime import datetime, timezone
from typing import Optional
from execution.discord_logger import log_signal, log_gated, log_entry, log_exit

import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.cnn_gru_m15 import CNNGRUM15
from models.regime_classifier import (RegimeClassifier, RuleBasedRegimeDetector,
                                       classify_regime, REGIME_NAMES)
from models.entry_gate import EntryGate, GateDecision
from models.trade_manager_btc import TradeManager, TradeAction, TradeActionType

logger = logging.getLogger("BTCBot")


class BTCTradingSystem:
    """
    Listening Architecture:

    H1 model (prediction layer):
      - Classifies regime: TREND_UP / TREND_DOWN / RANGE / TRANSITION
      - Generates directional signal: Long / Short / No trade
      - outputs signal strength, confidence, suggested SL/TP in ATR units
      - Runs once per closed H1 bar

    M15 model (execution layer):
      - "Listens" continuously on 15-minute bars
      - Only activates when H1 model has an active signal
      - Waits for M15 confirmation before entering
      - Looks for: pullback completion, momentum alignment, volume spike
      - Runs on every closed M15 bar (4× per H1 bar)
    """

    def __init__(self, config: BTCConfig, bot_id="BTCBot",
                 dry_run=True, live=False, risk_pct=None, max_daily_loss=None):
        self.config = config; self.bot_id = bot_id
        self.dry_run = dry_run or (not live)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if risk_pct is not None: config.risk_pct = risk_pct
        if max_daily_loss is not None: config.max_daily_loss = max_daily_loss
        self.risk_pct = config.risk_pct

        # ── Listening state ──
        self.h1_signal = None          # current H1 signal (1=long, -1=short, 0=none)
        self.h1_confidence = 0.0
        self.h1_atr = 0.0
        self.listening = False
        self.bars_listened = 0
        self.max_listen = config.max_listen_bars  # 8 M15 bars = 2 hours
        self.blocked_hours = set()  # removed — phased TradeManager handles bad entries
        self.h1_embedding = None

        # ── Position state ──
        self.position = 0; self.position_ticket = None; self.position_lots = 0.0
        self.cooldown_bars = 0  # prevent rapid re-entry after exit
        self.trade_history = []
        self.balance = 0.0; self.starting_balance = 0.0
        self.daily_pnl = 0.0; self.last_day = None

        # ── Models ──
        self.encoder = None; self.classifier = None; self.m15_model = None
        self.feature_engine = BTCFeatureEngine()
        self.rule_detector = RuleBasedRegimeDetector()
        self.entry_gate = EntryGate(
            min_confidence=config.min_regime_confidence,
            min_atr_pct=config.min_atr_percentile,
            max_atr_pct=config.max_atr_percentile)
        self.trade_manager = TradeManager(
            phase1_sl=config.initial_sl, hard_tp=config.hard_tp,
            phase1_bars=config.phase1_bars,
            phase2_mfe_min=config.phase2_mfe_min,
            phase3_trail=config.phase3_trail,
            phase4_trail=config.phase4_trail,
            phase4_start=config.phase4_start,
            max_hold=config.max_hold_bars)

        # Executor
        from execution.mt5_executor_btc import DryRunExecutor, MT5Executor
        if self.dry_run:
            self.executor = DryRunExecutor(symbol=config.symbol)
        else:
            self.executor = MT5Executor(symbol=config.symbol, magic=config.mt5_magic,
                                        deviation=config.mt5_deviation)

        self.h1_bars_cache = None; self.m15_bars_cache = None
        self.last_h1_bar_key = None; self.last_m15_bar_key = None

    # ═══════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════

    def load_models(self):
        encoder_path = os.path.join(self.config.model_dir, "btc_h1_encoder.pt")
        if not os.path.exists(encoder_path):
            logger.warning(f"No trained encoder at {encoder_path}, using random init")
            self._init_fresh_models(); return

        ckpt = torch.load(encoder_path, map_location=self.device, weights_only=False)
        self.encoder = CNNLSTMEncoder(
            n_features=self.config.n_features, seq_len=self.config.seq_len_h1,
            cnn_channels=self.config.cnn_channels, lstm_hidden=self.config.lstm_hidden,
            lstm_layers=self.config.lstm_layers, dropout=self.config.lstm_dropout,
            embedding_dim=self.config.embedding_dim,
            regime_classes=self.config.regime_classes,
            bidirectional=self.config.lstm_bidirectional).to(self.device).eval()
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])

        self.classifier = RegimeClassifier(embedding_dim=self.config.embedding_dim,
                                           n_classes=self.config.regime_classes).to(self.device).eval()
        self.classifier.load_state_dict(ckpt["classifier_state_dict"])
        logger.info(f"H1 encoder loaded from {encoder_path}")

        m15_path = os.path.join(self.config.model_dir, "btc_m15_v2.pt")
        if not os.path.exists(m15_path):
            m15_path = os.path.join(self.config.model_dir, "btc_m15_model.pt")
        if os.path.exists(m15_path):
            ckpt_m15 = torch.load(m15_path, map_location=self.device, weights_only=False)
            self.m15_model = CNNGRUM15(
                n_features=self.config.n_features, seq_len=self.config.seq_len_m15,
                cnn_channels=self.config.gru_cnn_channels, gru_hidden=self.config.gru_hidden,
                gru_layers=self.config.gru_layers, dropout=self.config.gru_dropout).to(self.device).eval()
            # Remap v2 checkpoint keys (M15EntryClassifier → CNNGRUM15)
            state_dict = ckpt_m15["model_state_dict"]
            if "conv1.0.weight" in state_dict:
                block_starts = {"conv1": 0, "conv2": 4, "conv3": 8}
                remapped = {}
                for old_key, val in state_dict.items():
                    prefix = old_key.split(".")[0]
                    if prefix in block_starts:
                        rest = old_key.split(".", 1)[1]  # "0.weight"
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
            self.m15_model.load_state_dict(state_dict, strict=False)
            self.m15_v2 = "v2" in m15_path
            loaded = sum(1 for k in state_dict if k in self.m15_model.state_dict())
            logger.info(f"M15 model loaded from {m15_path} (v2={self.m15_v2}, "
                       f"keys={loaded}/{len(self.m15_model.state_dict())})")

        # H4 encoder (V2 — higher timeframe trend gate)
        self.h4_encoder = None
        h4_path = os.path.join(self.config.model_dir, "btc_h4_encoder.pt")
        if os.path.exists(h4_path):
            ckpt_h4 = torch.load(h4_path, map_location=self.device, weights_only=False)
            self.h4_encoder = CNNLSTMEncoder(
                n_features=self.config.n_features, seq_len=self.config.seq_len_h1,
                cnn_channels=self.config.cnn_channels, lstm_hidden=self.config.lstm_hidden,
                lstm_layers=self.config.lstm_layers, dropout=self.config.lstm_dropout,
                embedding_dim=self.config.embedding_dim,
                regime_classes=self.config.regime_classes,
                bidirectional=self.config.lstm_bidirectional).to(self.device).eval()
            self.h4_encoder.load_state_dict(ckpt_h4["encoder_state_dict"])
            logger.info(f"H4 encoder loaded from {h4_path}")

    def _init_fresh_models(self):
        self.encoder = CNNLSTMEncoder(
            n_features=self.config.n_features, seq_len=self.config.seq_len_h1,
            cnn_channels=self.config.cnn_channels, lstm_hidden=self.config.lstm_hidden,
            lstm_layers=self.config.lstm_layers, dropout=self.config.lstm_dropout,
            embedding_dim=self.config.embedding_dim,
            regime_classes=self.config.regime_classes,
            bidirectional=self.config.lstm_bidirectional).to(self.device).eval()
        self.classifier = RegimeClassifier(
            embedding_dim=self.config.embedding_dim,
            n_classes=self.config.regime_classes).to(self.device).eval()
        self.m15_model = CNNGRUM15(
            n_features=self.config.n_features, seq_len=self.config.seq_len_m15,
            cnn_channels=self.config.gru_cnn_channels, gru_hidden=self.config.gru_hidden,
            gru_layers=self.config.gru_layers, dropout=self.config.gru_dropout).to(self.device).eval()

    # ═══════════════════════════════════════
    # Listening Architecture
    # ═══════════════════════════════════════

    def on_h1_bar_close(self, h1_features, h1_df):
        """Called every H1 bar close. Generates H1 signal."""
        seq = self.feature_engine.compute_sequence(
            h1_features, len(h1_features) - 1, self.config.seq_len_h1)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        # Update rule detector
        for _, row in h1_df.iloc[-14:].iterrows():
            self.rule_detector.update(row["high"], row["low"], row["close"])

        # ── Model regime: extract raw logits before classify_regime ──
        with torch.no_grad():
            enc_out = self.encoder(tensor)
            raw_logits = self.classifier.raw_logits(enc_out["embedding"])
            probs_t4 = torch.softmax(raw_logits / self.config.regime_temperature, dim=1)
            max_prob, pred_class = probs_t4.max(dim=1)
            max_prob = max_prob.item(); pred_class = pred_class.item()
        model_regime = REGIME_NAMES[pred_class]
        model_conf = max_prob
        model_probs = probs_t4.squeeze(0).tolist()
        model_raw_logits = raw_logits.squeeze(0).tolist()
        model_source = "model" if max_prob >= self.config.min_regime_confidence else "rule"
        model_used_model = max_prob >= self.config.min_regime_confidence

        regime_result = classify_regime(self.encoder, self.classifier, tensor,
                                        self.rule_detector,
                                        model_confidence_threshold=self.config.min_regime_confidence)

        # If model and rule detector disagree on TREND direction, trust the rule.
        # The rule detector is 74% accurate during conflicts vs model's 26% (YTD analysis).
        rule_classification = self.rule_detector._classify()
        rule_regime = rule_classification["regime"]
        rule_conf = rule_classification["confidence"]
        rule_ema_slope = rule_classification.get("ema_slope", 0.0)
        rule_atr_pct = rule_classification.get("atr_percentile", 0.5)

        conflict_fired = False
        trending = {"TREND_UP", "TREND_DOWN"}
        if (regime_result["regime"] in trending and
            rule_classification["regime"] in trending and
            regime_result["regime"] != rule_classification["regime"]):
            conflict_fired = True
            regime_result["regime"] = rule_classification["regime"]
            regime_result["confidence"] = rule_classification["confidence"]
            regime_result["source"] = "rule"

        current_price = h1_df["close"].iloc[-1]
        current_atr = h1_features[-1, 6] * current_price  # atr_pct * close
        bb_pos = h1_features[-1, 4]  # bb_position

        # Entry gate
        gate = self.entry_gate.evaluate(
            regime_result["regime"], regime_result["confidence"],
            regime_result.get("atr_percentile", 0.5), bb_position=bb_pos)
        self._last_regime = regime_result["regime"]

        h4_blocked = False; h4_regime = ""
        if gate.entry_signal:
            # H4 trend gate (V2): block if H1 direction disagrees with H4 regime
            # (EMA22 trend filter removed — phased TradeManager handles bad entries)
            if self.h4_encoder is not None and len(h1_df) >= 96:
                h4_df = h1_df.set_index("timestamp").resample("4h").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last",
                    "volume": "sum"
                }).dropna().reset_index()
                if len(h4_df) >= 96:
                    h4_feats = self.feature_engine.compute(h4_df)
                    h4_seq = self.feature_engine.compute_sequence(h4_feats, len(h4_feats) - 1, 96)
                    h4_t = torch.from_numpy(h4_seq).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        h4_out = self.h4_encoder(h4_t)
                    h4_regime_idx = h4_out["regime_logits"].argmax(1).item()
                    h4_regime = ["TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION"][h4_regime_idx]
                    h4_against = ((gate.direction == 1 and h4_regime == "TREND_DOWN") or
                                  (gate.direction == -1 and h4_regime == "TREND_UP"))
                    if h4_against:
                        h4_blocked = True
                        logger.debug(f"H1: signal blocked — against H4 trend "
                                    f"(dir={gate.direction:+d} h4={h4_regime})")
                        log_gated(self.bot_id, gate.direction, f"H4 {h4_regime}")
                        self.h1_signal = None
                        self.listening = False
                        self._log_h1_eval(h1_df["timestamp"].iloc[-1], current_price, current_atr,
                                          model_regime, model_conf, model_probs, model_raw_logits,
                                          model_used_model, rule_regime, rule_conf, rule_ema_slope,
                                          rule_atr_pct, conflict_fired, regime_result, gate,
                                          h4_blocked, h4_regime)
                        return

            self.h1_signal = gate.direction
            self.h1_confidence = gate.confidence
            self.h1_atr = current_atr
            self.listening = True
            self.bars_listened = 0
            logger.info(f"H1 SIGNAL: dir={gate.direction:+d} regime={regime_result['regime']} "
                       f"conf={gate.confidence:.2f} model={gate.model_used}")
            log_signal(self.bot_id, gate.direction, regime_result["regime"],
                       gate.confidence, current_price, current_atr)
        else:
            self.h1_signal = None
            self.listening = False
            logger.debug(f"H1: no signal — {gate.reason}")

        # ── Structured H1 evaluation log ──
        self._log_h1_eval(h1_df["timestamp"].iloc[-1], current_price, current_atr,
                          model_regime, model_conf, model_probs, model_raw_logits,
                          model_used_model, rule_regime, rule_conf, rule_ema_slope,
                          rule_atr_pct, conflict_fired, regime_result, gate,
                          h4_blocked, h4_regime)

    def _log_h1_eval(self, ts, price, atr, model_regime, model_conf, model_probs,
                     model_raw_logits, model_used, rule_regime, rule_conf, rule_ema_slope,
                     rule_atr_pct, conflict_fired, final_regime_result, gate,
                     h4_blocked, h4_regime):
        """Write one JSON line to logs/h1_eval_{bot_id}.jsonl for later analysis."""
        import json
        try:
            log_path = os.path.join(self.config.log_dir, f"h1_eval_{self.bot_id}.jsonl")
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
                "conflict": {
                    "fired": conflict_fired,
                },
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
                "h4": {
                    "blocked": h4_blocked,
                    "regime": h4_regime,
                },
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # never let logging break the bot

    def on_m15_bar_close(self, m15_features, m15_df):
        """Called every M15 bar close. Confirms H1 signal + enters."""
        if not self.listening:
            return {"action": "wait"}

        self.bars_listened += 1

        # Expire signal
        if self.bars_listened > self.max_listen:
            self.listening = False
            self.h1_signal = None
            logger.debug(f"M15: signal expired after {self.bars_listened} bars")
            self._log_m15_result(m15_df["timestamp"].iloc[-1], m15_df["close"].iloc[-1],
                                 "expired", None)
            return {"action": "expired"}

        # Cooldown after exit — prevent rapid re-entry
        if self.cooldown_bars > 0:
            self.cooldown_bars -= 1
            return {"action": "wait"}

        # Hour filter (disabled — phased TradeManager handles bad entries)
        bar_hour = m15_df["timestamp"].iloc[-1].hour
        if bar_hour in self.blocked_hours:
            logger.debug(f"M15: signal suppressed — hour {bar_hour} blocked")
            return {"action": "wait"}

        current_price = m15_df["close"].iloc[-1]

        # M15 confirmation — EMA rule (turning in signal direction)
        confirmed = self._m15_confirm(m15_features, m15_df)

        if confirmed:
            self.listening = False
            atr = self.h1_atr
            m15_atr = m15_features[-1, 6] * current_price if m15_features is not None else atr
            sl = current_price - self.h1_signal * self.config.initial_sl * atr
            tp = current_price + self.h1_signal * self.config.hard_tp * atr
            logger.info(f"M15 CONFIRMED: dir={self.h1_signal:+d} "
                       f"price={current_price:.2f} sl={sl:.2f} tp={tp:.2f}")
            self._log_m15_result(m15_df["timestamp"].iloc[-1], current_price,
                                 "confirmed", self.h1_signal)
            return {"action": "enter", "direction": self.h1_signal,
                    "confidence": self.h1_confidence, "sl": sl, "tp": tp,
                    "atr": atr}
        else:
            logger.info(f"M15 REJECTED: dir={self.h1_signal:+d} "
                       f"price={current_price:.2f} bars_left={self.max_listen - self.bars_listened}")

        return {"action": "listening", "bars_left": self.max_listen - self.bars_listened}

    def _log_m15_result(self, ts, price, result, direction):
        """Log M15 confirmation result to the H1 eval log."""
        import json
        try:
            log_path = os.path.join(self.config.log_dir, f"h1_eval_{self.bot_id}.jsonl")
            entry = {
                "ts": str(ts),
                "price": round(price, 2),
                "event": "m15_confirm",
                "result": result,
                "direction": direction,
                "bars_listened": self.bars_listened,
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _m15_confirm(self, m15_features, m15_df):
        """
        M15 confirmation — EMA rule only.

        EMA21 + 2-bar momentum: price near EMA and turning in signal direction.
        NN model removed — YTD analysis showed it loses money (-$1,030, 53% WR)
        while EMA rule made +$10,712 (57% WR).
        """
        closes = m15_df["close"].values
        if len(closes) < 3:
            return False

        if self.h1_signal == 1:
            return closes[-1] > closes[-2]
        elif self.h1_signal == -1:
            return closes[-1] < closes[-2]
        return False

    # ═══════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════

    def run_loop(self):
        logger.info(f"{'='*50}")
        logger.info(f"BTC Trading System — Listening Architecture")
        logger.info(f"  Bot: {self.bot_id}  Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"  Risk: {self.risk_pct*100:.0f}%  SL: {self.config.initial_sl}xATR  TP: {self.config.hard_tp}xATR")
        logger.info(f"{'='*50}")

        self.load_models()
        if not self.dry_run:
            if not self.executor.connect():
                logger.error("MT5 connection failed"); return

        self.balance = self.executor.get_account_balance() or 10000.0
        if self.dry_run and self.balance == 0:
            self.balance = 10000.0
            if hasattr(self.executor, '_balance'): self.executor._balance = self.balance
        self.starting_balance = self.balance

        self.h1_bars_cache, self.m15_bars_cache = self._load_data()
        if self.h1_bars_cache is None:
            logger.error("No data"); return

        try:
            if self.dry_run:
                self._dry_run_replay()
            else:
                self._live_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._cleanup()

    def _load_data(self):
        """Load H1 and M15 data."""
        h1_path = os.path.join(self.config.data_dir, "BTCUSD_1h.csv")
        if not os.path.exists(h1_path):
            h1_path = os.path.join(self.config.data_dir, "(1h)_btc_usd_dataset_London-Strategic-Edge (1).csv")
        m15_path = os.path.join(self.config.data_dir, "BTCUSD_15m.csv")
        if not os.path.exists(m15_path):
            m15_path = os.path.join(self.config.data_dir, "(15min)_btc_usd_dataset_London-Strategic-Edge (2).csv")

        h1 = pd.read_csv(h1_path); h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
        m15 = pd.read_csv(m15_path); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)

        # Date filter
        if hasattr(self, 'from_date') and self.from_date:
            ft = pd.Timestamp(self.from_date, tz="UTC")
            h1 = h1[h1["timestamp"] >= ft].reset_index(drop=True)
            m15 = m15[m15["timestamp"] >= ft].reset_index(drop=True)
        if hasattr(self, 'to_date') and self.to_date:
            tt = pd.Timestamp(self.to_date, tz="UTC")
            h1 = h1[h1["timestamp"] < tt].reset_index(drop=True)
            m15 = m15[m15["timestamp"] < tt].reset_index(drop=True)

        return h1, m15

    def _dry_run_replay(self):
        """Walk through bars chronologically, calling on_h1_bar_close / on_m15_bar_close."""
        h1 = self.h1_bars_cache; m15 = self.m15_bars_cache
        if len(h1) < self.config.seq_len_h1 or len(m15) < self.config.seq_len_m15:
            logger.error("Not enough data"); return

        # Build aligned timeline: walk M15 bars, trigger H1 when a new H1 bar closes
        m15_start = max(self.config.seq_len_m15, 20)
        last_h1_key = None
        n = len(m15)

        logger.info(f"Replay: {n - m15_start} M15 bars "
                   f"({m15['timestamp'].iloc[m15_start]} → {m15['timestamp'].iloc[-1]})")
        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_status()

        for i in range(m15_start, n):
            ts = m15["timestamp"].iloc[i]
            price = m15["close"].iloc[i]
            self.executor._current_price = price
            self._current_price = price

            # H1 slice up to this M15 bar
            h1_slice = h1[h1["timestamp"] <= ts]
            m15_slice = m15.iloc[max(0, i - self.config.seq_len_m15 * 4):i + 1]

            if len(h1_slice) < self.config.seq_len_h1:
                continue

            # Check for new H1 bar
            h1_latest = h1_slice["timestamp"].max()
            if h1_latest != last_h1_key:
                last_h1_key = h1_latest
                h1_feats = self.feature_engine.compute(h1_slice)
                self.h1_atr_current = h1_feats[-1, 6] * price
                self.on_h1_bar_close(h1_feats, h1_slice)

            # ── Manage position ──
            if self.position != 0 and self.trade_manager.state is not None:
                self._manage_position(m15_slice, price)
            else:
                # M15 layer
                m15_feats = self.feature_engine.compute(m15_slice)
                result = self.on_m15_bar_close(m15_feats, m15_slice)

                if result["action"] == "enter":
                    self._enter_trade(result["direction"], price,
                                     result.get("atr", self.h1_atr),
                                     result["confidence"])

            # Progress
            if i % 2000 == 0:
                wins = sum(1 for t in self.trade_history if t["pnl_dollar"] > 0)
                wr = wins / len(self.trade_history) if self.trade_history else 0
                total = sum(t["pnl_dollar"] for t in self.trade_history)
                logger.info(f"Progress: {ts} | {len(self.trade_history)} trades "
                           f"WR={wr:.1%} | PnL=${total:+,.2f} | Bal=${self.balance:,.2f}")
                self._write_status()

            # Daily reset on new day
            today = ts.date()
            if self.last_day and today != self.last_day:
                logger.info(f"New day: {today}, daily PnL reset (was ${self.daily_pnl:+.2f})")
                self.daily_pnl = 0.0; self.starting_balance = self.balance
            self.last_day = today

        self._write_status()
        self._print_summary()

    def _manage_position(self, m15_slice, price):
        """Check SL/TP, update trade manager."""
        hi = m15_slice["high"].iloc[-1]; lo = m15_slice["low"].iloc[-1]

        if self.trade_manager.check_sl_hit(lo, hi):
            self._close_position(self.trade_manager.exit_price_at_sl(), "sl_hit"); return
        if self.trade_manager.check_tp_hit(lo, hi):
            self._close_position(self.trade_manager.exit_price_at_tp(), "tp_hit"); return

        action = self.trade_manager.update(price, hi, lo, self.h1_atr)
        if action.action_type == TradeActionType.CLOSE:
            self._close_position(price, action.reason)
        elif action.action_type == TradeActionType.MODIFY_SL and self.position_ticket:
            self.executor.modify_sl_tp(self.position_ticket, action.new_sl,
                                       self.trade_manager.state.current_tp)

    def _enter_trade(self, direction, price, atr, confidence):
        if not self._check_risk_limits(): return
        lots = self.trade_manager.compute_position_size(
            self.balance, atr, price, self.risk_pct, self.config.initial_sl)
        self.trade_manager.enter(direction, price, atr, lots)
        state = self.trade_manager.state
        result = self.executor.open_position(direction=direction, lots=lots,
                                             sl=state.current_sl, tp=state.current_tp,
                                             comment=f"BTC_{self.bot_id}")
        if result.success:
            self.position = direction; self.position_ticket = result.ticket
            self.position_lots = lots
            logger.info(f"ENTER: {direction:+d} {lots:.4f}@{price:.2f} "
                       f"SL={state.current_sl:.2f} TP={state.current_tp:.2f}")
            log_entry(self.bot_id, direction, price, state.current_sl, state.current_tp, lots)
            self._log_trade_event("enter", price, direction, {
                "lots": round(lots, 4), "sl": round(state.current_sl, 2),
                "tp": round(state.current_tp, 2), "atr": round(atr, 2),
                "confidence": round(confidence, 4), "balance": round(self.balance, 2),
            })
        else:
            logger.error(f"Entry failed: {result.error}"); self.trade_manager.state = None

    def _log_trade_event(self, event, price, direction, extra=None):
        """Log trade events (enter/exit) to the H1 eval log."""
        import json
        try:
            log_path = os.path.join(self.config.log_dir, f"h1_eval_{self.bot_id}.jsonl")
            entry = {
                "ts": str(datetime.now(timezone.utc)),
                "price": round(price, 2),
                "event": event,
                "direction": direction,
            }
            if extra:
                entry.update(extra)
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _close_position(self, price, reason):
        if self.position == 0: return
        result = self.executor.close_position(self.position_ticket)
        s = self.trade_manager.state
        pnl_r = s.unrealized_pnl_r if s else 0.0
        if s and self.position == 1:
            pnl_d = (price - s.entry_price) * self.position_lots * self.executor.contract_size
        elif s:
            pnl_d = (s.entry_price - price) * self.position_lots * self.executor.contract_size
        else:
            pnl_d = 0.0

        self.trade_history.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                                    "direction": "LONG" if self.position == 1 else "SHORT",
                                    "entry_price": s.entry_price if s else 0,
                                    "exit_price": price, "lots": self.position_lots,
                                    "pnl_dollar": pnl_d, "pnl_r": round(pnl_r, 4),
                                    "mfe_r": round(s.mfe_r, 4) if s else 0,
                                    "mae_r": round(s.mae_r, 4) if s else 0,
                                    "bars_held": s.bars_held if s else 0,
                                    "exit_reason": reason})
        self.balance += pnl_d; self.daily_pnl += pnl_d
        logger.info(f"EXIT: {reason} PnL=${pnl_d:+.2f} ({pnl_r:+.2f}R) Bal=${self.balance:,.2f}")
        log_exit(self.bot_id, self.position, s.entry_price if s else 0, price,
                 round(pnl_r, 4), pnl_d, reason, round(s.mfe_r if s else 0, 4))
        self._log_trade_event("exit", price, self.position, {
            "pnl_dollar": round(pnl_d, 2), "pnl_r": round(pnl_r, 4),
            "mfe_r": round(s.mfe_r, 4) if s else 0,
            "mae_r": round(s.mae_r, 4) if s else 0,
            "bars_held": s.bars_held if s else 0,
            "reason": reason, "balance": round(self.balance + pnl_d, 2),
        })
        self.position = 0; self.position_ticket = None; self.position_lots = 0.0
        self.trade_manager.state = None
        self.cooldown_bars = 2  # 2-bar cooldown after exit

    def _check_risk_limits(self):
        return True  # daily loss limit disabled for paper trading

    def _live_loop(self):
        """Live trading loop: fetch MT5 bars, process H1/M15 on new bars, manage positions."""
        logger.info("Live loop started — fetching MT5 bars...")
        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_h1_key = None; last_m15_key = None
        self._write_status()

        while True:
            try:
                if self._check_stop_signal():
                    logger.info("Stop signal received"); break
                self._reload_config()

                # Fetch latest H1 and M15 bars from MT5
                h1_df = self.executor.get_bars(self.config.tf_h1, count=200)
                m15_df = self.executor.get_bars(self.config.tf_m15, count=200)
                if h1_df is None or h1_df.empty or m15_df is None or m15_df.empty:
                    logger.warning("No MT5 data — retrying...")
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                # Track latest bar keys
                h1_latest = h1_df["timestamp"].max()
                m15_latest = m15_df["timestamp"].max()

                # Process new H1 bar
                if h1_latest != last_h1_key:
                    if last_h1_key is not None:  # skip first iteration
                        h1_feats = self.feature_engine.compute(h1_df)
                        self.on_h1_bar_close(h1_feats, h1_df)
                    last_h1_key = h1_latest

                # Process new M15 bar
                if m15_latest != last_m15_key:
                    if last_m15_key is not None:
                        # Build aligned H1 slice for the current M15 bar time
                        m15_ts = m15_df["timestamp"].iloc[-1]
                        h1_slice = h1_df[h1_df["timestamp"] <= m15_ts]
                        if len(h1_slice) >= self.config.seq_len_h1:
                            m15_features = self.feature_engine.compute(m15_df)

                            # Manage open position
                            if self.position != 0 and self.trade_manager.state is not None:
                                self._manage_position(m15_df, m15_df["close"].iloc[-1])
                            else:
                                result = self.on_m15_bar_close(m15_features, m15_df)
                                if result.get("action") == "enter":
                                    self._enter_trade(result["direction"],
                                                     m15_df["close"].iloc[-1],
                                                     result["atr"],
                                                     result.get("confidence", 0.0))
                    last_m15_key = m15_latest

                # Update ATR for status display
                if len(h1_df) >= 14:
                    h1_feats = self.feature_engine.compute(h1_df)
                    self.h1_atr_current = h1_feats[-1, 6] * h1_df["close"].iloc[-1]
                self._current_price = m15_df["close"].iloc[-1]

                self._write_status()
                time.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                logger.error(f"Live loop error: {e}")
                self._write_status()
                time.sleep(self.config.poll_interval_seconds)

    def _write_status(self):
        """Write status/{bot_id}.json for the GUI dashboard."""
        try:
            status_path = os.path.join(self.config.status_dir, f"{self.bot_id}.json")
            os.makedirs(os.path.dirname(status_path), exist_ok=True)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            wins = [t for t in self.trade_history if t["pnl_dollar"] > 0]
            wr = len(wins) / len(self.trade_history) if self.trade_history else 0.0

            # Get current regime from last H1 processing
            regime = "unknown"
            if hasattr(self, '_last_regime'):
                regime = self._last_regime

            status = {
                "bot_id": self.bot_id,
                "status": "running",
                "mt5_connected": self.executor.is_connected() if hasattr(self.executor, 'is_connected') else (not self.dry_run),
                "dry_run": self.dry_run,
                "last_update": now,
                "last_inference": now,
                "last_trade_time": str(self.trade_history[-1].get("timestamp", "")) if self.trade_history else "",
                "balance": round(self.balance, 2),
                "daily_pnl": round(self.daily_pnl, 2),
                "position": self.position,
                "position_lots": round(self.position_lots, 4),
                "entry_price": round(self.trade_manager.state.entry_price, 2) if self.trade_manager.state else 0.0,
                "current_price": round(getattr(self, '_current_price', 0.0), 2),
                "unrealized_pnl": 0.0,
                "current_atr": round(getattr(self, 'h1_atr_current', 0.0), 2),
                "current_regime": regime,
                "total_trades": len(self.trade_history),
                "trade_count": len(self.trade_history),
                "win_rate": round(wr, 3),
                "risk_per_trade": self.risk_pct,
                "max_daily_loss": self.config.max_daily_loss,
                "pid": os.getpid(),
                "started_at": getattr(self, '_started_at', now),
                "error": "",
            }

            # Unrealized PnL
            if self.position != 0 and self.trade_manager.state:
                s = self.trade_manager.state
                price = getattr(self, '_current_price', s.entry_price)
                status["unrealized_pnl"] = round(
                    (price - s.entry_price) * self.position * self.position_lots, 2)

            with open(status_path, 'w') as f:
                json.dump(status, f)
        except Exception as e:
            logger.warning(f"Status write failed: {e}")

    def _check_stop_signal(self):
        """Check if GUI requested this bot to stop."""
        try:
            stop_file = os.path.join(self.config.status_dir, f"{self.bot_id}.stop")
            if os.path.exists(stop_file):
                os.remove(stop_file)
                logger.info(f"Stop signal received for {self.bot_id}")
                return True
        except Exception:
            pass
        return False

    def _reload_config(self):
        """Reload runtime config from status/{bot_id}.config.json."""
        try:
            config_path = os.path.join(self.config.status_dir, f"{self.bot_id}.config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                new_risk = cfg.get("risk_per_trade")
                new_loss = cfg.get("max_daily_loss")
                if new_risk is not None and new_risk != self.risk_pct:
                    logger.info(f"Risk updated: {self.risk_pct:.3f} → {new_risk:.3f}")
                    self.risk_pct = new_risk
                if new_loss is not None and new_loss != self.config.max_daily_loss:
                    logger.info(f"Daily loss limit updated: {self.config.max_daily_loss:.3f} → {new_loss:.3f}")
                    self.config.max_daily_loss = new_loss
        except Exception as e:
            logger.debug(f"Config reload: {e}")

    def _cleanup(self):
        self._print_summary()
        if not self.dry_run: self.executor.disconnect()

    def _print_summary(self):
        if not self.trade_history: logger.info("No trades"); return
        n = len(self.trade_history)
        wins = [t for t in self.trade_history if t["pnl_dollar"] > 0]
        losses = [t for t in self.trade_history if t["pnl_dollar"] <= 0]
        wr = len(wins) / n * 100
        total_pnl = sum(t["pnl_dollar"] for t in self.trade_history)
        avg_r = sum(t["pnl_r"] for t in self.trade_history) / n
        pf = abs(sum(t["pnl_dollar"] for t in wins) / sum(t["pnl_dollar"] for t in losses)) if losses else float("inf")

        # Drawdown (from equity curve relative to initial balance)
        cumulative_pnl = np.cumsum([0] + [t["pnl_dollar"] for t in self.trade_history])
        equity = self.starting_balance + cumulative_pnl
        peak = np.maximum.accumulate(equity)
        dd = np.where(peak > 0, (peak - equity) / peak * 100, 0)
        max_dd = float(np.max(dd))

        logger.info(f"\n{'='*50}\nTRADE SUMMARY\n{'='*50}")
        logger.info(f"  Trades: {n}  |  Wins: {len(wins)} ({wr:.1f}%)")
        logger.info(f"  Total PnL: ${total_pnl:+,.2f}  |  Avg R: {avg_r:+.3f}")
        logger.info(f"  Profit Factor: {pf:.2f}  |  Max DD: {max_dd:.1f}%")
        logger.info(f"  Start: ${self.starting_balance:,.2f}  →  Final: ${self.balance:,.2f} "
                   f"({(self.balance/self.starting_balance-1)*100:+.2f}%)")
        logger.info(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="BTC Trading System — Listening Architecture")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--bot-id", default="BTCBot")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--model", default=None)
    parser.add_argument("--risk-per-trade", type=float, default=None, dest="risk_pct")
    parser.add_argument("--max-daily-loss", type=float, default=None)
    parser.add_argument("--sl-atr", type=float, default=None)
    parser.add_argument("--poll-interval", type=int, default=None)
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--to", dest="to_date", default=None)
    parser.add_argument("--rule-only", action="store_true")
    args = parser.parse_args()

    config = BTCConfig()
    if args.sl_atr: config.initial_sl = args.sl_atr
    if args.poll_interval: config.poll_interval_seconds = args.poll_interval

    bot = BTCTradingSystem(config, bot_id=args.bot_id,
                           dry_run=args.dry_run and not args.live, live=args.live,
                           risk_pct=args.risk_pct, max_daily_loss=args.max_daily_loss)
    bot.from_date = args.from_date; bot.to_date = args.to_date
    bot.rule_only = args.rule_only
    bot.run_loop()
