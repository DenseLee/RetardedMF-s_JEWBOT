"""
BacktestDataManager — offline data preparation for backtesting.

Lifecycle:
  1. fetch M1 bars from MT5 (or fall back to CSV)
  2. resample to H1 and M15
  3. pre-compute all features, model outputs, and regime data
  4. cache everything to disk so simulation runs are fast
"""
import json, os, pickle, sys, time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from data.feature_engine_btc import BTCFeatureEngine, N_FEATURES
from models.cnn_lstm_encoder import CNNLSTMEncoder
from models.regime_classifier import RegimeClassifier, RuleBasedRegimeDetector, classify_regime, REGIME_NAMES
from models.cnn_gru_m15 import CNNGRUM15


@dataclass
class BacktestDataset:
    """All pre-computed data for a backtest run."""
    # OHLC
    m1_df: Optional[pd.DataFrame] = None
    h1_df: Optional[pd.DataFrame] = None
    m15_df: Optional[pd.DataFrame] = None

    # Pre-computed features (n_bars, 17) float32
    h1_features: Optional[np.ndarray] = None
    m15_features: Optional[np.ndarray] = None
    m1_features: Optional[np.ndarray] = None

    # H1 encoder outputs per bar
    h1_embeddings: Optional[np.ndarray] = None       # (n_h1, 128)
    h1_class_probs: Optional[np.ndarray] = None       # (n_h1, 4)
    h1_combined_regime: list = field(default_factory=list)  # list of dicts

    # M15 model outputs per bar
    m15_confidence: Optional[np.ndarray] = None       # (n_m15,)
    m15_direction_bias: Optional[np.ndarray] = None   # (n_m15,)

    # Rule detector outputs per H1 bar
    h1_atr_percentile: Optional[np.ndarray] = None    # (n_h1,)
    h1_rule_confidence: Optional[np.ndarray] = None   # (n_h1,)
    h1_rule_regime: Optional[np.ndarray] = None       # (n_h1,) str

    # H4 regime per H1 bar (if H4 encoder available)
    h1_h4_regime: Optional[np.ndarray] = None          # (n_h1,) str

    # Metadata
    start_date: str = ""
    end_date: str = ""
    n_h1: int = 0
    n_m15: int = 0
    n_m1: int = 0
    has_m1: bool = False


class BacktestDataManager:
    """Fetch, resample, pre-compute, cache — run once before simulation."""

    def __init__(self, config: BTCConfig = None, device: str = "auto"):
        self.config = config or BTCConfig()
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "mps" if device == "auto" and torch.backends.mps.is_available() else
            "cpu" if device == "auto" else device)
        self.engine = BTCFeatureEngine()
        self._cache_dir = os.path.join(self.config.project_root, "backtest", "cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def prepare(self, start: str, end: str,
                use_cache: bool = True, force_refresh: bool = False) -> BacktestDataset:
        """Main entry point: fetch + pre-compute, returns a ready-to-use dataset."""
        t0 = time.time()

        if force_refresh or not use_cache:
            dataset = self._build_dataset(start, end)
            self._save_cache(dataset, start, end)
        else:
            dataset = self._load_cache(start, end)
            if dataset is None:
                dataset = self._build_dataset(start, end)
                self._save_cache(dataset, start, end)

        elapsed = time.time() - t0
        print(f"Data ready: {dataset.n_h1} H1, {dataset.n_m15} M15, "
              f"{dataset.n_m1} M1 bars ({elapsed:.1f}s)")
        return dataset

    # ── build ─────────────────────────────────────────────────────────────

    def _build_dataset(self, start: str, end: str) -> BacktestDataset:
        ds = BacktestDataset(start_date=start, end_date=end)

        # Step 1 — fetch + resample
        ds.m1_df, ds.h1_df, ds.m15_df = self._fetch_and_resample(start, end)
        ds.has_m1 = ds.m1_df is not None and len(ds.m1_df) > 0
        ds.n_h1 = len(ds.h1_df)
        ds.n_m15 = len(ds.m15_df)
        ds.n_m1 = len(ds.m1_df) if ds.has_m1 else 0

        # Step 2 — pre-compute features
        print("  Computing features...")
        ds.h1_features = self.engine.compute(ds.h1_df)
        ds.m15_features = self.engine.compute(ds.m15_df)
        if ds.has_m1:
            ds.m1_features = self.engine.compute(ds.m1_df)

        # Step 3 — run H1 encoder + classifier on every bar
        print("  Running H1 encoder + classifier...")
        self._precompute_h1_model(ds)

        # Step 4 — pre-warm rule detector + store per-bar state
        print("  Running rule-based regime detector...")
        self._precompute_rule_detector(ds)

        # Step 5 — combine model + rule into combined regime per bar
        print("  Combining regime sources...")
        self._combine_regimes(ds)

        # Step 6 — run M15 model on every bar
        print("  Running M15 model...")
        self._precompute_m15_model(ds)

        # Step 7 — H4 encoder if available
        print("  Running H4 encoder...")
        self._precompute_h4(ds)

        return ds

    # ── step 1: fetch + resample ──────────────────────────────────────────

    def _fetch_and_resample(self, start: str, end: str):
        """Pull M1 from MT5, resample to H1 and M15. Fall back to CSV."""
        m1_df, h1_df, m15_df = None, None, None

        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                raise RuntimeError("mt5.initialize() failed")
            print(f"  Fetching M1 bars from MT5 ({start} → {end})...")
            symbol = self.config.symbol

            # MT5 copy_rates_range interprets naive datetimes in server timezone.
            # We want UTC dates, so add 1-day padding to handle any timezone offset.
            from datetime import timedelta
            t_start = datetime.fromisoformat(start) - timedelta(days=1)
            t_end = datetime.fromisoformat(end) + timedelta(days=1)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, t_start, t_end)
            mt5.shutdown()

            if rates is None or len(rates) == 0:
                raise RuntimeError("No M1 bars returned")

            m1_df = pd.DataFrame(rates)
            m1_df = m1_df.rename(columns={
                'time': 'timestamp', 'tick_volume': 'volume'})
            m1_df['timestamp'] = pd.to_datetime(m1_df['timestamp'], unit='s', utc=True)
            m1_df = m1_df.sort_values('timestamp').reset_index(drop=True)
            print(f"    → {len(m1_df)} M1 bars")

            # Trim to requested UTC date range
            since = pd.Timestamp(start, tz='UTC')
            until = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)  # include end date
            m1_df = m1_df[(m1_df['timestamp'] >= since) & (m1_df['timestamp'] <= until)]
            m1_df = m1_df.reset_index(drop=True)
            print(f"    → {len(m1_df)} M1 bars in range [{since} → {until}]")

            print("  Resampling M1 → H1 and M15...")
            h1_df, m15_df = self._resample_from_m1(m1_df)
            return m1_df, h1_df, m15_df

        except Exception as e:
            print(f"  MT5 unavailable ({e}), falling back to CSV...")
            h1_df, m15_df = self._load_csv(start, end)
            return None, h1_df, m15_df

    @staticmethod
    def _resample_from_m1(m1_df: pd.DataFrame):
        """Resample M1 bars → H1 and M15 OHLC."""
        def _resample(df, rule):
            resampled = df.set_index('timestamp').resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna().reset_index()
            if 'spread' in df.columns:
                spread_col = df.set_index('timestamp')['spread'].resample(rule).mean()
            return resampled

        h1 = _resample(m1_df, '1h')
        m15 = _resample(m1_df, '15min')
        return h1, m15

    def _load_csv(self, start: str, end: str):
        """Fallback: load H1 and M15 from CSV files."""
        data_dir = self.config.data_dir
        h1_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if '1h' in f.lower() and f.endswith('.csv')]
        m15_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if '15m' in f.lower() and f.endswith('.csv')]

        if not h1_paths:
            raise FileNotFoundError(f"No H1 CSV found in {data_dir}")

        h1_df = pd.read_csv(h1_paths[0])
        h1_df['timestamp'] = pd.to_datetime(h1_df['timestamp'], utc=True)
        h1_df = h1_df.rename(columns={'tick_volume': 'volume'} if 'tick_volume' in h1_df.columns else {})
        if 'volume' not in h1_df.columns:
            h1_df['volume'] = 0
        since = pd.Timestamp(start, tz='UTC')
        until = pd.Timestamp(end, tz='UTC')
        h1_df = h1_df[(h1_df['timestamp'] >= since) & (h1_df['timestamp'] <= until)]
        h1_df = h1_df.sort_values('timestamp').reset_index(drop=True)

        m15_df = None
        if m15_paths:
            m15_df = pd.read_csv(m15_paths[0])
            m15_df['timestamp'] = pd.to_datetime(m15_df['timestamp'], utc=True)
            m15_df = m15_df.rename(columns={'tick_volume': 'volume'} if 'tick_volume' in m15_df.columns else {})
            if 'volume' not in m15_df.columns:
                m15_df['volume'] = 0
            m15_df = m15_df[(m15_df['timestamp'] >= since) & (m15_df['timestamp'] <= until)]
            m15_df = m15_df.sort_values('timestamp').reset_index(drop=True)
        else:
            # Resample H1→M15 as fallback (loses intrabar detail)
            m15_df = h1_df.set_index('timestamp').resample('15min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna().reset_index()

        print(f"    CSV loaded: {len(h1_df)} H1, {len(m15_df)} M15 bars")
        return h1_df, m15_df

    # ── step 3: H1 encoder + classifier ───────────────────────────────────

    def _precompute_h1_model(self, ds: BacktestDataset):
        encoder, classifier = self._load_h1_models()
        seq_len = self.config.seq_len_h1
        n = ds.n_h1

        ds.h1_embeddings = np.zeros((n, self.config.embedding_dim), dtype=np.float32)
        ds.h1_class_probs = np.zeros((n, self.config.regime_classes), dtype=np.float32)

        for i in range(seq_len - 1, n):
            seq = self.engine.compute_sequence(ds.h1_features, i, seq_len)
            t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                enc_out = encoder(t)
                raw_logits = classifier.raw_logits(enc_out["embedding"])
                probs = torch.softmax(raw_logits / 4.0, dim=1)
            ds.h1_embeddings[i] = enc_out["embedding"].squeeze(0).cpu().numpy()
            ds.h1_class_probs[i] = probs.squeeze(0).cpu().numpy()

        del encoder, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_h1_models(self):
        mpath = os.path.join(self.config.model_dir, "btc_h1_encoder.pt")
        encoder = CNNLSTMEncoder(
            n_features=N_FEATURES, seq_len=self.config.seq_len_h1,
            cnn_channels=self.config.cnn_channels, lstm_hidden=self.config.lstm_hidden,
            lstm_layers=self.config.lstm_layers, dropout=self.config.lstm_dropout,
            embedding_dim=self.config.embedding_dim, regime_classes=self.config.regime_classes,
            bidirectional=self.config.lstm_bidirectional).to(self.device).eval()
        classifier = RegimeClassifier(
            embedding_dim=self.config.embedding_dim,
            n_classes=self.config.regime_classes).to(self.device).eval()
        ckpt = torch.load(mpath, map_location=self.device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        return encoder, classifier

    # ── step 4: rule detector ─────────────────────────────────────────────

    def _precompute_rule_detector(self, ds: BacktestDataset):
        rd = RuleBasedRegimeDetector()
        n = ds.n_h1
        ds.h1_atr_percentile = np.zeros(n, dtype=np.float32)
        ds.h1_rule_confidence = np.zeros(n, dtype=np.float32)
        ds.h1_rule_regime = np.full(n, 'RANGE', dtype=object)

        for i in range(n):
            row = ds.h1_df.iloc[i]
            result = rd.update(row['high'], row['low'], row['close'])
            ds.h1_atr_percentile[i] = result.get('atr_percentile', 0.5)
            ds.h1_rule_confidence[i] = result.get('confidence', 0.0)
            ds.h1_rule_regime[i] = result.get('regime', 'RANGE')

    # ── step 5: combined regime ───────────────────────────────────────────

    def _combine_regimes(self, ds: BacktestDataset):
        threshold = self.config.min_regime_confidence
        ds.h1_combined_regime = []

        for i in range(ds.n_h1):
            if i < self.config.seq_len_h1 - 1:
                ds.h1_combined_regime.append({
                    'regime': 'RANGE', 'direction': 0, 'confidence': 0.0,
                    'source': 'cold_start', 'atr_percentile': 0.5})
                continue

            max_prob = float(ds.h1_class_probs[i].max())
            pred_class = int(np.argmax(ds.h1_class_probs[i]))

            if max_prob >= threshold:
                regime = REGIME_NAMES[pred_class]
                ds.h1_combined_regime.append({
                    'regime': regime,
                    'direction': {'TREND_UP': 1, 'TREND_DOWN': -1}.get(regime, 0),
                    'confidence': max_prob, 'source': 'model',
                    'atr_percentile': float(ds.h1_atr_percentile[i])})
            else:
                # Use rule detector's actual classification (matching live classify_regime)
                rule_regime = ds.h1_rule_regime[i] if ds.h1_rule_regime is not None else 'RANGE'
                ds.h1_combined_regime.append({
                    'regime': rule_regime,
                    'direction': {'TREND_UP': 1, 'TREND_DOWN': -1}.get(rule_regime, 0),
                    'confidence': float(ds.h1_rule_confidence[i]),
                    'source': 'rule',
                    'atr_percentile': float(ds.h1_atr_percentile[i])})

    # ── step 6: M15 model ─────────────────────────────────────────────────

    def _precompute_m15_model(self, ds: BacktestDataset):
        try:
            model = self._load_m15_model()
        except FileNotFoundError:
            ds.m15_confidence = np.zeros(ds.n_m15, dtype=np.float32)
            ds.m15_direction_bias = np.zeros(ds.n_m15, dtype=np.float32)
            return

        seq_len = self.config.seq_len_m15
        n = ds.n_m15
        ds.m15_confidence = np.zeros(n, dtype=np.float32)
        ds.m15_direction_bias = np.zeros(n, dtype=np.float32)

        for i in range(seq_len - 1, n):
            seq = self.engine.compute_sequence(ds.m15_features, i, seq_len)
            t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(t)
            ds.m15_confidence[i] = float(out['entry_confidence'].squeeze().cpu().numpy())
            ds.m15_direction_bias[i] = float(out.get('direction_bias',
                np.zeros(1)).squeeze() if hasattr(out.get('direction_bias', None), 'squeeze') else 0.0)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_m15_model(self):
        mpath = os.path.join(self.config.model_dir, "btc_m15_v2.pt")
        model = CNNGRUM15(
            n_features=N_FEATURES, seq_len=self.config.seq_len_m15,
            cnn_channels=self.config.gru_cnn_channels, gru_hidden=self.config.gru_hidden,
            gru_layers=self.config.gru_layers, dropout=self.config.gru_dropout).to(self.device).eval()
        ckpt = torch.load(mpath, map_location=self.device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        # Remap v2 checkpoint keys (M15EntryClassifier → CNNGRUM15)
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
        model.load_state_dict(state_dict, strict=False)
        return model

    # ── step 7: H4 encoder ────────────────────────────────────────────────

    def _precompute_h4(self, ds: BacktestDataset):
        ds.h1_h4_regime = np.full(ds.n_h1, None, dtype=object)
        h4_path = os.path.join(self.config.model_dir, "btc_h4_encoder.pt")
        if not os.path.exists(h4_path):
            print("    (no H4 encoder found, skipping)")
            return

        try:
            from models.h4_encoder import H4Encoder
            h4_encoder = H4Encoder(
                n_features=N_FEATURES, embedding_dim=getattr(self.config, 'h4_embedding_dim', 64),
                n_classes=self.config.regime_classes).to(self.device).eval()
            ckpt = torch.load(h4_path, map_location=self.device, weights_only=False)
            h4_encoder.load_state_dict(ckpt.get('encoder_state_dict', ckpt), strict=False)
        except Exception as e:
            print(f"    H4 encoder load failed: {e}")
            return

        # Resample H1 → H4
        h4_df = ds.h1_df.set_index('timestamp').resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()
        if len(h4_df) < 24:
            del h4_encoder; return

        h4_feats = self.engine.compute(h4_df)
        h4_regimes = np.full(len(h4_df), None, dtype=object)
        h4_seq_len = getattr(self.config, 'h4_seq_len', 24)
        for i in range(h4_seq_len - 1, len(h4_df)):
            seq = self.engine.compute_sequence(h4_feats, i, h4_seq_len)
            t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = h4_encoder(t)
            cls_out = out.get('regime_logits', out.get('logits'))
            if cls_out is not None:
                probs = torch.softmax(cls_out, dim=1).squeeze(0).cpu().numpy()
                h4_regimes[i] = REGIME_NAMES[int(np.argmax(probs))]

        # Map H4 regime to each H1 bar
        for h1_i in range(ds.n_h1):
            h1_ts = ds.h1_df['timestamp'].iloc[h1_i]
            h4_idx = (h4_df['timestamp'] <= h1_ts).sum() - 1
            if 0 <= h4_idx < len(h4_regimes) and h4_regimes[h4_idx] is not None:
                ds.h1_h4_regime[h1_i] = h4_regimes[h4_idx]

        del h4_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── cache I/O ─────────────────────────────────────────────────────────

    def _cache_paths(self, start: str, end: str) -> Dict[str, str]:
        tag = f"{start}_{end}".replace(":", "").replace("-", "")
        base = self._cache_dir
        return {
            'meta': os.path.join(base, f"meta_{tag}.json"),
            'h1_df': os.path.join(base, f"h1_{tag}.parquet"),
            'm15_df': os.path.join(base, f"m15_{tag}.parquet"),
            'm1_df': os.path.join(base, f"m1_{tag}.parquet"),
            'arrays': os.path.join(base, f"arrays_{tag}.npz"),
        }

    def _save_cache(self, ds: BacktestDataset, start: str, end: str):
        paths = self._cache_paths(start, end)
        ds.h1_df.to_parquet(paths['h1_df'])
        ds.m15_df.to_parquet(paths['m15_df'])
        if ds.has_m1:
            ds.m1_df.to_parquet(paths['m1_df'])

        # Save arrays in a compressed .npz
        arrays = {
            'h1_features': ds.h1_features,
            'm15_features': ds.m15_features,
            'h1_embeddings': ds.h1_embeddings,
            'h1_class_probs': ds.h1_class_probs,
            'h1_atr_percentile': ds.h1_atr_percentile,
            'h1_rule_confidence': ds.h1_rule_confidence,
            'm15_confidence': ds.m15_confidence,
            'm15_direction_bias': ds.m15_direction_bias,
        }
        if ds.m1_features is not None:
            arrays['m1_features'] = ds.m1_features
        np.savez_compressed(paths['arrays'], **arrays)

        # Save combined regime, rule regime, and H4 as pickle
        aux = {
            'h1_combined_regime': ds.h1_combined_regime,
            'h1_rule_regime': ds.h1_rule_regime.tolist() if ds.h1_rule_regime is not None else None,
            'h1_h4_regime': ds.h1_h4_regime.tolist() if ds.h1_h4_regime is not None else None,
        }
        aux_path = os.path.join(self._cache_dir, f"aux_{start}_{end}".replace(":", "").replace("-", "") + ".pkl")
        with open(aux_path, 'wb') as f:
            pickle.dump(aux, f)

        # Config snapshot for cache validation
        meta = {
            'start': start, 'end': end,
            'n_h1': ds.n_h1, 'n_m15': ds.n_m15, 'n_m1': ds.n_m1, 'has_m1': ds.has_m1,
            'seq_len_h1': self.config.seq_len_h1,
            'seq_len_m15': self.config.seq_len_m15,
            'n_features': N_FEATURES,
            'embedding_dim': self.config.embedding_dim,
            'regime_classes': self.config.regime_classes,
        }
        with open(paths['meta'], 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"  Cached to {self._cache_dir}")

    def _load_cache(self, start: str, end: str) -> Optional[BacktestDataset]:
        paths = self._cache_paths(start, end)
        if not os.path.exists(paths['meta']):
            return None

        with open(paths['meta']) as f:
            meta = json.load(f)

        # Validate cache
        if meta.get('seq_len_h1') != self.config.seq_len_h1:
            print("  Cache invalid (seq_len_h1 changed), rebuilding...")
            return None
        if meta.get('seq_len_m15') != self.config.seq_len_m15:
            print("  Cache invalid (seq_len_m15 changed), rebuilding...")
            return None

        print("  Loading cached data...")
        ds = BacktestDataset(
            start_date=start, end_date=end,
            n_h1=meta['n_h1'], n_m15=meta['n_m15'], n_m1=meta['n_m1'],
            has_m1=meta.get('has_m1', False),
            h1_df=pd.read_parquet(paths['h1_df']),
            m15_df=pd.read_parquet(paths['m15_df']),
        )

        if ds.has_m1 and os.path.exists(paths['m1_df']):
            ds.m1_df = pd.read_parquet(paths['m1_df'])

        data = np.load(paths['arrays'])
        ds.h1_features = data['h1_features']
        ds.m15_features = data['m15_features']
        ds.h1_embeddings = data['h1_embeddings']
        ds.h1_class_probs = data['h1_class_probs']
        ds.h1_atr_percentile = data['h1_atr_percentile']
        ds.h1_rule_confidence = data['h1_rule_confidence']
        ds.m15_confidence = data['m15_confidence']
        ds.m15_direction_bias = data['m15_direction_bias']
        if 'm1_features' in data:
            ds.m1_features = data['m1_features']

        # Load auxiliary data
        aux_tag = f"aux_{start}_{end}".replace(":", "").replace("-", "")
        aux_path = os.path.join(self._cache_dir, f"{aux_tag}.pkl")
        if os.path.exists(aux_path):
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            ds.h1_combined_regime = aux['h1_combined_regime']
            ds.h1_rule_regime = np.array(aux['h1_rule_regime'], dtype=object) if aux.get('h1_rule_regime') else None
            ds.h1_h4_regime = np.array(aux['h1_h4_regime'], dtype=object) if aux.get('h1_h4_regime') else None

        return ds
