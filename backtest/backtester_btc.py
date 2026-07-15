"""
BTCBacktester — drip-feed simulation using pre-computed data.

Walks M15 bars chronologically, detecting H1 closes, managing a listening
window, confirming on M15, entering trades, and using M1 intrabar data
for precise SL/TP fill detection.
"""
import os, sys, time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from models.entry_gate import EntryGate
from models.trade_manager_btc import TradeManager, TradeActionType
from .data_manager import BacktestDataset
from .slippage_model import SlippageModel, SlippageConfig

BLOCKED_HOURS = set()  # removed — phased TradeManager handles bad entries


@dataclass
class BacktestTrade:
    entry_time: datetime = None
    exit_time: datetime = None
    direction: int = 0           # 1=long, -1=short
    entry_price: float = 0.0
    exit_price: float = 0.0
    lots: float = 0.0
    pnl_dollar: float = 0.0
    pnl_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    regime_at_entry: str = ""
    confidence_at_entry: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    m15_entry_idx: int = 0


@dataclass
class BacktestResult:
    config_snapshot: dict = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    avg_bars_held: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    avg_capture_ratio: float = 0.0
    total_slippage_cost: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    monthly: Optional[pd.DataFrame] = None
    weekly: Optional[pd.DataFrame] = None
    by_hour: Optional[pd.DataFrame] = None
    by_regime: Optional[pd.DataFrame] = None
    by_exit_reason: Optional[pd.DataFrame] = None
    by_direction: dict = field(default_factory=dict)


class BTCBacktester:
    """Drip-feed backtest using pre-computed data with M1 intrabar resolution."""

    def __init__(self, config: BTCConfig = None,
                 initial_balance: float = 10000.0,
                 slippage: SlippageConfig = None):
        self.config = config or BTCConfig()
        self.initial_balance = initial_balance
        self.slippage = SlippageModel(slippage)

    # ── public API ────────────────────────────────────────────────────────

    def run(self, dataset: BacktestDataset, verbose: bool = True,
            warm_start_bars: int = None,
            no_rule_fallback: bool = False,
            m15_conf_threshold: float = 0.5,
            price_action_filter: bool = False) -> BacktestResult:
        """Run the drip-feed simulation. Returns full BacktestResult.

        Args:
            no_rule_fallback: If True, skip EMA21 rule-based M15 confirmation.
            m15_conf_threshold: Minimum M15 model confidence (default 0.5).
            price_action_filter: If True, block SHORT if price made a higher
                M15 low in last 8 bars (and vice versa for LONG).
        """
        t0 = time.time()
        ds = dataset

        # ── Unpack pre-computed arrays ──
        h1_df = ds.h1_df
        m15_df = ds.m15_df
        m1_hi = ds.m1_df['high'].values if ds.has_m1 else None
        m1_lo = ds.m1_df['low'].values if ds.has_m1 else None

        h1_feats = ds.h1_features
        h1_regime_list = ds.h1_combined_regime
        h1_atr_pctl = ds.h1_atr_percentile
        h1_rule_regime = ds.h1_rule_regime
        m15_conf = ds.m15_confidence
        h1_h4_regime = ds.h1_h4_regime

        # ── Initialize state ──
        tm = TradeManager(
            phase1_sl=self.config.initial_sl, hard_tp=self.config.hard_tp,
            phase1_bars=self.config.phase1_bars,
            phase2_mfe_min=self.config.phase2_mfe_min,
            phase3_trail=self.config.phase3_trail,
            phase4_trail=self.config.phase4_trail,
            phase4_start=self.config.phase4_start,
            max_hold=self.config.max_hold_bars)
        gate = EntryGate(
            min_confidence=self.config.min_regime_confidence,
            min_atr_pct=self.config.min_atr_percentile,
            max_atr_pct=self.config.max_atr_percentile)

        listening = False
        h1_signal = 0
        h1_confidence = 0.0
        h1_atr = 0.0
        bars_listened = 0
        max_listen = self.config.max_listen_bars

        position = 0
        position_lots = 0.0
        balance = self.initial_balance
        daily_pnl = 0.0
        last_day = None
        starting_balance_daily = self.initial_balance
        trades: List[BacktestTrade] = []
        equity_curve = [self.initial_balance]
        last_entry_info = {}
        last_regime = None
        cooldown_bars = 0  # prevent rapid re-entry after exit

        # ── Warm start: need seq_len_h1 H1 bars of context ──
        min_h1 = self.config.seq_len_h1  # 96
        if ds.n_h1 < min_h1:
            raise ValueError(
                f"Need at least {min_h1} H1 bars for warmup, have {ds.n_h1}. "
                f"Extend the start date earlier.")
        # Find the M15 bar index after the warmup H1 bars
        warm_h1_ts = ds.h1_df['timestamp'].iloc[min_h1 - 1]
        warm_h1_close = warm_h1_ts + pd.Timedelta(hours=1)
        # Skip M15 bars before the 96th H1 bar closes
        warm_m15 = warm_start_bars or int((ds.m15_df['timestamp'] < warm_h1_close).sum())
        warm_m15 = max(warm_m15, self.config.seq_len_m15)
        if warm_m15 >= len(m15_df) - 1:
            raise ValueError(
                f"Not enough M15 bars after warmup: warm_start={warm_m15}, "
                f"total={len(m15_df)}. Use a larger date range.")

        last_h1_idx = -1
        n_m15 = len(m15_df)

        # ── Main simulation loop ──
        for m15_i in range(warm_m15, n_m15):
            ts = m15_df['timestamp'].iloc[m15_i]
            price = m15_df['close'].iloc[m15_i]

            # Map M15 timestamp to H1 index
            h1_i = self._h1_index(ts, h1_df)

            # ── New H1 bar close ──
            if h1_i != last_h1_idx and h1_i >= 0:
                last_h1_idx = h1_i
                ri = h1_regime_list[h1_i]
                current_regime = ri['regime']
                atr_pct = float(h1_atr_pctl[h1_i])
                current_atr = float(h1_feats[h1_i, 6]) * float(h1_df['close'].iloc[h1_i])
                bb_pos = float(h1_feats[h1_i, 4])

                # Rule-wins-conflict: if model and rule disagree on TREND direction, trust rule
                if h1_rule_regime is not None and h1_rule_regime[h1_i] is not None:
                    trending = {'TREND_UP', 'TREND_DOWN'}
                    rule_r = h1_rule_regime[h1_i]
                    if (current_regime in trending and rule_r in trending
                            and current_regime != rule_r):
                        current_regime = rule_r
                        ri['regime'] = rule_r
                        ri['source'] = 'rule'

                gd = gate.evaluate(
                    current_regime, ri['confidence'], atr_pct, bb_position=bb_pos)

                if gd.entry_signal:
                    # H4 trend filter only (EMA22 removed — phased TradeManager handles bad entries)
                    if h1_h4_regime is not None and h1_h4_regime[h1_i] is not None:
                        h4r = h1_h4_regime[h1_i]
                        against = ((gd.direction == 1 and h4r == 'TREND_DOWN') or
                                   (gd.direction == -1 and h4r == 'TREND_UP'))
                        if against:
                            listening = False
                            h1_signal = 0
                            continue

                    h1_signal = gd.direction
                    h1_confidence = gd.confidence
                    h1_atr = current_atr
                    listening = True
                    bars_listened = 0
                    last_regime = current_regime
                else:
                    h1_signal = 0
                    listening = False

            # ── Manage open position ──
            if position != 0 and tm.state is not None:
                exit_info = None
                if ds.has_m1:
                    exit_info = self._check_m1_sl_tp(
                        m15_i, ts, tm, ds, position)

                if exit_info is not None:
                    last_entry_info['exit_ts'] = ts
                    pnl_r, pnl_dollar = self._close_trade(
                        exit_info['price'], exit_info['reason'], tm, position,
                        position_lots, balance, daily_pnl, trades, last_entry_info,
                        self.slippage)
                    balance += pnl_dollar
                    daily_pnl += pnl_dollar
                    position = 0
                    position_lots = 0.0
                    tm.state = None
                    cooldown_bars = 2  # 2-bar cooldown after exit
                    equity_curve.append(balance)
                    continue

                # Trade manager update (breakeven, trailing, time stop)
                m15_hi = float(m15_df['high'].iloc[m15_i])
                m15_lo = float(m15_df['low'].iloc[m15_i])
                action = tm.update(price, m15_hi, m15_lo, h1_atr)

                if action.action_type == TradeActionType.CLOSE:
                    exit_px = self.slippage.exit_price(price, position)
                    last_entry_info['exit_ts'] = ts
                    pnl_r, pnl_dollar = self._close_trade(
                        exit_px, action.reason, tm, position, position_lots,
                        balance, daily_pnl, trades, last_entry_info, self.slippage)
                    balance += pnl_dollar
                    daily_pnl += pnl_dollar
                    position = 0
                    position_lots = 0.0
                    tm.state = None
                    cooldown_bars = 2  # 2-bar cooldown after exit
                    equity_curve.append(balance)
                    continue

            # ── Listening window ──
            if not listening:
                continue

            bars_listened += 1
            if bars_listened > max_listen:
                listening = False
                h1_signal = 0
                continue

            if ts.hour in BLOCKED_HOURS:
                continue

            # Cooldown after exit
            if cooldown_bars > 0:
                cooldown_bars -= 1
                continue

            # M15 confirmation — only if no position open and cooldown expired
            if position == 0 and self._m15_confirm(m15_i, ds, h1_signal, m15_df,
                                 no_rule_fallback, m15_conf_threshold,
                                 price_action_filter):
                lots = TradeManager.compute_position_size(
                    balance, h1_atr, price, self.config.risk_pct,
                    self.config.initial_sl)
                entry_fill = self.slippage.entry_price(price, h1_signal)
                tm.enter(h1_signal, entry_fill, h1_atr, lots,
                         regime=last_regime or "")
                position = h1_signal
                position_lots = lots
                listening = False
                last_entry_info = {
                    'h1_i': h1_i, 'm15_i': m15_i,
                    'signal_price': price, 'fill_price': entry_fill,
                    'regime': last_regime, 'confidence': h1_confidence,
                    'entry_ts': ts,
                }

            # ── Daily PnL reset ──
            today = ts.date()
            if last_day and today != last_day:
                daily_pnl = 0.0
                starting_balance_daily = balance
            last_day = today

            # ── Track equity ──
            if position == 0:
                equity_curve.append(balance)

        # Force-close any remaining position at last price
        if position != 0 and tm.state is not None:
            final_px = float(m15_df['close'].iloc[-1])
            exit_px = self.slippage.exit_price(final_px, position)
            last_entry_info['exit_ts'] = m15_df['timestamp'].iloc[-1]
            pnl_r, pnl_dollar = self._close_trade(
                exit_px, 'end_of_data', tm, position, position_lots,
                balance, daily_pnl, trades, last_entry_info, self.slippage)
            balance += pnl_dollar
            position = 0
            position_lots = 0.0
            tm.state = None
            cooldown_bars = 0
            equity_curve.append(balance)

        result = self._build_result(
            trades, equity_curve, balance, ds, time.time() - t0)

        if verbose:
            self._print_summary(result)

        return result

    # ── H1 index mapping ──────────────────────────────────────────────────

    @staticmethod
    def _h1_index(m15_ts, h1_df):
        """Find the latest H1 bar whose timestamp <= m15_ts."""
        idx = (h1_df['timestamp'] <= m15_ts).sum() - 1
        return idx if idx >= 0 else -1

    # ── M1 intrabar SL/TP check ───────────────────────────────────────────

    def _check_m1_sl_tp(self, m15_i, m15_ts, tm, ds, position):
        """Iterate M1 bars within this M15 bar for SL/TP hits."""
        if ds.m1_df is None:
            return None

        m1_ts = ds.m1_df['timestamp']
        m1_hi = ds.m1_df['high'].values
        m1_lo = ds.m1_df['low'].values

        m15_start = m15_ts - pd.Timedelta(minutes=15)
        mask = (m1_ts > m15_start) & (m1_ts <= m15_ts)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None

        state = tm.state
        if state is None:
            return None

        for j in indices:
            hi = float(m1_hi[j])
            lo = float(m1_lo[j])

            if position == 1:  # Long
                if lo <= state.current_sl:
                    fill = self.slippage.sl_fill_price(state.current_sl, 1)
                    return {'price': fill, 'reason': 'sl_hit', 'm1_idx': int(j)}
                if hi >= state.current_tp:
                    fill = self.slippage.tp_fill_price(state.current_tp, 1)
                    return {'price': fill, 'reason': 'tp_hit', 'm1_idx': int(j)}
            else:  # Short
                if hi >= state.current_sl:
                    fill = self.slippage.sl_fill_price(state.current_sl, -1)
                    return {'price': fill, 'reason': 'sl_hit', 'm1_idx': int(j)}
                if lo <= state.current_tp:
                    fill = self.slippage.tp_fill_price(state.current_tp, -1)
                    return {'price': fill, 'reason': 'tp_hit', 'm1_idx': int(j)}

        return None

    # ── M15 confirmation ──────────────────────────────────────────────────

    def _m15_confirm(self, m15_i, ds, h1_sig, m15_df,
                     no_rule_fallback=False, m15_threshold=0.5,
                     price_action_filter=False):
        """M15 confirmation — 2-bar momentum in signal direction.

        Simple rule: candle must be turning in the H1 signal direction.
        No EMA proximity requirement — pullback entry in trends, reversal
        entry in ranges. The phased TradeManager's wide SL + wrong-detection
        handles the bad entries.
        """
        m15_closes = m15_df['close'].values[:m15_i + 1]
        if len(m15_closes) < 3:
            return False

        if h1_sig == 1:
            turning = m15_closes[-1] > m15_closes[-2]
        elif h1_sig == -1:
            turning = m15_closes[-1] < m15_closes[-2]
        else:
            return False

        return turning

    @staticmethod
    def _passes_price_action(m15_i, h1_sig, m15_df, lookback=8):
        """Block counter-trend entries by checking M15 swing structure."""
        if m15_i < lookback:
            return True  # not enough bars, allow

        lows = m15_df['low'].values[:m15_i + 1]
        highs = m15_df['high'].values[:m15_i + 1]

        if h1_sig == -1:  # SHORT: price should be making LOWER lows
            recent_low = float(np.min(lows[-lookback:]))
            prior_low = float(np.min(lows[-lookback*2:-lookback])) if len(lows) >= lookback*2 else recent_low
            # Block if recent low is HIGHER than prior low (price structure turning up)
            return recent_low <= prior_low
        elif h1_sig == 1:  # LONG: price should be making HIGHER highs
            recent_high = float(np.max(highs[-lookback:]))
            prior_high = float(np.max(highs[-lookback*2:-lookback])) if len(highs) >= lookback*2 else recent_high
            return recent_high >= prior_high

        return True

    # ── Trade execution ───────────────────────────────────────────────────

    def _close_trade(self, exit_price, reason, tm, position, lots, balance, daily_pnl,
                     trades, entry_info, slippage):
        """Compute PnL and append trade record. Returns (pnl_r, pnl_dollar)."""
        state = tm.state
        if state is None:
            return 0.0, 0.0

        direction = 1 if position == 1 else -1
        if direction == 1:
            pnl_dollar = (exit_price - state.entry_price) * lots
        else:
            pnl_dollar = (state.entry_price - exit_price) * lots

        sd = state.entry_atr * self.config.initial_sl
        pnl_r = pnl_dollar / max(sd * lots, 1e-9)

        entry_slip = state.entry_price - entry_info.get('signal_price', state.entry_price)
        exit_slip = 0.0  # approximated

        trades.append(BacktestTrade(
            entry_time=entry_info.get('entry_ts'),
            exit_time=entry_info.get('exit_ts', datetime.now()),
            direction=position,
            entry_price=state.entry_price,
            exit_price=exit_price,
            lots=lots,
            pnl_dollar=round(pnl_dollar, 2),
            pnl_r=round(float(pnl_r), 4),
            mfe_r=round(float(state.mfe_r), 4),
            mae_r=round(float(state.mae_r), 4),
            bars_held=state.bars_held,
            exit_reason=reason,
            regime_at_entry=entry_info.get('regime', ''),
            confidence_at_entry=entry_info.get('confidence', 0.0),
            entry_slippage=round(entry_slip, 2),
            exit_slippage=round(exit_slip, 2),
            m15_entry_idx=entry_info.get('m15_i', 0),
        ))

        return pnl_r, pnl_dollar

    # ── Metrics ───────────────────────────────────────────────────────────

    def _build_result(self, trades, equity, balance, ds, elapsed) -> BacktestResult:
        n = len(trades)
        if n == 0:
            return BacktestResult(
                start_date=ds.start_date, end_date=ds.end_date,
                initial_balance=self.initial_balance, final_balance=self.initial_balance,
                equity_curve=equity)

        pnls = np.array([t.pnl_dollar for t in trades])
        rs = np.array([t.pnl_r for t in trades])
        eq = np.array(equity)

        wins = np.sum(rs > 0)
        losses = np.sum(rs <= 0)
        wr = wins / n * 100

        total_pnl = float(np.sum(pnls))
        final_balance = self.initial_balance + total_pnl

        tg = float(np.sum(rs[rs > 0])) if wins > 0 else 0.0
        tl = float(np.abs(np.sum(rs[rs <= 0]))) if losses > 0 else 0.001
        pf = tg / max(tl, 0.001)

        avg_r = float(np.mean(rs))
        avg_win_r = float(np.mean(rs[rs > 0])) if wins > 0 else 0.0
        avg_loss_r = float(np.mean(rs[rs <= 0])) if losses > 0 else 0.0

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1)
        max_dd = float(np.max(dd)) * 100
        dd_end = int(np.argmax(dd))
        dd_start = int(np.argmax(eq[:dd_end + 1])) if dd_end > 0 else 0
        dd_duration = dd_end - dd_start

        # Sharpe / Sortino (per-trade, using R-units)
        std_rs = float(np.std(rs))
        sharpe = float(np.mean(rs) / max(std_rs, 0.001)) if n > 1 and std_rs > 1e-9 else 0.0
        down = rs[rs < 0]
        if len(down) > 1:
            std_down = float(np.std(down))
            sortino = float(np.mean(rs) / max(std_down, 0.001))
        else:
            sortino = sharpe

        # Calmar
        calmar = (final_balance / self.initial_balance - 1) / max(max_dd / 100, 0.001) if max_dd > 0.01 else 0.0

        # Expectancy
        expectancy = avg_r

        avg_bars = float(np.mean([t.bars_held for t in trades]))
        avg_mfe = float(np.mean([t.mfe_r for t in trades]))
        avg_mae = float(np.mean([t.mae_r for t in trades]))
        avg_capture = float(np.mean([t.pnl_r / max(t.mfe_r, 0.001) for t in trades])) if n > 0 else 0.0

        total_slippage = float(np.sum([abs(t.entry_slippage) + abs(t.exit_slippage) for t in trades]))

        # Breakdowns
        monthly = self._monthly_breakdown(trades, ds)
        by_hour = self._hourly_breakdown(trades)
        by_regime = self._regime_breakdown(trades)
        by_exit = self._exit_reason_breakdown(trades)

        long_trades = [t for t in trades if t.direction == 1]
        short_trades = [t for t in trades if t.direction == -1]
        by_dir = {
            'long': {'count': len(long_trades),
                     'win_rate': sum(1 for t in long_trades if t.pnl_r > 0) / max(len(long_trades), 1) * 100,
                     'pnl': sum(t.pnl_dollar for t in long_trades)},
            'short': {'count': len(short_trades),
                      'win_rate': sum(1 for t in short_trades if t.pnl_r > 0) / max(len(short_trades), 1) * 100,
                      'pnl': sum(t.pnl_dollar for t in short_trades)},
        }

        return BacktestResult(
            config_snapshot={
                'sl': self.config.initial_sl, 'tp': self.config.hard_tp,
                'risk': self.config.risk_pct, 'trail': self.config.trail_trigger,
                'breakeven': self.config.breakeven_trigger,
                'max_hold': self.config.max_hold_bars,
            },
            start_date=ds.start_date, end_date=ds.end_date,
            initial_balance=self.initial_balance, final_balance=final_balance,
            total_pnl=total_pnl, total_return_pct=(final_balance / self.initial_balance - 1) * 100,
            total_trades=n, win_count=int(wins), loss_count=int(losses),
            win_rate=wr, profit_factor=pf, avg_r=avg_r,
            avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
            max_drawdown_pct=-max_dd, max_drawdown_duration_bars=dd_duration,
            sharpe_ratio=sharpe, sortino_ratio=sortino, calmar_ratio=calmar,
            expectancy=expectancy,
            avg_bars_held=avg_bars, avg_mfe_r=avg_mfe, avg_mae_r=avg_mae,
            avg_capture_ratio=avg_capture, total_slippage_cost=total_slippage,
            trades=trades, equity_curve=eq.tolist(),
            monthly=monthly, by_hour=by_hour,
            by_regime=by_regime, by_exit_reason=by_exit,
            by_direction=by_dir,
        )

    # ── Breakdowns ────────────────────────────────────────────────────────

    @staticmethod
    def _monthly_breakdown(trades, ds):
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            if t.entry_time is None:
                continue
            month_key = f"{t.entry_time.year}-{t.entry_time.month:02d}"
            rows.append({'month': month_key, 'pnl': t.pnl_dollar, 'r': t.pnl_r,
                         'direction': 'LONG' if t.direction == 1 else 'SHORT'})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.groupby('month').agg(
            trades=('pnl', 'count'), pnl=('pnl', 'sum'),
            avg_r=('r', 'mean'), wr=('r', lambda x: (x > 0).mean() * 100),
        ).round(2)

    @staticmethod
    def _hourly_breakdown(trades):
        if not trades:
            return pd.DataFrame()
        rows = [{'hour': t.entry_time.hour if t.entry_time else 0,
                 'pnl': t.pnl_dollar, 'r': t.pnl_r} for t in trades]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.groupby('hour').agg(
            trades=('pnl', 'count'), pnl=('pnl', 'sum'),
            avg_r=('r', 'mean'),
        ).round(2)

    @staticmethod
    def _regime_breakdown(trades):
        if not trades:
            return pd.DataFrame()
        rows = [{'regime': t.regime_at_entry or 'unknown',
                 'pnl': t.pnl_dollar, 'r': t.pnl_r} for t in trades]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.groupby('regime').agg(
            trades=('pnl', 'count'), pnl=('pnl', 'sum'),
            avg_r=('r', 'mean'), wr=('r', lambda x: (x > 0).mean() * 100),
        ).round(2)

    @staticmethod
    def _exit_reason_breakdown(trades):
        if not trades:
            return pd.DataFrame()
        rows = [{'reason': t.exit_reason or 'unknown',
                 'pnl': t.pnl_dollar, 'r': t.pnl_r} for t in trades]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.groupby('reason').agg(
            trades=('pnl', 'count'), pnl=('pnl', 'sum'),
            avg_r=('r', 'mean'),
        ).round(2)

    # ── Print ─────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(r: BacktestResult):
        print(f"\n{'='*65}")
        print(f"BACKTEST RESULTS  ({r.start_date} → {r.end_date})")
        print(f"{'='*65}")
        print(f"Balance:    ${r.initial_balance:,.0f} → ${r.final_balance:,.0f}  "
              f"({r.total_return_pct:+.1f}%)")
        print(f"Trades:     {r.total_trades}  |  Win: {r.win_count}  Loss: {r.loss_count}  "
              f"WR: {r.win_rate:.1f}%")
        print(f"PF:         {r.profit_factor:.2f}  |  Avg R: {r.avg_r:+.3f}  "
              f"(win {r.avg_win_r:+.3f} / loss {r.avg_loss_r:+.3f})")
        print(f"Max DD:     {r.max_drawdown_pct:.1f}%  |  DD duration: {r.max_drawdown_duration_bars} bars")
        print(f"Sharpe:     {r.sharpe_ratio:.2f}  Sortino: {r.sortino_ratio:.2f}  "
              f"Calmar: {r.calmar_ratio:.2f}")
        print(f"Expectancy: {r.expectancy:+.3f}R  |  Avg hold: {r.avg_bars_held:.1f} bars")
        print(f"MFE:        {r.avg_mfe_r:+.3f}R  |  MAE: {r.avg_mae_r:+.3f}R  "
              f"Capture: {r.avg_capture_ratio:.1%}")
        print(f"Slippage:   ${r.total_slippage_cost:,.0f}")

        if r.by_regime is not None and not r.by_regime.empty:
            print(f"\n-- By Regime --")
            print(r.by_regime.to_string())

        if r.by_exit_reason is not None and not r.by_exit_reason.empty:
            print(f"\n-- By Exit Reason --")
            print(r.by_exit_reason.to_string())

        # Print last 10 trades for inspection
        if r.trades:
            print(f"\n-- Recent Trades --")
            print(f"{'Entry':20s} {'Dir':5s} {'Entry$':>9s} {'Exit$':>9s} "
                  f"{'PnL':>9s} {'R':>7s} {'MFE':>6s} {'Hold':>5s} {'Exit':12s} {'Regime':15s}")
            print(f"{'─'*100}")
            for t in r.trades[-10:]:
                ts = str(t.entry_time)[:19] if t.entry_time else ''
                print(f"{ts:20s} {'LONG' if t.direction==1 else 'SHORT':5s} "
                      f"${t.entry_price:>8.1f} ${t.exit_price:>8.1f} "
                      f"${t.pnl_dollar:>8.1f} {t.pnl_r:>+6.3f} {t.mfe_r:>+6.3f} "
                      f"{t.bars_held:>4d}h {t.exit_reason:12s} {t.regime_at_entry:15s}")
