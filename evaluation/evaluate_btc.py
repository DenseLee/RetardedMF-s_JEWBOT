"""
Comprehensive BTC Bot evaluation.
Computes per-trade and aggregate metrics, equity curve, regime breakdown.

Usage:
    python evaluation/evaluate_btc.py --trades trades.json
    python evaluation/evaluate_btc.py --from 2026-01-01 --to 2026-05-01
"""
import os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig


@dataclass
class TradeMetrics:
    pnl_dollar: float
    pnl_r: float
    mfe_r: float
    mae_r: float
    bars_held: int
    exit_reason: str
    direction: str
    entry_price: float
    exit_price: float
    capture_ratio: float = 0.0   # pnl / mfe

    def __post_init__(self):
        if self.mfe_r > 0.01:
            self.capture_ratio = self.pnl_r / self.mfe_r


@dataclass
class AggregateMetrics:
    total_trades: int = 0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_bars_held: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    avg_capture: float = 0.0
    total_r: float = 0.0
    # By regime
    by_regime: dict = field(default_factory=dict)
    # By confidence decile
    by_confidence: dict = field(default_factory=dict)


class BTCEvaluator:
    """Computes comprehensive trading metrics."""

    def __init__(self, config: BTCConfig = None):
        self.config = config or BTCConfig()

    def load_trades(self, trades_data) -> List[TradeMetrics]:
        """Load trades from list of dicts or JSON file path."""
        if isinstance(trades_data, str):
            with open(trades_data) as f:
                raw = json.load(f)
        else:
            raw = trades_data

        trades = []
        for t in raw:
            trades.append(TradeMetrics(
                pnl_dollar=t.get("pnl_dollar", 0),
                pnl_r=t.get("pnl_r", 0),
                mfe_r=t.get("mfe_r", 0),
                mae_r=t.get("mae_r", 0),
                bars_held=t.get("bars_held", 0),
                exit_reason=t.get("exit_reason", "unknown"),
                direction=t.get("direction", "LONG"),
                entry_price=t.get("entry_price", 0),
                exit_price=t.get("exit_price", 0),
            ))
        return trades

    def compute_aggregate(self, trades: List[TradeMetrics]) -> AggregateMetrics:
        """Compute aggregate metrics from trade list."""
        if not trades:
            return AggregateMetrics()

        m = AggregateMetrics()
        m.total_trades = len(trades)
        m.total_pnl = sum(t.pnl_dollar for t in trades)

        wins = [t for t in trades if t.pnl_dollar > 0]
        losses = [t for t in trades if t.pnl_dollar <= 0]

        m.win_count = len(wins)
        m.loss_count = len(losses)
        m.win_rate = m.win_count / m.total_trades if m.total_trades else 0
        m.avg_win = np.mean([t.pnl_dollar for t in wins]) if wins else 0
        m.avg_loss = np.mean([t.pnl_dollar for t in losses]) if losses else 0

        m.total_r = sum(t.pnl_r for t in trades)
        m.avg_r = m.total_r / m.total_trades if m.total_trades else 0

        total_gains = sum(t.pnl_dollar for t in wins)
        total_losses = abs(sum(t.pnl_dollar for t in losses))
        m.profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

        m.avg_bars_held = np.mean([t.bars_held for t in trades])
        m.avg_mfe_r = np.mean([t.mfe_r for t in trades])
        m.avg_mae_r = np.mean([abs(t.mae_r) for t in trades])
        m.avg_capture = np.mean([t.capture_ratio for t in trades])

        # Sharpe (approximate, using R-multiples)
        r_returns = [t.pnl_r for t in trades]
        if len(r_returns) > 1:
            m.sharpe_ratio = np.mean(r_returns) / max(np.std(r_returns), 0.01)

        # Max drawdown from cumulative PnL
        cumulative = np.cumsum([t.pnl_dollar for t in trades])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / max(peak, 1)
        m.max_drawdown_pct = float(np.max(drawdowns)) * 100 if len(drawdowns) > 0 else 0

        # By exit reason
        m.by_regime = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in m.by_regime:
                m.by_regime[reason] = {"count": 0, "total_pnl": 0.0, "avg_r": 0.0}
            m.by_regime[reason]["count"] += 1
            m.by_regime[reason]["total_pnl"] += t.pnl_dollar

        for reason in m.by_regime:
            c = m.by_regime[reason]["count"]
            m.by_regime[reason]["avg_pnl"] = m.by_regime[reason]["total_pnl"] / c if c else 0

        return m

    def compute_equity_curve(self, trades: List[TradeMetrics],
                             initial_balance=10000.0) -> np.ndarray:
        """Compute equity curve from trade history."""
        equity = np.zeros(len(trades) + 1)
        equity[0] = initial_balance
        for i, t in enumerate(trades):
            equity[i + 1] = equity[i] + t.pnl_dollar
        return equity

    def report(self, metrics: AggregateMetrics) -> str:
        """Format a readable report string."""
        lines = []
        lines.append("=" * 60)
        lines.append("BTC BOT EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"  Total Trades:    {metrics.total_trades}")
        lines.append(f"  Wins:            {metrics.win_count} ({metrics.win_rate*100:.1f}%)")
        lines.append(f"  Losses:          {metrics.loss_count}")
        lines.append(f"  Total PnL:       ${metrics.total_pnl:+,.2f}")
        lines.append(f"  Total R:         {metrics.total_r:+.2f}R")
        lines.append(f"  Avg R/trade:     {metrics.avg_r:+.3f}R")
        lines.append(f"  Avg Win:         ${metrics.avg_win:+,.2f}")
        lines.append(f"  Avg Loss:        ${metrics.avg_loss:+,.2f}")
        lines.append(f"  Profit Factor:   {metrics.profit_factor:.2f}")
        lines.append(f"  Sharpe (R):      {metrics.sharpe_ratio:.2f}")
        lines.append(f"  Max Drawdown:    {metrics.max_drawdown_pct:.1f}%")
        lines.append(f"  Avg Bars Held:   {metrics.avg_bars_held:.1f}")
        lines.append(f"  Avg MFE:         {metrics.avg_mfe_r:.2f}R")
        lines.append(f"  Avg |MAE|:       {metrics.avg_mae_r:.2f}R")
        lines.append(f"  Avg Capture:     {metrics.avg_capture*100:.1f}%")
        lines.append("")
        lines.append("By Exit Reason:")
        for reason, stats in sorted(metrics.by_regime.items()):
            lines.append(f"  {reason:>20s}: {stats['count']:4d} trades, "
                        f"avg ${stats['avg_pnl']:+,.2f}")
        lines.append("=" * 60)

        # Expected value check
        expected_r = metrics.avg_r
        if expected_r > 0.3:
            lines.append("VERDICT: Profitable system (avg R > 0.3)")
        elif expected_r > 0:
            lines.append("VERDICT: Marginally profitable")
        else:
            lines.append("VERDICT: Unprofitable — needs tuning")
        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example: evaluate from trade history
    config = BTCConfig()
    evaluator = BTCEvaluator(config)

    # Sample trades for demonstration
    sample_trades = [
        {"pnl_dollar": 1500, "pnl_r": 3.0, "mfe_r": 3.5, "mae_r": -0.8,
         "bars_held": 6, "exit_reason": "tp_hit", "direction": "LONG",
         "entry_price": 80000, "exit_price": 81500},
        {"pnl_dollar": -800, "pnl_r": -0.8, "mfe_r": 0.3, "mae_r": -1.5,
         "bars_held": 4, "exit_reason": "mae_guard", "direction": "LONG",
         "entry_price": 82000, "exit_price": 81200},
        {"pnl_dollar": 2100, "pnl_r": 2.8, "mfe_r": 3.2, "mae_r": -0.5,
         "bars_held": 10, "exit_reason": "trailing_stop", "direction": "SHORT",
         "entry_price": 85000, "exit_price": 82900},
        {"pnl_dollar": -700, "pnl_r": -0.7, "mfe_r": 0.5, "mae_r": -1.2,
         "bars_held": 3, "exit_reason": "sl_hit", "direction": "SHORT",
         "entry_price": 83000, "exit_price": 83700},
        {"pnl_dollar": 1200, "pnl_r": 2.0, "mfe_r": 2.4, "mae_r": -0.9,
         "bars_held": 7, "exit_reason": "scale_out", "direction": "LONG",
         "entry_price": 80000, "exit_price": 81200},
    ]

    trades = evaluator.load_trades(sample_trades)
    metrics = evaluator.compute_aggregate(trades)
    print(evaluator.report(metrics))
