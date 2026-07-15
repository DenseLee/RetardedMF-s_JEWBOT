"""
BTC Backtest CLI — drip-feed simulation with M1 intrabar resolution.

Usage:
    python -m BTC_BOT.backtest.run_backtest --from 2026-01-01 --to 2026-05-22
    python -m BTC_BOT.backtest.run_backtest --from 2026-01-01 --to 2026-05-22 --sl 0.8 --tp 1.6
    python -m BTC_BOT.backtest.run_backtest --from 2026-01-01 --to 2026-05-22 --slippage 0.05
    python -m BTC_BOT.backtest.run_backtest --compare cfg_a.json cfg_b.json
    python -m BTC_BOT.backtest.run_backtest --grid-search --sl "0.5,0.6,0.8,1.0" --tp "1.2,1.4,1.6,1.8"
    python -m BTC_BOT.backtest.run_backtest --no-cache
"""
import argparse, json, os, sys, time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_btc import BTCConfig
from .data_manager import BacktestDataManager
from .backtester_btc import BTCBacktester
from .slippage_model import SlippageConfig


def parse_args():
    p = argparse.ArgumentParser(description="BTC Backtest Framework")
    p.add_argument("--from", dest="from_date", default="2026-01-01",
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to", dest="to_date", default="2026-05-22",
                   help="End date YYYY-MM-DD")
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--sl", type=str, help="Override initial_sl ATR multiplier")
    p.add_argument("--tp", type=str, help="Override hard_tp ATR multiplier")
    p.add_argument("--risk", type=str, help="Override risk_pct")
    p.add_argument("--trail-trigger", type=float, help="Override trail_trigger")
    p.add_argument("--trail-dist", type=float, help="Override trail_dist")
    p.add_argument("--breakeven", type=float, help="Override breakeven_trigger")
    p.add_argument("--max-hold", type=int, help="Override max_hold_bars")
    p.add_argument("--slippage", type=float, default=0.02, help="Slippage % (default 0.02)")
    p.add_argument("--rule-only", action="store_true", help="Use rule-based regime only")
    p.add_argument("--no-cache", action="store_true", help="Force re-fetch & re-compute")
    p.add_argument("--compare", nargs=2, help="Two config JSONs to A/B compare")
    p.add_argument("--grid-search", action="store_true", help="Run grid search over parameters")
    p.add_argument("--save-results", help="Path to save JSON results")
    p.add_argument("--save-trades", help="Path to save trades CSV")
    return p.parse_args()


def make_config(args) -> BTCConfig:
    cfg = BTCConfig()
    if args.sl is not None and ',' not in args.sl:
        cfg.initial_sl = float(args.sl)
    if args.tp is not None and ',' not in args.tp:
        cfg.hard_tp = float(args.tp)
    if args.risk is not None and ',' not in args.risk:
        cfg.risk_pct = float(args.risk)
    if args.trail_trigger is not None:
        cfg.trail_trigger = args.trail_trigger
    if args.trail_dist is not None:
        cfg.trail_dist = args.trail_dist
        cfg.trail_dist_s = args.trail_dist * 0.67
    if args.breakeven is not None:
        cfg.breakeven_trigger = args.breakeven
    if args.max_hold is not None:
        cfg.max_hold_bars = args.max_hold
    return cfg


def run_single(args, config_override=None):
    cfg = config_override or make_config(args)
    slip = SlippageConfig(
        entry_slippage_pct=args.slippage,
        exit_slippage_pct=args.slippage,
        sl_slippage_pct=args.slippage / 2,
        tp_slippage_pct=args.slippage / 2)

    print(f"\n{'='*65}")
    print(f"BTC Backtest: {args.from_date} → {args.to_date}")
    print(f"SL={cfg.initial_sl}  TP={cfg.hard_tp}  Risk={cfg.risk_pct}  "
          f"Trail={cfg.trail_trigger}  BE={cfg.breakeven_trigger}  "
          f"MaxHold={cfg.max_hold_bars}")
    print(f"Slippage={args.slippage}% | Balance=${args.initial_balance:,.0f}")
    print(f"{'='*65}")

    dm = BacktestDataManager(cfg)
    dataset = dm.prepare(args.from_date, args.to_date,
                         use_cache=not args.no_cache,
                         force_refresh=args.no_cache)

    bt = BTCBacktester(cfg, initial_balance=args.initial_balance, slippage=slip)
    result = bt.run(dataset, verbose=True)

    if args.save_results:
        _save_results(result, args.save_results)
    if args.save_trades:
        _save_trades_csv(result, args.save_trades)

    return result


def run_compare(args):
    """A/B compare two config files."""
    cfg_a = _load_config_json(args.compare[0])
    cfg_b = _load_config_json(args.compare[1])

    print(f"\n{'='*65}")
    print(f"A/B COMPARE: {args.compare[0]} vs {args.compare[1]}")
    print(f"{'='*65}")

    dm = BacktestDataManager(BTCConfig())
    dataset = dm.prepare(args.from_date, args.to_date,
                         use_cache=not args.no_cache,
                         force_refresh=args.no_cache)

    results = {}
    for label, cfg in [("A", cfg_a), ("B", cfg_b)]:
        print(f"\n--- Config {label} ---")
        bt = BTCBacktester(cfg, initial_balance=args.initial_balance,
                           slippage=SlippageConfig(entry_slippage_pct=args.slippage))
        results[label] = bt.run(dataset, verbose=False)

    _print_comparison(results["A"], results["B"], args.compare)


def run_grid_search(args):
    """Grid search over SL and TP combinations."""
    sl_vals = _parse_float_list(args.sl, "0.6,0.8,1.0,1.2")
    tp_vals = _parse_float_list(args.tp, "1.2,1.4,1.6,2.0")
    risk_vals = _parse_float_list(args.risk, str(BTCConfig().risk_pct))

    print(f"\n{'='*65}")
    print(f"GRID SEARCH: {len(sl_vals)}×{len(tp_vals)}×{len(risk_vals)} = "
          f"{len(sl_vals)*len(tp_vals)*len(risk_vals)} combinations")
    print(f"SL: {sl_vals}  TP: {tp_vals}  Risk: {risk_vals}")
    print(f"{'='*65}")

    dm = BacktestDataManager(BTCConfig())
    dataset = dm.prepare(args.from_date, args.to_date,
                         use_cache=not args.no_cache,
                         force_refresh=args.no_cache)

    rows = []
    best = None
    best_score = -999

    for sl, tp, risk in product(sl_vals, tp_vals, risk_vals):
        cfg = BTCConfig()
        cfg.initial_sl = sl
        cfg.hard_tp = tp
        cfg.risk_pct = risk

        bt = BTCBacktester(cfg, initial_balance=args.initial_balance,
                           slippage=SlippageConfig(entry_slippage_pct=args.slippage))
        r = bt.run(dataset, verbose=False)

        # Score: blend of PF and total return
        score = r.profit_factor * (1 + r.total_return_pct / 100)
        rows.append({
            'sl': sl, 'tp': tp, 'risk': risk,
            'trades': r.total_trades, 'wr': round(r.win_rate, 1),
            'pf': round(r.profit_factor, 2), 'avg_r': round(r.avg_r, 4),
            'pnl': round(r.total_pnl, 0), 'dd': round(r.max_drawdown_pct, 1),
            'sharpe': round(r.sharpe_ratio, 2), 'score': round(score, 2),
        })

        if score > best_score and r.total_trades >= 5:
            best_score = score
            best = (sl, tp, risk, r)

    # Print sorted results
    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    print(f"\n{'─'*85}")
    print(f"{'SL':>6s} {'TP':>6s} {'Risk':>6s} {'Trades':>7s} {'WR':>6s} "
          f"{'PF':>6s} {'AvgR':>7s} {'PnL':>8s} {'DD':>6s} {'Sharpe':>7s} {'Score':>7s}")
    print(f"{'─'*85}")
    for _, row in df.iterrows():
        print(f"{row['sl']:>6.1f} {row['tp']:>6.1f} {row['risk']:>6.2f} "
              f"{int(row['trades']):>7d} {row['wr']:>5.1f}% {row['pf']:>6.2f} "
              f"{row['avg_r']:>+7.3f} ${row['pnl']:>7.0f} {row['dd']:>5.1f}% "
              f"{row['sharpe']:>7.2f} {row['score']:>7.2f}")

    if best:
        print(f"\nBest: SL={best[0]}, TP={best[1]}, Risk={best[2]}")
        print(f"  PnL=${best[3].total_pnl:,.0f}  WR={best[3].win_rate:.1f}%  "
              f"PF={best[3].profit_factor:.2f}  DD={best[3].max_drawdown_pct:.1f}%")


# ── helpers ───────────────────────────────────────────────────────────────

def _load_config_json(path):
    cfg = BTCConfig()
    with open(path) as f:
        overrides = json.load(f)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def _parse_float_list(arg, default_str):
    if arg is None:
        return [float(x.strip()) for x in default_str.split(",")]
    return [float(x.strip()) for x in arg.split(",")]

def _save_results(result, path):
    data = {
        'start': result.start_date, 'end': result.end_date,
        'initial_balance': result.initial_balance,
        'final_balance': result.final_balance,
        'total_pnl': result.total_pnl,
        'total_return_pct': result.total_return_pct,
        'total_trades': result.total_trades,
        'win_count': result.win_count,
        'loss_count': result.loss_count,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'avg_r': result.avg_r,
        'avg_win_r': result.avg_win_r,
        'avg_loss_r': result.avg_loss_r,
        'max_drawdown_pct': result.max_drawdown_pct,
        'max_drawdown_duration_bars': result.max_drawdown_duration_bars,
        'sharpe_ratio': result.sharpe_ratio,
        'sortino_ratio': result.sortino_ratio,
        'calmar_ratio': result.calmar_ratio,
        'expectancy': result.expectancy,
        'avg_bars_held': result.avg_bars_held,
        'avg_mfe_r': result.avg_mfe_r,
        'avg_mae_r': result.avg_mae_r,
        'total_slippage_cost': result.total_slippage_cost,
        'by_direction': result.by_direction,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {path}")

def _save_trades_csv(result, path):
    rows = []
    for t in result.trades:
        rows.append({
            'entry_time': t.entry_time.isoformat() if t.entry_time else '',
            'exit_time': t.exit_time.isoformat() if t.exit_time else '',
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_price': t.entry_price, 'exit_price': t.exit_price,
            'lots': t.lots, 'pnl_dollar': t.pnl_dollar, 'pnl_r': t.pnl_r,
            'mfe_r': t.mfe_r, 'mae_r': t.mae_r, 'bars_held': t.bars_held,
            'exit_reason': t.exit_reason, 'regime_at_entry': t.regime_at_entry,
            'confidence': t.confidence_at_entry,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trades saved to {path}")

def _print_comparison(r_a, r_b, labels):
    print(f"\n{'='*85}")
    print(f"COMPARISON: {labels[0]} vs {labels[1]}")
    print(f"{'='*85}")
    print(f"{'Metric':25s} {'Config A':>20s} {'Config B':>20s} {'Delta':>15s}")
    print(f"{'─'*85}")
    metrics = [
        ('Total PnL', f"${r_a.total_pnl:,.0f}", f"${r_b.total_pnl:,.0f}",
         f"${r_b.total_pnl - r_a.total_pnl:+,.0f}"),
        ('Win Rate', f"{r_a.win_rate:.1f}%", f"{r_b.win_rate:.1f}%",
         f"{r_b.win_rate - r_a.win_rate:+.1f}%"),
        ('Profit Factor', f"{r_a.profit_factor:.2f}", f"{r_b.profit_factor:.2f}",
         f"{r_b.profit_factor - r_a.profit_factor:+.2f}"),
        ('Avg R', f"{r_a.avg_r:+.3f}", f"{r_b.avg_r:+.3f}",
         f"{r_b.avg_r - r_a.avg_r:+.3f}"),
        ('Trades', str(r_a.total_trades), str(r_b.total_trades),
         f"{r_b.total_trades - r_a.total_trades:+d}"),
        ('Max DD', f"{r_a.max_drawdown_pct:.1f}%", f"{r_b.max_drawdown_pct:.1f}%",
         f"{r_b.max_drawdown_pct - r_a.max_drawdown_pct:+.1f}%"),
        ('Sharpe', f"{r_a.sharpe_ratio:.2f}", f"{r_b.sharpe_ratio:.2f}",
         f"{r_b.sharpe_ratio - r_a.sharpe_ratio:+.2f}"),
        ('Slippage Cost', f"${r_a.total_slippage_cost:,.0f}",
         f"${r_b.total_slippage_cost:,.0f}",
         f"${r_b.total_slippage_cost - r_a.total_slippage_cost:+,.0f}"),
    ]
    for name, a, b, d in metrics:
        print(f"{name:25s} {a:>20s} {b:>20s} {d:>15s}")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        run_compare(args)
    elif args.grid_search:
        run_grid_search(args)
    else:
        run_single(args)
