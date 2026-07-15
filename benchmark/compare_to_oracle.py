"""Score model trades against oracle benchmark."""
import sys, os, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, os.path.dirname(BENCH_DIR))
from oracle_labeler import OracleLabel, OracleLabeler


def compare(trades, oracle_labels):
    """Compare model trades to oracle. Returns metrics dict."""
    oracle_by_hour = {}
    for ol in oracle_labels:
        oracle_by_hour[ol.timestamp[:13]] = ol

    n = len(trades)
    metrics = {
        'total': n, 'correct_dir': 0, 'wrong_dir': 0,
        'entered_chop': 0, 'exited_early': 0, 'caught_tp': 0,
        'details': [],
    }

    for t in trades:
        hour_key = t['entry_ts'][:13]
        ol = oracle_by_hour.get(hour_key)
        if ol is None:
            for dh in [-1, 1]:
                adj = (pd.Timestamp(t['entry_ts']) + timedelta(hours=dh)).strftime('%Y-%m-%d %H')
                ol = oracle_by_hour.get(adj)
                if ol: break
        if ol is None:
            continue

        mdir = 1 if t.get('direction', '') == 'LONG' else -1
        mr = t.get('pnl_r', 0); mpnl = t.get('pnl_d', 0); mex = t.get('exit_reason', '')

        d = {'ts': t['entry_ts'][:19], 'dir': t.get('direction','?'),
             'model_r': mr, 'model_pnl': mpnl, 'exit': mex,
             'oracle_label': ol.label, 'oracle_dir': ol.optimal_dir,
             'oracle_r': ol.optimal_r, 'oracle_long': ol.best_long_r,
             'oracle_short': ol.best_short_r}

        if ol.label == 'CHOP':
            metrics['entered_chop'] += 1
            d['verdict'] = 'CHOP'
        elif ol.optimal_dir != 0 and mdir != ol.optimal_dir:
            metrics['wrong_dir'] += 1
            d['verdict'] = 'WRONG_DIR'
        elif ol.optimal_dir != 0 and mdir == ol.optimal_dir:
            metrics['correct_dir'] += 1
            if mr < 0.5 and ol.optimal_r > 1.0:
                metrics['exited_early'] += 1
                d['verdict'] = 'EARLY_EXIT'
            elif mr >= 0.5:
                d['verdict'] = 'GOOD'
                if 'tp' in mex.lower():
                    metrics['caught_tp'] += 1
            else:
                d['verdict'] = 'OK_DIR'
        else:
            d['verdict'] = 'NO_SIGNAL'

        metrics['details'].append(d)

    # Compute rates
    if n > 0:
        metrics['correct_pct'] = metrics['correct_dir'] / n * 100
        metrics['wrong_pct'] = metrics['wrong_dir'] / n * 100
        metrics['chop_pct'] = metrics['entered_chop'] / n * 100
        metrics['early_pct'] = metrics['exited_early'] / n * 100
    return metrics


def print_report(m):
    n = m['total']
    print()
    print('=' * 60)
    print('MODEL vs ORACLE BENCHMARK')
    print('=' * 60)
    print()
    print('Direction Accuracy:')
    print('  Correct:    {:>3d} / {} ({:.1f}%)'.format(m['correct_dir'], n, m.get('correct_pct', 0)))
    print('  Wrong dir:  {:>3d} / {} ({:.1f}%)'.format(m['wrong_dir'], n, m.get('wrong_pct', 0)))
    print('  During CHOP: {:>3d} / {} ({:.1f}%)'.format(m['entered_chop'], n, m.get('chop_pct', 0)))
    print()
    print('Exit Quality:')
    print('  Exited early: {:>3d} ({:.1f}%) — oracle had >1R, model got <0.5R'.format(
        m['exited_early'], m.get('early_pct', 0)))
    print('  Caught TP:    {:>3d}'.format(m['caught_tp']))
    print()

    # Show examples
    details = m['details']
    wrong = [d for d in details if d['verdict'] == 'WRONG_DIR'][:3]
    early = [d for d in details if d['verdict'] == 'EARLY_EXIT'][:3]
    chop = [d for d in details if d['verdict'] == 'CHOP'][:3]

    if wrong:
        print('Wrong direction examples:')
        for d in wrong:
            print('  {} {} oracle={} ({}R avail) model={:+.2f}R'.format(
                d['ts'], d['dir'], d['oracle_label'], d['oracle_r'], d['model_r']))
        print()

    if early:
        print('Exited too early examples:')
        for d in early:
            print('  {} {} oracle had {:+.1f}R → model got {:+.2f}R ({})'.format(
                d['ts'], d['dir'], d['oracle_r'], d['model_r'], d['exit']))
        print()

    if chop:
        print('Entered during CHOP examples:')
        for d in chop:
            print('  {} {} oracle=CHOP model={:+.2f}R'.format(
                d['ts'], d['dir'], d['model_r']))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', required=True)
    p.add_argument('--trades-csv')
    p.add_argument('--from', dest='start', default='2026-04-25')
    p.add_argument('--to', dest='end', default='2026-05-25')

    args = p.parse_args()

    # Load oracle
    labels = OracleLabeler.load(args.oracle)
    print('Oracle: {} labels loaded'.format(len(labels)))

    if args.trades_csv:
        df = pd.read_csv(args.trades_csv)
        trades = df.to_dict('records')
        print('Trades: {} loaded from {}'.format(len(trades), args.trades_csv))
    else:
        # Run backtest
        from backtest.data_manager import BacktestDataManager
        from backtest.backtester_btc import BTCBacktester
        from backtest.slippage_model import SlippageConfig
        from config_btc import BTCConfig

        cfg = BTCConfig()
        dm = BacktestDataManager(cfg)
        ds = dm.prepare(args.start, args.end, use_cache=True)
        bt = BTCBacktester(cfg, initial_balance=10000.0,
                           slippage=SlippageConfig(
                               entry_slippage_pct=0.0, exit_slippage_pct=0.0,
                               sl_slippage_pct=0.0, tp_slippage_pct=0.0))
        result = bt.run(ds, verbose=False)
        trades = []
        for t in result.trades:
            if t.entry_time is None: continue
            trades.append({
                'entry_ts': str(t.entry_time)[:19],
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price,
                'pnl_r': t.pnl_r, 'pnl_d': t.pnl_dollar,
                'exit_reason': t.exit_reason,
            })
        print('Backtest: {} trades'.format(len(trades)))

    m = compare(trades, labels)
    print_report(m)

    # Monthly breakdown
    print()
    print('=== BY MONTH ===')
    details = m['details']
    if details:
        df = pd.DataFrame(details)
        df['month'] = pd.to_datetime(df['ts']).dt.strftime('%Y-%m')
        for month, grp in df.groupby('month'):
            n_m = len(grp)
            correct = (grp['verdict'].isin(['GOOD', 'OK_DIR', 'EARLY_EXIT'])).sum()
            wrong = (grp['verdict'] == 'WRONG_DIR').sum()
            chop = (grp['verdict'] == 'CHOP').sum()
            early = (grp['verdict'] == 'EARLY_EXIT').sum()
            pnl = grp['model_pnl'].sum()
            print('  {}: {} trades  correct={}  wrong={}  chop={}  early_exit={}  PnL=${:+.1f}'.format(
                month, n_m, correct, wrong, chop, early, pnl))
