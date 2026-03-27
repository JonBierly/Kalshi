"""
P&L breakdown by model confidence bucket.

For each probability bucket (50-55%, 55-60%, …) shows:
  - Trade count
  - Actual win rate vs model-stated probability  (calibration)
  - Average and total realized P&L
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

DB_PATH = 'data/nba_data.db'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    type=str, default="2026-02-09", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",      type=str, default=None,         help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-edge", type=float, default=2.0,        help="Min CI-based edge (cents)")
    parser.add_argument("--output",   type=str, default="reports/pnl_by_prob.png")
    return parser.parse_args()


def analyze_pnl_by_prob(start_date=None, end_date=None, min_edge=2.0, output_path="reports/pnl_by_prob.png"):
    if not os.path.exists(DB_PATH):
        print("Database not found"); return

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT trade_id, side, fill_price, realized_pnl, size,
               model_fair_value, model_ci_lower, model_ci_upper, created_at
        FROM trades
        WHERE realized_pnl IS NOT NULL AND size > 0
    """
    conditions, args = [], []
    if start_date:
        conditions.append("DATE(created_at) >= ?"); args.append(start_date)
    if end_date:
        conditions.append("DATE(created_at) <= ?"); args.append(end_date)
    if conditions:
        query += " AND " + " AND ".join(conditions)

    df = pd.read_sql_query(query, conn, params=args)
    conn.close()

    if df.empty:
        print("No trades found"); return

    # CI-based edge
    has_ci  = df['model_ci_lower'].notna() & df['model_ci_upper'].notna()
    ci_edge = np.where(df['side'] == 'buy',
                       df['model_ci_lower'] - df['fill_price'],
                       df['fill_price'] - df['model_ci_upper'])
    mean_edge = abs(df['model_fair_value'] - df['fill_price'])
    df['edge'] = np.where(has_ci, ci_edge, mean_edge)
    df = df[df['edge'] >= min_edge].copy()

    if df.empty:
        print(f"No trades with edge >= {min_edge}¢"); return

    # Bet probability & win flag
    df['bet_prob'] = np.where(df['side'] == 'buy',
                              df['model_fair_value'],
                              100 - df['model_fair_value'])
    df['win'] = (df['realized_pnl'] > 0).astype(int)

    # Bucket into 5-point bins starting at 50
    bins   = list(range(50, 105, 5))
    labels = [f"{b}-{b+5}%" for b in bins[:-1]]
    df['prob_bin'] = pd.cut(df['bet_prob'], bins=bins, labels=labels, include_lowest=True)

    # Per-bin summary
    summary = df.groupby('prob_bin', observed=True).agg(
        n          =('win', 'count'),
        wins       =('win', 'sum'),
        model_prob =('bet_prob', 'mean'),
        avg_pnl    =('realized_pnl', 'mean'),
        total_pnl  =('realized_pnl', 'sum'),
    ).reset_index()
    summary['actual_win_rate'] = summary['wins'] / summary['n']
    summary['expected_win_rate'] = summary['model_prob'] / 100
    summary['calibration_diff']  = summary['actual_win_rate'] - summary['expected_win_rate']

    print(f"\n{'='*75}")
    print(f"  P&L BY CONFIDENCE BUCKET  |  min_edge={min_edge}¢  |  {start_date or 'all'} →")
    print(f"{'='*75}")
    print(f"  {'Bucket':<11} {'N':>5} {'Model%':>8} {'Actual%':>8} {'CalDiff':>8}  "
          f"{'AvgPnL':>8}  {'TotalPnL':>9}")
    for _, r in summary.iterrows():
        flag = " ←" if abs(r['calibration_diff']) > 0.07 and r['n'] >= 20 else ""
        print(f"  {str(r['prob_bin']):<11} {r['n']:>5} "
              f"{r['expected_win_rate']:>8.1%} {r['actual_win_rate']:>8.1%} "
              f"{r['calibration_diff']:>+8.1%}  "
              f"${r['avg_pnl']:>+7.3f}  ${r['total_pnl']:>+8.2f}{flag}")

    # Plots
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Performance by Model Confidence  |  min_edge={min_edge}¢  |  {start_date or 'all'} →",
        fontsize=13, fontweight='bold'
    )

    # 1. Calibration
    valid = summary[summary['n'] >= 10]
    axes[0].plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, label='Perfect')
    axes[0].scatter(valid['expected_win_rate'], valid['actual_win_rate'],
                    s=valid['n'] * 2, color='#8e44ad', zorder=5)
    for _, r in valid.iterrows():
        axes[0].annotate(f"n={r['n']}", (r['expected_win_rate'], r['actual_win_rate']),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
    axes[0].set_xlim(0.48, 1.02); axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel('Model probability'); axes[0].set_ylabel('Actual win rate')
    axes[0].set_title('Calibration')
    axes[0].legend()

    # 2. Avg P&L per trade
    colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in summary['avg_pnl']]
    axes[1].bar(range(len(summary)), summary['avg_pnl'], color=colors)
    axes[1].set_xticks(range(len(summary)))
    axes[1].set_xticklabels(summary['prob_bin'].astype(str), rotation=45, ha='right', fontsize=8)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title('Avg P&L per Trade by Confidence')
    axes[1].set_ylabel('Avg Realized P&L ($)')

    # 3. Total P&L
    colors2 = ['#27ae60' if v >= 0 else '#e74c3c' for v in summary['total_pnl']]
    axes[2].bar(range(len(summary)), summary['total_pnl'], color=colors2)
    axes[2].set_xticks(range(len(summary)))
    axes[2].set_xticklabels(summary['prob_bin'].astype(str), rotation=45, ha='right', fontsize=8)
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].set_title('Total P&L by Confidence')
    axes[2].set_ylabel('Total Realized P&L ($)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved → {output_path}")


if __name__ == "__main__":
    args = parse_args()
    analyze_pnl_by_prob(
        start_date=args.start,
        end_date=args.end,
        min_edge=args.min_edge,
        output_path=args.output,
    )
