#!/usr/bin/env python
"""
Analyze P&L drivers to identify profitability leaks and separate luck from model quality.

Analyses:
1. Daily P&L breakdown
2. P&L by market spread width / game phase / entry edge
3. Calibration: model probability vs actual win rate
4. Game-level P&L (corrects for correlated bets on the same game)
5. Cumulative expected vs actual P&L (are we within variance?)
6. Statistical significance: Z-score of actual returns
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import argparse
from datetime import datetime
from scipy import stats

sns.set_style("darkgrid")
DB_PATH = 'data/nba_data.db'


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze P&L drivers.")
    parser.add_argument("--start", type=str, default="2026-02-09", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   type=str, default=None,         help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-edge", type=float, default=2.0,     help="Min CI-based edge in cents")
    parser.add_argument("--output", type=str, default="reports/pnl_drivers_analysis.png")
    return parser.parse_args()


def load_trades(start_date=None, end_date=None, min_edge=2.0):
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT trade_id, timestamp, ticker, game_id, side, fill_price, size,
               model_fair_value, model_ci_lower, model_ci_upper,
               market_spread, seconds_remaining,
               realized_pnl, created_at
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
        return df

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date']       = df['created_at'].dt.date

    # CI-based edge (what triggered the trade)
    has_ci  = df['model_ci_lower'].notna() & df['model_ci_upper'].notna()
    ci_edge = np.where(df['side'] == 'buy',
                       df['model_ci_lower'] - df['fill_price'],
                       df['fill_price'] - df['model_ci_upper'])
    mean_edge = abs(df['model_fair_value'] - df['fill_price'])
    df['entry_edge'] = np.where(has_ci, ci_edge, mean_edge)
    df = df[df['entry_edge'] >= min_edge].copy()

    # Expected PnL per trade: (model_fair_value - fill_price) * size / 100 for buy,
    # flipped for sell.  Uses mean estimate so cumulative expected PnL is meaningful.
    df['expected_pnl'] = np.where(
        df['side'] == 'buy',
        (df['model_fair_value'] - df['fill_price']) * df['size'] / 100,
        (df['fill_price'] - df['model_fair_value']) * df['size'] / 100
    )

    # WIN flag (positive realized pnl = bet resolved in our favour)
    df['win'] = (df['realized_pnl'] > 0).astype(int)

    # "Bet probability" — the probability we're betting on being true
    df['bet_prob'] = np.where(
        df['side'] == 'buy',
        df['model_fair_value'],
        100 - df['model_fair_value']
    )

    # CI width (model uncertainty)
    df['ci_width'] = df['model_ci_upper'] - df['model_ci_lower']
    df['mins_remaining'] = df['seconds_remaining'] / 60.0

    return df


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_pnl(v):
    return f"${v:+.2f}"


def _wilson_ci(wins, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


# ── analysis ─────────────────────────────────────────────────────────────────

def analyze(df, output_path='reports/pnl_drivers_analysis.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_trades = len(df)
    total_pnl = df['realized_pnl'].sum()
    total_expected = df['expected_pnl'].sum()
    win_rate = df['win'].mean()
    n_games  = df['game_id'].nunique()

    print(f"\n{'='*60}")
    print(f"  POST-TRAINING PERFORMANCE REPORT")
    print(f"  {df['date'].min()} → {df['date'].max()}")
    print(f"{'='*60}")
    print(f"  Trades (post-edge filter):  {n_trades:,}")
    print(f"  Unique games:               {n_games}")
    print(f"  Avg trades/game:            {n_trades/max(n_games,1):.1f}  ← correlated bets")
    print(f"  Overall win rate:           {win_rate:.1%}")
    print(f"  Total realized P&L:         {_fmt_pnl(total_pnl)}")
    print(f"  Total expected P&L:         {_fmt_pnl(total_expected)}")
    print(f"  P&L vs expectation:         {_fmt_pnl(total_pnl - total_expected)}")

    # ── Statistical significance ──────────────────────────────────────────
    # Use game-level PnL as the unit (corrects for within-game correlation)
    game_pnl    = df.groupby('game_id')['realized_pnl'].sum()
    game_exp    = df.groupby('game_id')['expected_pnl'].sum()
    game_excess = game_pnl - game_exp

    n_g   = len(game_pnl)
    mu_g  = game_excess.mean()
    std_g = game_excess.std()
    se_g  = std_g / np.sqrt(n_g)
    z     = mu_g / se_g if se_g > 0 else 0
    p_val = 2 * stats.norm.sf(abs(z))   # two-tailed

    print(f"\n── Statistical Significance (game-level) ──────────────────")
    print(f"  Games analysed:             {n_g}")
    print(f"  Mean excess PnL/game:       {_fmt_pnl(mu_g)}  (actual − expected)")
    print(f"  Std dev PnL/game:           {_fmt_pnl(std_g)}")
    print(f"  Z-score:                    {z:+.2f}")
    print(f"  Two-tailed p-value:         {p_val:.3f}")
    if abs(z) < 1.65:
        print(f"  → Within normal variance (|Z| < 1.65). Could be luck.")
    elif abs(z) < 2.0:
        print(f"  → Borderline (1.65 < |Z| < 2.0). Worth monitoring.")
    else:
        print(f"  → Statistically significant (|Z| > 2.0). Likely structural.")

    # ── Bucketed breakdowns ───────────────────────────────────────────────
    df['spread_bucket'] = pd.cut(df['market_spread'],
                                 bins=[0, 10, 25, 50, 200],
                                 labels=['Tight\n(0-10)', 'Normal\n(10-25)',
                                         'Wide\n(25-50)', 'Very Wide\n(>50)'])
    df['time_bucket'] = pd.cut(df['mins_remaining'],
                               bins=[0, 5, 12, 24, 36, 48],
                               labels=['Crunch\n(0-5m)', '4th Qtr\n(5-12m)',
                                       '2nd Half', '1st Half', 'Early'])
    df['edge_bucket'] = pd.cut(df['entry_edge'],
                               bins=[0, 2, 5, 8, 100],
                               labels=['Thin\n(0-2¢)', 'Med\n(2-5¢)',
                                       'Fat\n(5-8¢)', 'Huge\n(>8¢)'])

    # ── Calibration ───────────────────────────────────────────────────────
    bins   = list(range(50, 105, 5))   # 50-55, 55-60, ... 95-100
    labels = [f"{b}-{b+5}%" for b in bins[:-1]]
    df['prob_bin'] = pd.cut(df['bet_prob'], bins=bins, labels=labels, include_lowest=True)

    calib = df.groupby('prob_bin', observed=True).agg(
        n     =('win', 'count'),
        wins  =('win', 'sum'),
        model_prob=('bet_prob', 'mean')
    ).reset_index()
    calib['actual_rate'] = calib['wins'] / calib['n']
    calib['expected_rate'] = calib['model_prob'] / 100
    calib['ci_lo'] = calib.apply(lambda r: _wilson_ci(r['wins'], r['n'])[0], axis=1)
    calib['ci_hi'] = calib.apply(lambda r: _wilson_ci(r['wins'], r['n'])[1], axis=1)

    print(f"\n── Calibration (model prob vs actual win rate) ─────────────")
    print(f"  {'Bucket':<10} {'N':>5} {'Model%':>8} {'Actual%':>8} {'Diff':>8}")
    for _, r in calib.iterrows():
        diff = r['actual_rate'] - r['expected_rate']
        flag = " ←" if abs(diff) > 0.07 and r['n'] >= 20 else ""
        print(f"  {str(r['prob_bin']):<10} {r['n']:>5} "
              f"{r['expected_rate']:>8.1%} {r['actual_rate']:>8.1%} "
              f"{diff:>+8.1%}{flag}")

    # ── Daily P&L ─────────────────────────────────────────────────────────
    daily = df.groupby('date').agg(
        realized=('realized_pnl', 'sum'),
        expected=('expected_pnl', 'sum'),
        n=('trade_id', 'count')
    ).reset_index()

    print(f"\n── Daily P&L ───────────────────────────────────────────────")
    for _, r in daily.iterrows():
        bar = '█' * int(abs(r['realized']) / 0.5)
        sign = '+' if r['realized'] >= 0 else '-'
        print(f"  {r['date']}  {_fmt_pnl(r['realized']):>8}  exp={_fmt_pnl(r['expected']):>8}  n={r['n']:>4}  {sign}{bar}")

    # ── Spread / Time / Edge breakdowns ──────────────────────────────────
    def bucket_summary(col, label):
        g = df.groupby(col, observed=True)['realized_pnl'].agg(['sum','count','mean']).reset_index()
        print(f"\n── P&L by {label} ──────────────────────────────────────────")
        for _, r in g.iterrows():
            print(f"  {str(r[col]):<18}  total={_fmt_pnl(r['sum']):>8}  n={r['count']:>4}  avg={_fmt_pnl(r['mean']):>8}")
        return g

    spread_pnl = bucket_summary('spread_bucket', 'Spread Width')
    time_pnl   = bucket_summary('time_bucket',   'Game Phase')
    edge_pnl   = bucket_summary('edge_bucket',   'Entry Edge')

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"P&L Analysis  |  {df['date'].min()} → {df['date'].max()}"
        f"  |  {n_trades:,} trades  |  {n_g} games  |  Z={z:+.2f}  p={p_val:.3f}",
        fontsize=14, fontweight='bold'
    )

    palette = sns.diverging_palette(10, 145, s=80, l=55, as_cmap=False, n=2)

    # 1. Daily P&L
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in daily['realized']]
    ax1.bar(range(len(daily)), daily['realized'], color=colors, label='Realized')
    ax1.plot(range(len(daily)), daily['expected'], 'b--o', markersize=4, linewidth=1.5, label='Expected')
    ax1.set_xticks(range(len(daily)))
    ax1.set_xticklabels([str(d) for d in daily['date']], rotation=45, ha='right', fontsize=8)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Daily P&L: Realized vs Expected')
    ax1.set_ylabel('P&L ($)')
    ax1.legend()

    # 2. Cumulative expected vs actual
    ax2 = fig.add_subplot(gs[1, :])
    df_sorted = df.sort_values('created_at')
    cum_actual   = df_sorted['realized_pnl'].cumsum().values
    cum_expected = df_sorted['expected_pnl'].cumsum().values
    ax2.plot(cum_actual,   color='#e74c3c', label='Cumulative Actual P&L',   linewidth=1.5)
    ax2.plot(cum_expected, color='#3498db', label='Cumulative Expected P&L', linewidth=1.5, linestyle='--')
    ax2.fill_between(range(len(cum_actual)), cum_actual, cum_expected,
                     alpha=0.15, color='orange', label='Divergence')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Cumulative P&L: Actual vs Expected  (divergence = luck ± model error)')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.set_xlabel('Trade #')
    ax2.legend()

    # 3. Calibration
    ax3 = fig.add_subplot(gs[2, 0])
    valid = calib[calib['n'] >= 10]
    ax3.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, label='Perfect calibration')
    ax3.errorbar(valid['expected_rate'], valid['actual_rate'],
                 yerr=[valid['actual_rate'] - valid['ci_lo'],
                       valid['ci_hi']  - valid['actual_rate']],
                 fmt='o', color='#8e44ad', capsize=4, markersize=8, label='Model')
    for _, r in valid.iterrows():
        ax3.annotate(f"n={r['n']}", (r['expected_rate'], r['actual_rate']),
                     textcoords='offset points', xytext=(6, 3), fontsize=7)
    ax3.set_xlim(0.48, 1.02); ax3.set_ylim(0, 1.05)
    ax3.set_xlabel('Model probability'); ax3.set_ylabel('Actual win rate')
    ax3.set_title('Calibration  (dots should lie on dashed line)')
    ax3.legend()

    # 4. Game-level P&L distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(game_pnl.values, bins=25, color='#2980b9', edgecolor='white', alpha=0.8)
    ax4.axvline(0, color='black', linewidth=1)
    ax4.axvline(game_pnl.mean(), color='#e74c3c', linewidth=2, linestyle='--',
                label=f'Mean = {_fmt_pnl(game_pnl.mean())}')
    ax4.set_xlabel('Realized P&L per game ($)')
    ax4.set_ylabel('# Games')
    ax4.set_title(f'Game-level P&L Distribution  (n={n_g} games)\n'
                  f'Avg trades/game={n_trades/max(n_g,1):.1f} — effective indep. obs = {n_g}')
    ax4.legend()

    # 5. P&L by spread / time / edge
    for ax, data, col, title in [
        (fig.add_subplot(gs[3, 0]), spread_pnl, 'spread_bucket', 'P&L by Spread Width'),
        (fig.add_subplot(gs[3, 1]), time_pnl,   'time_bucket',   'P&L by Game Phase'),
    ]:
        colors2 = ['#27ae60' if v >= 0 else '#e74c3c' for v in data['sum']]
        ax.bar(range(len(data)), data['sum'], color=colors2)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data[col].astype(str), fontsize=8)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(title); ax.set_ylabel('Total P&L ($)')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved → {output_path}")


def main():
    args = parse_args()
    print(f"Loading trades from {args.start or 'all time'} …")
    df = load_trades(args.start, args.end, args.min_edge)
    if df.empty:
        print("No trades found."); return
    analyze(df, args.output)


if __name__ == "__main__":
    main()
