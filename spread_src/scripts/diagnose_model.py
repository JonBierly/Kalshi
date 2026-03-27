#!/usr/bin/env python
"""
Model Diagnostic Report

Three investigations:
  1. Direction bug check — sells where model > fill_price (wrong direction)
  2. Overfitting check  — Brier score pre vs post training cutoff (game-level)
  3. Feature distribution shift — are post-training inputs out-of-distribution?

Usage:
    python -m spread_src.scripts.diagnose_model --start 2026-02-09
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os

DB_PATH = 'data/nba_data.db'
TRAINING_CUTOFF = '2026-02-09'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=str, default=TRAINING_CUTOFF)
    p.add_argument('--output', type=str, default='reports/model_diagnosis.png')
    return p.parse_args()


def load_trades(conn):
    df = pd.read_sql_query("""
        SELECT trade_id, ticker, game_id, side, fill_price, model_fair_value,
               model_ci_lower, model_ci_upper, realized_pnl, size,
               seconds_remaining, created_at
        FROM trades
        WHERE realized_pnl IS NOT NULL AND size > 0
    """, conn)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['period'] = np.where(df['created_at'] < TRAINING_CUTOFF, 'pre_training', 'post_training')

    # Settle direction
    df['settled_yes'] = (
        ((df['side'] == 'buy')  & (df['realized_pnl'] > 0)) |
        ((df['side'] == 'sell') & (df['realized_pnl'] < 0))
    ).astype(int)

    # Expected edge
    df['edge'] = np.where(df['side'] == 'buy',
                          df['model_fair_value'] - df['fill_price'],
                          df['fill_price'] - df['model_fair_value'])
    df['expected_pnl'] = df['edge'] * df['size'] / 100.0

    # Direction correct flag
    df['direction_ok'] = (df['edge'] > 0)

    # Game phase
    bins = [-1, 240, 720, 1440, 2160, 99999]
    labels = ['Q4 late\n(<4min)', 'Q4 early\n(4-12min)', 'Q3\n(12-24min)',
              'Q2\n(24-36min)', 'Q1\n(>36min)']
    df['phase'] = pd.cut(df['seconds_remaining'], bins=bins, labels=labels)

    return df


# ── 1. Direction Bug ──────────────────────────────────────────────────────────

def direction_bug_report(df):
    post = df[df['period'] == 'post_training']

    bad_sells = post[(post['side'] == 'sell') & (post['model_fair_value'] > post['fill_price'])]
    bad_buys  = post[(post['side'] == 'buy')  & (post['model_fair_value'] < post['fill_price'])]

    print(f"\n{'='*65}")
    print("  INVESTIGATION 1: DIRECTION BUG CHECK")
    print(f"{'='*65}")
    print(f"\n  Post-training trades: {len(post):,}")
    print(f"\n  SELL trades where model > fill (wrong direction):")
    print(f"    Count:      {len(bad_sells):,}  ({100*len(bad_sells)/len(post[post['side']=='sell']):.1f}% of sells)")
    if len(bad_sells):
        print(f"    Avg model:  {bad_sells['model_fair_value'].mean():.1f}¢")
        print(f"    Avg fill:   {bad_sells['fill_price'].mean():.1f}¢")
        print(f"    Avg edge:   {bad_sells['edge'].mean():.1f}¢  (should be POSITIVE for buys)")
        print(f"    Win rate:   {(bad_sells['realized_pnl'] > 0).mean():.1%}")
        print(f"    Total PnL:  ${bad_sells['realized_pnl'].sum():.2f}")

    print(f"\n  BUY trades where model < fill (wrong direction):")
    print(f"    Count:      {len(bad_buys):,}  ({100*len(bad_buys)/max(len(post[post['side']=='buy']),1):.1f}% of buys)")
    if len(bad_buys):
        print(f"    Total PnL:  ${bad_buys['realized_pnl'].sum():.2f}")

    total_wrong_pnl = bad_sells['realized_pnl'].sum() + bad_buys['realized_pnl'].sum()
    print(f"\n  Combined PnL from wrong-direction trades: ${total_wrong_pnl:.2f}")

    # Are these wrong-direction trades clustered by ticker or time?
    if len(bad_sells) > 0:
        top_tickers = bad_sells.groupby('ticker')['realized_pnl'].agg(['count','sum']).sort_values('count', ascending=False).head(10)
        print(f"\n  Top tickers with wrong-direction sells:")
        for tk, row in top_tickers.iterrows():
            print(f"    {tk:<45} n={int(row['count']):>4}  pnl=${row['sum']:>+7.2f}")

    return bad_sells, bad_buys


# ── 2. Overfitting ────────────────────────────────────────────────────────────

def overfitting_report(df):
    print(f"\n{'='*65}")
    print("  INVESTIGATION 2: OVERFITTING / TRAIN-TEST SPLIT")
    print(f"{'='*65}")

    # Collapse to one row per ticker, per period
    def game_ticker_summary(group):
        settled = int(round(group['settled_yes'].mean()))
        p_model = group['model_fair_value'].mean() / 100.0
        return pd.Series({
            'p_model': p_model,
            'settled_yes': settled,
            'n_trades': len(group),
            'brier': (p_model - settled) ** 2,
        })

    for period in ['pre_training', 'post_training']:
        sub = df[df['period'] == period]
        gt = sub.groupby(['ticker', 'game_id']).apply(game_ticker_summary).reset_index()
        n = len(gt)
        brier = gt['brier'].mean()
        avg_model = gt['p_model'].mean()
        avg_actual = gt['settled_yes'].mean()
        bias = avg_model - avg_actual  # positive = overestimates YES
        label = "PRE-TRAINING (in-sample)" if period == 'pre_training' else "POST-TRAINING (out-of-sample)"
        print(f"\n  {label}")
        print(f"    Independent (game×ticker) observations: {n:,}")
        print(f"    Brier score:   {brier:.4f}")
        print(f"    Avg model P:   {avg_model:.3f}  ({avg_model*100:.1f}¢)")
        print(f"    Avg actual P:  {avg_actual:.3f}  ({avg_actual*100:.1f}¢)")
        direction = 'OVER-estimates YES' if bias > 0 else 'UNDER-estimates YES'
        print(f"    Avg bias:      {bias:+.3f}  → model {direction} by {abs(bias)*100:.1f}¢")

    # PnL by game phase (post-training only)
    post = df[df['period'] == 'post_training'].copy()
    phase_summary = post.groupby('phase', observed=True).agg(
        n=('realized_pnl', 'count'),
        total_pnl=('realized_pnl', 'sum'),
        avg_pnl=('realized_pnl', 'mean'),
        win_rate=('realized_pnl', lambda x: (x > 0).mean()),
        avg_expected_pnl=('expected_pnl', 'mean'),
        avg_confidence=('model_fair_value', lambda x: (x - 50).abs().mean()),
    ).reset_index()

    print(f"\n  PnL by game phase (post-training):")
    print(f"  {'Phase':<22} {'N':>5} {'AvgPnL':>8} {'TotalPnL':>10} {'WinRate':>8} "
          f"{'AvgExpPnL':>10} {'AvgConf':>8}")
    for _, r in phase_summary.iterrows():
        flag = " ← LOSING PHASE" if r['total_pnl'] < -30 else ""
        print(f"  {str(r['phase']):<22} {r['n']:>5} ${r['avg_pnl']:>+7.3f} "
              f"${r['total_pnl']:>+9.2f} {r['win_rate']:>8.1%} "
              f"${r['avg_expected_pnl']:>+9.3f} {r['avg_confidence']:>8.1f}¢{flag}")

    return phase_summary


# ── 3. Feature Distribution Shift ────────────────────────────────────────────

def distribution_shift_report(df):
    print(f"\n{'='*65}")
    print("  INVESTIGATION 3: FEATURE DISTRIBUTION SHIFT")
    print(f"{'='*65}")

    pre  = df[df['period'] == 'pre_training']
    post = df[df['period'] == 'post_training']

    features = {
        'seconds_remaining':   'Seconds remaining in game',
        'fill_price':          'Market fill price (¢)',
        'model_fair_value':    'Model fair value (¢)',
        'edge':                'Perceived edge (¢)',
    }

    print(f"\n  {'Feature':<32} {'Pre mean':>10} {'Post mean':>10} {'Δ':>8} {'Δ/pre_std':>10}")
    shifts = {}
    for col, label in features.items():
        if col not in df.columns:
            continue
        pre_mean  = pre[col].mean()
        post_mean = post[col].mean()
        pre_std   = pre[col].std()
        delta     = post_mean - pre_mean
        z         = delta / pre_std if pre_std > 0 else 0
        flag = " ←" if abs(z) > 0.3 else ""
        print(f"  {label:<32} {pre_mean:>10.2f} {post_mean:>10.2f} {delta:>+8.2f} {z:>+10.2f}{flag}")
        shifts[col] = {'pre': pre_mean, 'post': post_mean, 'delta': delta, 'z': z}

    # Buy/sell mix shift
    print(f"\n  Side mix:")
    for period, sub in [('pre', pre), ('post', post)]:
        buy_pct  = (sub['side'] == 'buy').mean()
        sell_pct = (sub['side'] == 'sell').mean()
        print(f"    {period}: buy={buy_pct:.1%}  sell={sell_pct:.1%}")

    # CI quality
    print(f"\n  CI availability (model_ci_lower/upper not null):")
    for period, sub in [('pre', pre), ('post', post)]:
        has_ci = sub['model_ci_lower'].notna().mean()
        print(f"    {period}: {has_ci:.1%} of trades have CIs")

    return shifts


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df, bad_sells, phase_summary, output_path):
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    pre  = df[df['period'] == 'pre_training']
    post = df[df['period'] == 'post_training']

    # ── Row 1: Direction bug ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Correct sells\n(model < fill)', 'Wrong sells\n(model > fill)', 'Buys']
    correct_sells = post[(post['side'] == 'sell') & (post['model_fair_value'] <= post['fill_price'])]
    buys = post[post['side'] == 'buy']
    pnls = [correct_sells['realized_pnl'].sum(), bad_sells['realized_pnl'].sum(), buys['realized_pnl'].sum()]
    colors = ['#27ae60' if p >= 0 else '#e74c3c' for p in pnls]
    ax1.bar(categories, pnls, color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Total PnL by Trade Type\n(post-training)', fontweight='bold')
    ax1.set_ylabel('Total PnL ($)')
    for i, (cat, val) in enumerate(zip(categories, pnls)):
        ax1.text(i, val + (5 if val >= 0 else -15), f'${val:+.0f}', ha='center', fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    if len(bad_sells) > 0:
        ax2.hist(bad_sells['model_fair_value'] - bad_sells['fill_price'], bins=30,
                 color='#e74c3c', alpha=0.7, edgecolor='white')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Model − Fill (¢)  [positive = wrong direction]')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Wrong-Direction Sells\n({len(bad_sells)} trades)', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(bad_sells['fill_price'], bad_sells['model_fair_value'],
                alpha=0.4, c='#e74c3c', s=20)
    ax3.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='model = fill')
    ax3.set_xlabel('Fill price (¢)'); ax3.set_ylabel('Model fair value (¢)')
    ax3.set_title('Wrong Sells: Model vs Fill Price', fontweight='bold')
    ax3.legend(fontsize=8)

    # ── Row 2: Overfitting ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    # Calibration scatter: pre vs post, bucketed
    for period, sub, color, label in [('pre', pre, '#3498db', 'Pre-training'),
                                        ('post', post, '#e74c3c', 'Post-training')]:
        bins = np.linspace(0, 100, 11)
        bucket_model, bucket_actual = [], []
        mfv = sub['model_fair_value'].values
        sy  = sub['settled_yes'].values
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (mfv >= lo) & (mfv < hi)
            if mask.sum() >= 10:
                bucket_model.append(mfv[mask].mean() / 100)
                bucket_actual.append(sy[mask].mean())
        ax4.scatter(bucket_model, bucket_actual, color=color, label=label, s=60, alpha=0.8)
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Model P(YES)'); ax4.set_ylabel('Actual YES rate')
    ax4.set_title('Calibration: Pre vs Post Training', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    ax5 = fig.add_subplot(gs[1, 1])
    phase_order = phase_summary['phase'].tolist()
    colors5 = ['#27ae60' if v >= 0 else '#e74c3c' for v in phase_summary['total_pnl']]
    ax5.bar(range(len(phase_summary)), phase_summary['total_pnl'], color=colors5, alpha=0.8)
    ax5.set_xticks(range(len(phase_summary)))
    ax5.set_xticklabels([str(p) for p in phase_order], fontsize=8)
    ax5.axhline(0, color='black', linewidth=1)
    ax5.set_title('Total PnL by Game Phase\n(post-training)', fontweight='bold')
    ax5.set_ylabel('Total PnL ($)')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(range(len(phase_summary)), phase_summary['avg_expected_pnl'],
            color='#8e44ad', alpha=0.5, label='Expected')
    ax6.bar(range(len(phase_summary)), phase_summary['avg_pnl'],
            color='#e74c3c', alpha=0.7, label='Actual')
    ax6.set_xticks(range(len(phase_summary)))
    ax6.set_xticklabels([str(p) for p in phase_order], fontsize=8)
    ax6.axhline(0, color='black', linewidth=1)
    ax6.set_title('Expected vs Actual PnL\nby Game Phase', fontweight='bold')
    ax6.set_ylabel('Avg PnL per trade ($)')
    ax6.legend(fontsize=8)

    # ── Row 3: Distribution shift ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(pre['model_fair_value'], bins=50, color='#3498db', alpha=0.5,
             density=True, label='Pre-training')
    ax7.hist(post['model_fair_value'], bins=50, color='#e74c3c', alpha=0.5,
             density=True, label='Post-training')
    ax7.set_xlabel('Model fair value (¢)')
    ax7.set_title('Model Fair Value Distribution\nShift', fontweight='bold')
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(pre['seconds_remaining'].dropna(), bins=50, color='#3498db', alpha=0.5,
             density=True, label='Pre-training')
    ax8.hist(post['seconds_remaining'].dropna(), bins=50, color='#e74c3c', alpha=0.5,
             density=True, label='Post-training')
    ax8.set_xlabel('Seconds remaining')
    ax8.set_title('Trade Timing Distribution\nShift', fontweight='bold')
    ax8.legend(fontsize=8)

    ax9 = fig.add_subplot(gs[2, 2])
    # Cumulative PnL post-training
    post_sorted = post.sort_values('created_at')
    ax9.plot(np.arange(len(post_sorted)), post_sorted['realized_pnl'].cumsum(),
             color='#e74c3c', linewidth=1.5, label='Actual')
    ax9.plot(np.arange(len(post_sorted)), post_sorted['expected_pnl'].cumsum(),
             color='#8e44ad', linewidth=1.5, linestyle='--', label='Expected')
    ax9.axhline(0, color='black', linewidth=0.8)
    ax9.set_xlabel('Trade #')
    ax9.set_ylabel('Cumulative PnL ($)')
    ax9.set_title('Cumulative PnL: Actual vs Expected\n(post-training)', fontweight='bold')
    ax9.legend(fontsize=8)

    fig.suptitle(
        f'Model Diagnosis Report  |  Training cutoff: {TRAINING_CUTOFF}  |  '
        f'Pre: {len(pre):,} trades  Post: {len(post):,} trades',
        fontsize=13, fontweight='bold'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    conn = sqlite3.connect(DB_PATH)
    df = load_trades(conn)
    conn.close()

    print(f"\nLoaded {len(df):,} trades  "
          f"(pre: {(df['period']=='pre_training').sum():,}  "
          f"post: {(df['period']=='post_training').sum():,})")

    bad_sells, bad_buys = direction_bug_report(df)
    phase_summary       = overfitting_report(df)
    _                   = distribution_shift_report(df)
    make_plots(df, bad_sells, phase_summary, args.output)


if __name__ == '__main__':
    main()
