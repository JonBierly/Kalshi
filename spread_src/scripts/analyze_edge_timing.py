#!/usr/bin/env python
"""
Edge Timing Analysis — When and where does the model beat the market?

Uses the model_predictions table (settled rows) and trades table to answer:
1. MODEL vs MARKET Brier scores sliced by game phase, spread level, score margin
2. CONDITIONAL P&L — when is the bot profitable vs unprofitable?
3. EDGE DECAY — does the model's advantage grow or shrink as the game progresses?
4. CALIBRATION — is the model's probability well-calibrated?
5. SWEET SPOT MAP — a heatmap of model edge by (time × spread level)

Usage:
    python -m spread_src.scripts.analyze_edge_timing [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import os
import sys
import sqlite3
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

DB_PATH = 'data/nba_data.db'

# Plotting style
sns.set_style("darkgrid")
plt.rcParams.update({
    'figure.figsize': (18, 12),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

REPORT_DIR = 'reports/edge_timing'


# ── Data Loading ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze when and where the model beats the market.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def extract_spread_value(ticker: str) -> float:
    """Extract spread value from ticker (e.g., KXNBASPREAD-...-OKC5 → 5.5)."""
    match = re.search(r'-([A-Z]{3})(\d+)$', ticker)
    if match:
        return float(match.group(2)) + 0.5
    return None


def load_predictions(start_date=None, end_date=None) -> pd.DataFrame:
    """Load settled model predictions."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            prediction_id, timestamp, game_id, ticker,
            seconds_remaining, actual_score_diff,
            predicted_prob, bid_price, ask_price,
            actual_outcome
        FROM model_predictions
        WHERE actual_outcome IS NOT NULL
          AND predicted_prob IS NOT NULL
          AND bid_price IS NOT NULL
          AND ask_price IS NOT NULL
    """
    
    conditions = []
    params = []
    
    if start_date:
        conditions.append("DATE(timestamp) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(timestamp) <= ?")
        params.append(end_date)
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return df
    
    # Derived columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['market_mid'] = (df['bid_price'] + df['ask_price']) / 2.0 / 100.0  # Convert cents → prob
    df['market_spread_width'] = (df['ask_price'] - df['bid_price'])  # In cents
    df['model_prob'] = df['predicted_prob'].clip(0.01, 0.99)
    df['outcome'] = df['actual_outcome'].astype(int)
    
    # Spread level from ticker
    df['spread_value'] = df['ticker'].apply(extract_spread_value)
    
    # Game phase
    df['mins_remaining'] = df['seconds_remaining'] / 60.0
    df['quarter'] = pd.cut(
        df['seconds_remaining'],
        bins=[0, 720, 1440, 2160, 2880],
        labels=['Q4', 'Q3', 'Q2', 'Q1'],
        include_lowest=True
    )
    df['game_phase'] = pd.cut(
        df['mins_remaining'],
        bins=[0, 6, 12, 24, 36, 48],
        labels=['Crunch (<6m)', 'Late (6-12m)', 'Mid (12-24m)', 'Early-Mid (24-36m)', 'Early (36-48m)'],
        include_lowest=True
    )
    
    # Spread category
    df['spread_category'] = pd.cut(
        df['spread_value'],
        bins=[0, 3.5, 8.5, 14.5, 100],
        labels=['Tight (≤3.5)', 'Mid (4.5-8.5)', 'Wide (9.5-14.5)', 'Extreme (≥15.5)'],
        include_lowest=True
    )
    
    # Score margin category
    df['score_margin'] = pd.cut(
        df['actual_score_diff'].abs(),
        bins=[-1, 5, 10, 20, 100],
        labels=['Tight (0-5)', 'Close (6-10)', 'Comfortable (11-20)', 'Blowout (>20)'],
        include_lowest=True
    )
    
    # Brier scores
    df['model_brier'] = (df['model_prob'] - df['outcome']) ** 2
    df['market_brier'] = (df['market_mid'] - df['outcome']) ** 2
    df['brier_advantage'] = df['market_brier'] - df['model_brier']  # Positive = model wins
    
    # Edge at entry (model vs market)
    df['model_edge'] = df['model_prob'] - df['market_mid']
    
    return df


def load_trades(start_date=None, end_date=None) -> pd.DataFrame:
    """Load closed trades with P&L."""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT trade_id, timestamp, ticker, game_id, side,
               fill_price, size, model_fair_value,
               market_spread, seconds_remaining,
               realized_pnl, status, created_at
        FROM trades
        WHERE realized_pnl IS NOT NULL AND status = 'closed'
    """
    
    conditions = []
    params = []
    if start_date:
        conditions.append("DATE(created_at) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(created_at) <= ?")
        params.append(end_date)
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return df
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['spread_value'] = df['ticker'].apply(extract_spread_value)
    df['mins_remaining'] = df['seconds_remaining'] / 60.0
    df['entry_edge'] = abs(df['fill_price'] - df['model_fair_value'])
    
    df['game_phase'] = pd.cut(
        df['mins_remaining'],
        bins=[0, 6, 12, 24, 36, 48],
        labels=['Crunch (<6m)', 'Late (6-12m)', 'Mid (12-24m)', 'Early-Mid (24-36m)', 'Early (36-48m)'],
        include_lowest=True
    )
    
    df['spread_category'] = pd.cut(
        df['spread_value'],
        bins=[0, 3.5, 8.5, 14.5, 100],
        labels=['Tight (≤3.5)', 'Mid (4.5-8.5)', 'Wide (9.5-14.5)', 'Extreme (≥15.5)'],
        include_lowest=True
    )
    
    return df


# ── Analysis Functions ───────────────────────────────────────────────────────

def analyze_overall_brier(df: pd.DataFrame):
    """Overall model vs market Brier score."""
    print("\n" + "=" * 90)
    print("  1. OVERALL: MODEL vs MARKET BRIER SCORE")
    print("     (Lower is better. Advantage > 0 means model beats market)")
    print("=" * 90)
    
    model_brier = df['model_brier'].mean()
    market_brier = df['market_brier'].mean()
    advantage = market_brier - model_brier
    
    n = len(df)
    # Bootstrap standard error for the advantage
    np.random.seed(42)
    boot_advantages = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        boot_adv = df['market_brier'].iloc[idx].mean() - df['model_brier'].iloc[idx].mean()
        boot_advantages.append(boot_adv)
    se = np.std(boot_advantages)
    
    pct_model_wins = (df['brier_advantage'] > 0).mean() * 100
    
    print(f"\n  Predictions analyzed: {n:,}")
    print(f"  Model Brier:  {model_brier:.6f}")
    print(f"  Market Brier: {market_brier:.6f}")
    print(f"  Advantage:    {advantage:+.6f} ± {se:.6f} (SE)")
    print(f"  Model wins:   {pct_model_wins:.1f}% of individual predictions")
    
    if advantage > 2 * se:
        print(f"  ✅ Model SIGNIFICANTLY beats the market (>2 SE)")
    elif advantage > 0:
        print(f"  🟡 Model slightly better but NOT statistically significant")
    else:
        print(f"  ❌ Market beats the model")
    
    return {'model_brier': model_brier, 'market_brier': market_brier, 'advantage': advantage, 'se': se}


def analyze_brier_by_dimension(df: pd.DataFrame, col: str, label: str):
    """Brier score breakdown by any categorical column."""
    print(f"\n{'─' * 90}")
    print(f"  {label}")
    print(f"{'─' * 90}")
    
    results = []
    
    print(f"\n  {'Category':<22} {'N':>7} {'Model Brier':>12} {'Market Brier':>13} "
          f"{'Advantage':>10} {'Model Wins%':>12} {'Verdict':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*12} {'-'*13} {'-'*10} {'-'*12} {'-'*10}")
    
    for cat in df[col].cat.categories if hasattr(df[col], 'cat') else sorted(df[col].unique()):
        subset = df[df[col] == cat]
        if len(subset) < 10:
            continue
        
        mb = subset['model_brier'].mean()
        mkb = subset['market_brier'].mean()
        adv = mkb - mb
        pct = (subset['brier_advantage'] > 0).mean() * 100
        verdict = "✅" if adv > 0.001 else ("🟡" if adv > 0 else "❌")
        
        results.append({'category': str(cat), 'n': len(subset), 'model_brier': mb, 
                       'market_brier': mkb, 'advantage': adv, 'pct_model_wins': pct})
        
        print(f"  {str(cat):<22} {len(subset):>7,} {mb:>12.6f} {mkb:>13.6f} "
              f"{adv:>+10.6f} {pct:>11.1f}% {verdict:>10}")
    
    return results


def analyze_calibration(df: pd.DataFrame, output_dir: str):
    """Model calibration: when model says 60%, does it happen ~60% of the time?"""
    print(f"\n{'─' * 90}")
    print(f"  CALIBRATION (When model says X%, how often does it happen?)")
    print(f"{'─' * 90}")
    
    df_cal = df.copy()
    df_cal['prob_bucket'] = pd.cut(
        df_cal['model_prob'],
        bins=np.arange(0, 1.05, 0.1),
        labels=[f"{int(i*100)}-{int((i+0.1)*100)}%" for i in np.arange(0, 1.0, 0.1)]
    )
    
    cal_data = df_cal.groupby('prob_bucket', observed=True).agg(
        n=('outcome', 'count'),
        actual_rate=('outcome', 'mean'),
        predicted_mean=('model_prob', 'mean'),
        market_mean=('market_mid', 'mean')
    ).dropna()
    
    print(f"\n  {'Predicted Range':<16} {'N':>7} {'Actual Hit%':>12} {'Model Mean%':>12} "
          f"{'Market Mean%':>13} {'Model Error':>12} {'Market Error':>13}")
    print(f"  {'-'*16} {'-'*7} {'-'*12} {'-'*12} {'-'*13} {'-'*12} {'-'*13}")
    
    for idx, row in cal_data.iterrows():
        model_err = abs(row['predicted_mean'] - row['actual_rate'])
        market_err = abs(row['market_mean'] - row['actual_rate'])
        better = "✅" if model_err < market_err else "❌"
        
        print(f"  {str(idx):<16} {int(row['n']):>7,} {row['actual_rate']*100:>11.1f}% "
              f"{row['predicted_mean']*100:>11.1f}% {row['market_mean']*100:>12.1f}% "
              f"{model_err:>11.4f} {market_err:>12.4f} {better}")
    
    # Calibration plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    perfect = np.linspace(0, 1, 100)
    ax.plot(perfect, perfect, 'k--', alpha=0.5, label='Perfect calibration')
    
    ax.scatter(cal_data['predicted_mean'], cal_data['actual_rate'], 
               s=cal_data['n']/5, c='#2ecc71', alpha=0.8, label='Model', zorder=5)
    ax.scatter(cal_data['market_mean'], cal_data['actual_rate'],
               s=cal_data['n']/5, c='#e74c3c', alpha=0.6, label='Market', zorder=4)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Outcome Rate')
    ax.set_title('Calibration: Model vs Market')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration.png', dpi=150)
    plt.close()
    print(f"\n  📊 Saved calibration plot to {output_dir}/calibration.png")


def analyze_edge_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: model Brier advantage by (game phase × spread level)."""
    
    print(f"\n{'─' * 90}")
    print(f"  SWEET SPOT MAP: Model advantage by Game Phase × Spread Level")
    print(f"{'─' * 90}")
    
    valid = df.dropna(subset=['game_phase', 'spread_category'])
    
    # Pivot table: advantage
    pivot_adv = valid.pivot_table(
        values='brier_advantage', 
        index='game_phase', 
        columns='spread_category',
        aggfunc='mean'
    )
    
    pivot_n = valid.pivot_table(
        values='brier_advantage',
        index='game_phase',
        columns='spread_category',
        aggfunc='count'
    )
    
    # Print text version
    print(f"\n  {'':>22}", end='')
    for col in pivot_adv.columns:
        print(f" {str(col):>18}", end='')
    print()
    print(f"  {'':>22}", end='')
    for col in pivot_adv.columns:
        print(f" {'-'*18}", end='')
    print()
    
    for phase in pivot_adv.index:
        print(f"  {str(phase):<22}", end='')
        for col in pivot_adv.columns:
            val = pivot_adv.loc[phase, col] if col in pivot_adv.columns and not pd.isna(pivot_adv.loc[phase, col]) else None
            n = pivot_n.loc[phase, col] if col in pivot_n.columns and not pd.isna(pivot_n.loc[phase, col]) else 0
            if val is not None:
                marker = "✅" if val > 0.001 else ("🟡" if val > 0 else "❌")
                print(f" {val:>+.5f} (n={int(n):>4}) {marker}", end='')
            else:
                print(f" {'N/A':>18}", end='')
        print()
    
    # Heatmap plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Advantage heatmap
    sns.heatmap(pivot_adv, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                ax=axes[0], cbar_kws={'label': 'Brier Advantage (+ = model wins)'})
    axes[0].set_title('Model Brier Advantage\n(Green = model beats market)')
    axes[0].set_ylabel('Game Phase')
    axes[0].set_xlabel('Spread Level')
    
    # Sample size heatmap
    sns.heatmap(pivot_n, annot=True, fmt='.0f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'Sample Count'})
    axes[1].set_title('Sample Size\n(How much data supports each cell)')
    axes[1].set_ylabel('Game Phase')
    axes[1].set_xlabel('Spread Level')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edge_heatmap.png', dpi=150)
    plt.close()
    print(f"\n  📊 Saved edge heatmap to {output_dir}/edge_heatmap.png")


def analyze_pnl_conditional(trades_df: pd.DataFrame, output_dir: str):
    """P&L breakdown by game phase and spread level."""
    
    print(f"\n{'─' * 90}")
    print(f"  CONDITIONAL P&L: Where is the bot making/losing money?")
    print(f"{'─' * 90}")
    
    if trades_df.empty:
        print("  No closed trades found.")
        return
    
    # By game phase
    print(f"\n  BY GAME PHASE:")
    print(f"  {'Phase':<22} {'N Trades':>9} {'Total P&L':>10} {'Avg P&L':>9} {'Win Rate':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")
    
    for phase in ['Early (36-48m)', 'Early-Mid (24-36m)', 'Mid (12-24m)', 'Late (6-12m)', 'Crunch (<6m)']:
        subset = trades_df[trades_df['game_phase'] == phase]
        if subset.empty:
            continue
        total = subset['realized_pnl'].sum()
        avg = subset['realized_pnl'].mean()
        wr = (subset['realized_pnl'] > 0).mean() * 100
        marker = "✅" if total > 0 else "❌"
        print(f"  {phase:<22} {len(subset):>9,} ${total:>8.2f} ${avg:>7.2f} {wr:>7.1f}% {marker}")
    
    # By spread level
    print(f"\n  BY SPREAD LEVEL:")
    print(f"  {'Spread':<22} {'N Trades':>9} {'Total P&L':>10} {'Avg P&L':>9} {'Win Rate':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")
    
    for cat in ['Tight (≤3.5)', 'Mid (4.5-8.5)', 'Wide (9.5-14.5)', 'Extreme (≥15.5)']:
        subset = trades_df[trades_df['spread_category'] == cat]
        if subset.empty:
            continue
        total = subset['realized_pnl'].sum()
        avg = subset['realized_pnl'].mean()
        wr = (subset['realized_pnl'] > 0).mean() * 100
        marker = "✅" if total > 0 else "❌"
        print(f"  {cat:<22} {len(subset):>9,} ${total:>8.2f} ${avg:>7.2f} {wr:>7.1f}% {marker}")
    
    # By entry edge size
    trades_df['edge_bucket'] = pd.cut(
        trades_df['entry_edge'],
        bins=[0, 3, 6, 10, 20, 100],
        labels=['Tiny (0-3¢)', 'Small (3-6¢)', 'Medium (6-10¢)', 'Large (10-20¢)', 'Huge (>20¢)'],
        include_lowest=True
    )
    
    print(f"\n  BY ENTRY EDGE:")
    print(f"  {'Edge':<22} {'N Trades':>9} {'Total P&L':>10} {'Avg P&L':>9} {'Win Rate':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")
    
    for cat in ['Tiny (0-3¢)', 'Small (3-6¢)', 'Medium (6-10¢)', 'Large (10-20¢)', 'Huge (>20¢)']:
        subset = trades_df[trades_df['edge_bucket'] == cat]
        if subset.empty:
            continue
        total = subset['realized_pnl'].sum()
        avg = subset['realized_pnl'].mean()
        wr = (subset['realized_pnl'] > 0).mean() * 100
        marker = "✅" if total > 0 else "❌"
        print(f"  {cat:<22} {len(subset):>9,} ${total:>8.2f} ${avg:>7.2f} {wr:>7.1f}% {marker}")
    
    # Cumulative P&L chart
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Conditional P&L Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative P&L over time
    daily = trades_df.groupby('date')['realized_pnl'].sum().cumsum()
    axes[0, 0].plot(daily.index, daily.values, 'b-', linewidth=2)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Cumulative P&L Over Time')
    axes[0, 0].set_ylabel('Cumulative P&L ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. P&L by game phase
    phase_pnl = trades_df.groupby('game_phase', observed=True)['realized_pnl'].sum()
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in phase_pnl.values]
    phase_pnl.plot(kind='barh', ax=axes[0, 1], color=colors)
    axes[0, 1].set_title('Total P&L by Game Phase')
    axes[0, 1].set_xlabel('P&L ($)')
    axes[0, 1].axvline(0, color='black', linewidth=0.5)
    
    # 3. P&L by spread level
    spread_pnl = trades_df.groupby('spread_category', observed=True)['realized_pnl'].sum()
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in spread_pnl.values]
    spread_pnl.plot(kind='barh', ax=axes[1, 0], color=colors)
    axes[1, 0].set_title('Total P&L by Spread Level')
    axes[1, 0].set_xlabel('P&L ($)')
    axes[1, 0].axvline(0, color='black', linewidth=0.5)
    
    # 4. P&L by edge size
    edge_pnl = trades_df.groupby('edge_bucket', observed=True)['realized_pnl'].sum()
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in edge_pnl.values]
    edge_pnl.plot(kind='barh', ax=axes[1, 1], color=colors)
    axes[1, 1].set_title('Total P&L by Entry Edge')
    axes[1, 1].set_xlabel('P&L ($)')
    axes[1, 1].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/conditional_pnl.png', dpi=150)
    plt.close()
    print(f"\n  📊 Saved P&L breakdown to {output_dir}/conditional_pnl.png")


def analyze_edge_decay(df: pd.DataFrame, output_dir: str):
    """How does model advantage change as the game progresses?"""
    
    print(f"\n{'─' * 90}")
    print(f"  EDGE DECAY: Model advantage over game time")
    print(f"{'─' * 90}")
    
    # Bin by minutes remaining (finer bins)
    df_time = df.copy()
    df_time['time_bin'] = pd.cut(
        df_time['mins_remaining'],
        bins=np.arange(0, 50, 4),
        labels=[f"{int(i)}-{int(i+4)}m" for i in np.arange(0, 48, 4)]
    )
    
    time_stats = df_time.groupby('time_bin', observed=True).agg(
        n=('brier_advantage', 'count'),
        avg_advantage=('brier_advantage', 'mean'),
        model_brier=('model_brier', 'mean'),
        market_brier=('market_brier', 'mean'),
        avg_spread_width=('market_spread_width', 'mean')
    ).dropna()
    
    print(f"\n  {'Time Remaining':<14} {'N':>7} {'Model Brier':>12} {'Market Brier':>13} "
          f"{'Advantage':>10} {'Avg B/A Width':>14}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*13} {'-'*10} {'-'*14}")
    
    for idx, row in time_stats.iterrows():
        marker = "✅" if row['avg_advantage'] > 0 else "❌"
        print(f"  {str(idx):<14} {int(row['n']):>7,} {row['model_brier']:>12.6f} "
              f"{row['market_brier']:>13.6f} {row['avg_advantage']:>+10.6f} "
              f"{row['avg_spread_width']:>13.1f}¢ {marker}")
    
    # Edge decay plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = range(len(time_stats))
    ax1.bar(x, time_stats['avg_advantage'], 
            color=['#2ecc71' if v > 0 else '#e74c3c' for v in time_stats['avg_advantage']],
            alpha=0.7, label='Brier Advantage')
    ax1.set_ylabel('Brier Advantage (+ = model wins)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_stats.index, rotation=45, ha='right')
    ax1.set_xlabel('Minutes Remaining')
    ax1.set_title('Model Advantage vs Market Over Game Time')
    
    # Overlay spread width
    ax2 = ax1.twinx()
    ax2.plot(x, time_stats['avg_spread_width'], 'ko-', alpha=0.5, label='Avg B/A Spread (¢)')
    ax2.set_ylabel('Avg Bid-Ask Spread (¢)')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edge_decay.png', dpi=150)
    plt.close()
    print(f"\n  📊 Saved edge decay plot to {output_dir}/edge_decay.png")


def analyze_daily_brier(df: pd.DataFrame, output_dir: str):
    """Day-by-day Brier advantage to see consistency."""
    
    print(f"\n{'─' * 90}")
    print(f"  DAILY CONSISTENCY: Does the model beat the market every day?")
    print(f"{'─' * 90}")
    
    daily = df.groupby('date').agg(
        n=('brier_advantage', 'count'),
        model_brier=('model_brier', 'mean'),
        market_brier=('market_brier', 'mean'),
        advantage=('brier_advantage', 'mean'),
    )
    
    wins = (daily['advantage'] > 0).sum()
    total = len(daily)
    
    print(f"\n  Model beats market on {wins}/{total} days ({100*wins/max(total,1):.0f}%)")
    print(f"\n  {'Date':<14} {'N Pred':>7} {'Model Brier':>12} {'Market Brier':>13} {'Advantage':>10}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*13} {'-'*10}")
    
    for date, row in daily.iterrows():
        marker = "✅" if row['advantage'] > 0 else "❌"
        print(f"  {str(date):<14} {int(row['n']):>7,} {row['model_brier']:>12.6f} "
              f"{row['market_brier']:>13.6f} {row['advantage']:>+10.6f} {marker}")
    
    # Daily advantage bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in daily['advantage']]
    bars = ax.bar(range(len(daily)), daily['advantage'], color=colors, alpha=0.8)
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels([str(d) for d in daily.index], rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Brier Advantage (+ = model wins)')
    ax.set_title(f'Daily Model vs Market Brier Advantage ({wins}/{total} winning days)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_brier.png', dpi=150)
    plt.close()
    print(f"\n  📊 Saved daily Brier chart to {output_dir}/daily_brier.png")


def print_verdict(overall, pred_df, trades_df):
    """Final summary and actionable recommendations."""
    
    print("\n" + "=" * 90)
    print("  VERDICT: SHOULD YOU KEEP TRADING?")
    print("=" * 90)
    
    adv = overall['advantage']
    se = overall['se']
    
    # Cumulative P&L
    cum_pnl = trades_df['realized_pnl'].sum() if not trades_df.empty else 0
    n_trades = len(trades_df)
    n_days = trades_df['date'].nunique() if not trades_df.empty else 0
    
    print(f"""
  📊 Evidence Summary:
    • Brier advantage: {adv:+.6f} ± {se:.6f} ({"SIGNIFICANT" if adv > 2*se else "NOT significant"})
    • Cumulative P&L:  ${cum_pnl:.2f} over {n_trades:,} trades across {n_days} days
    • Avg daily P&L:   ${cum_pnl/max(n_days,1):.2f}/day
    """)
    
    if adv > 2 * se and cum_pnl > 0:
        print("""  ✅ KEEP GOING — Your model has a statistically significant Brier advantage AND
     you're making money. This is real edge. Focus on the game phases and spread
     levels where your advantage is strongest (check the heatmap).
     
     ⚠️  BUT: Monitor weekly. Edge can disappear if the market adapts.""")
    
    elif adv > 0 and cum_pnl > 0:
        print("""  🟡 CAUTIOUSLY CONTINUE — You're making money and the model is slightly better
     than the market, but it's not statistically significant yet. You need more data
     (aim for 50+ trading days). The profit could be variance.
     
     📋 Action items:
       1. Identify your best phase/spread combos from the heatmap
       2. Consider ONLY trading in those sweet spots
       3. Track cumulative Brier advantage weekly""")
    
    elif adv <= 0 and cum_pnl > 0:
        print("""  🟠 LUCKY SO FAR — Your model doesn't beat the market on predictions, but you're
     still making money. This is likely because:
       • You're capturing favorable bid-ask spreads (execution alpha)
       • You're trading at moments of market inefficiency
       • Variance is helping you
     
     📋 Action items:
       1. This could reverse. Reduce position sizes.
       2. Focus your research on improving the model's calibration.
       3. Consider if the trading fees you save (maker vs taker) explain the P&L.""")
    
    else:
        print("""  ❌ PAUSE AND REASSESS — Model doesn't beat market AND you're losing money.
     
     📋 Action items:
       1. Stop live trading temporarily
       2. Focus on improving model calibration (see calibration chart)
       3. Backtest on historical data before resuming
       4. Consider a simpler strategy (e.g., only trade when edge > 10%)""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    print("=" * 90)
    print("  EDGE TIMING ANALYSIS")
    print("  When and where does your model beat the market?")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    
    # Load data
    print("\n📡 Loading data...")
    pred_df = load_predictions(args.start, args.end)
    trades_df = load_trades(args.start, args.end)
    
    if pred_df.empty:
        print("❌ No settled predictions found. Run the bot for a while first!")
        return
    
    print(f"   Found {len(pred_df):,} settled predictions")
    print(f"   Found {len(trades_df):,} closed trades")
    print(f"   Date range: {pred_df['date'].min()} to {pred_df['date'].max()}")
    
    # Setup output
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Run analyses
    overall = analyze_overall_brier(pred_df)
    
    analyze_brier_by_dimension(pred_df, 'game_phase', 
                                "2. BRIER BY GAME PHASE (When in the game does the model shine?)")
    analyze_brier_by_dimension(pred_df, 'spread_category',
                                "3. BRIER BY SPREAD LEVEL (Which spreads does the model predict best?)")
    analyze_brier_by_dimension(pred_df, 'score_margin',
                                "4. BRIER BY SCORE MARGIN (Tight games vs blowouts?)")
    analyze_brier_by_dimension(pred_df, 'quarter',
                                "5. BRIER BY QUARTER")
    
    analyze_calibration(pred_df, REPORT_DIR)
    analyze_edge_heatmap(pred_df, REPORT_DIR)
    analyze_edge_decay(pred_df, REPORT_DIR)
    analyze_daily_brier(pred_df, REPORT_DIR)
    analyze_pnl_conditional(trades_df, REPORT_DIR)
    
    print_verdict(overall, pred_df, trades_df)
    
    print(f"\n📂 All charts saved to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
