#!/usr/bin/env python
"""
Performance Summary Report - Aggregate analysis across date range.

Generates comprehensive analysis of trading performance across multiple days,
including game-level ROI analysis.

Usage:
    python -m spread_src.scripts.performance_summary --start-date 2025-12-12 --end-date 2025-12-14
    python -m spread_src.scripts.performance_summary --all  # All available data
"""

import sys
import os
import re
import sqlite3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
DB_PATH = 'data/nba_data.db'

# Plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Generate performance summary across date range.")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all", action='store_true', help="Use all available data")
    parser.add_argument("--output-dir", type=str, default="reports", help="Base output directory")
    parser.add_argument("--min-edge", type=float, default=2.0, help="Minimum edge in cents to include in analysis")
    return parser.parse_args()


def load_trades_from_db(start_date=None, end_date=None, min_edge=2.0):
    """Load trades from local database for a date range."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
    if start_date and end_date:
        query = """
            SELECT 
                trade_id, timestamp, ticker, game_id, side,
                order_price, fill_price, size,
                model_fair_value, model_ci_lower, model_ci_upper,
                market_spread, seconds_remaining,
                position_before, position_after,
                realized_pnl, status,
                created_at, filled_at, closed_at
            FROM trades
            WHERE DATE(created_at) BETWEEN ? AND ?
            ORDER BY created_at
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    else:
        query = """
            SELECT 
                trade_id, timestamp, ticker, game_id, side,
                order_price, fill_price, size,
                model_fair_value, model_ci_lower, model_ci_upper,
                market_spread, seconds_remaining,
                position_before, position_after,
                realized_pnl, status,
                created_at, filled_at, closed_at
            FROM trades
            ORDER BY created_at
        """
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['created_at'].dt.date
        
        # Derived fields
        # Edge = difference between model fair value and fill price (in cents)
        df['edge'] = abs(df['model_fair_value'] - df['fill_price'])
        df = df[df['edge'] >= min_edge]
        df['ci_width'] = df['model_ci_upper'] - df['model_ci_lower']
        df['mins_remaining'] = df['seconds_remaining'] / 60.0
        
        # Extract spread line from ticker
        df['spread_line'] = df['ticker'].apply(extract_spread_line)
        df['matchup'] = df['ticker'].apply(extract_matchup)
        
        # Extract game identifier more robustly
        df['game_key'] = df['ticker'].apply(extract_game_key)
        
        # Extract actual game date from game_id (e.g., 25DEC11BOS -> 2025-12-11)
        df['game_date'] = df['game_id'].apply(extract_game_date)
    
    return df


def extract_spread_line(ticker):
    """Extract spread line from ticker (e.g., OKC5 -> 5)."""
    if not ticker:
        return None
    match = re.search(r'-([A-Z]+)(\d+)$', ticker)
    if match:
        return int(match.group(2))
    return None


def extract_matchup(ticker):
    """Extract matchup from ticker."""
    if not ticker or 'KXNBASPREAD' not in ticker:
        return None
    # Pattern: KXNBASPREAD-25DEC12MINGSW-GSW7
    match = re.search(r'-\d{2}[A-Z]{3}\d{2}([A-Z]+)([A-Z]+)-', ticker)
    if match:
        return f"{match.group(1)} @ {match.group(2)}"
    return None


def extract_game_key(ticker):
    """Extract game key from ticker (e.g., 25DEC12MINGSW from KXNBASPREAD-25DEC12MINGSW-GSW7)."""
    if not ticker or 'KXNBASPREAD' not in ticker:
        return None
    parts = ticker.split('-')
    if len(parts) >= 2:
        return parts[1]  # e.g., 25DEC12MINGSW
    return None


def extract_game_date(game_id):
    """Extract actual game date from game_id (e.g., 25DEC11BOS -> 2025-12-11)."""
    if not game_id:
        return None
    try:
        # game_id format: 25DEC11BOS (YYMMMDD + team code)
        # Extract first 7 chars: 25DEC11
        date_part = game_id[:7]
        year = 2000 + int(date_part[:2])  # 25 -> 2025
        month_str = date_part[2:5]  # DEC
        day = int(date_part[5:7])  # 11
        
        # Month mapping
        months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                  'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        month = months.get(month_str.upper(), 1)
        
        return datetime(year, month, day).date()
    except:
        return None


def analyze_game_roi(df, output_dir):
    """Analyze ROI per game."""
    print("\n📊 Game-Level ROI Analysis...")
    
    # Only use closed/filled trades with realized P&L
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    filled = filled.dropna(subset=['realized_pnl', 'fill_price', 'size'])
    
    if filled.empty:
        print("  No filled trades to analyze")
        return {}
    
    # Calculate investment per trade
    # For a BUY: you pay fill_price cents per contract
    # For a SELL: you receive fill_price cents, but your max loss is (100 - fill_price)
    # Investment = capital at risk
    def calc_investment(row):
        if row['side'] == 'buy':
            return (row['fill_price'] / 100) * row['size']
        else:  # sell
            return ((100 - row['fill_price']) / 100) * row['size']
    
    filled['investment'] = filled.apply(calc_investment, axis=1)
    
    # Group by game_key
    game_stats = filled.groupby('game_key').agg({
        'investment': 'sum',
        'realized_pnl': 'sum',
        'trade_id': 'count',
        'matchup': 'first',
        'date': 'first'
    }).reset_index()
    game_stats.columns = ['game_key', 'investment', 'profit', 'num_trades', 'matchup', 'date']
    
    # Calculate ROI and profit per trade
    game_stats['roi_pct'] = (game_stats['profit'] / game_stats['investment']) * 100
    game_stats['profit_per_trade'] = game_stats['profit'] / game_stats['num_trades']
    game_stats = game_stats.sort_values('date')
    
    # Print summary
    print(f"  Games analyzed: {len(game_stats)}")
    print(f"  Total investment: ${game_stats['investment'].sum():.2f}")
    print(f"  Total profit: ${game_stats['profit'].sum():.2f}")
    avg_roi = (game_stats['profit'].sum() / game_stats['investment'].sum()) * 100
    print(f"  Overall ROI: {avg_roi:.1f}%")
    
    # Plot ROI by game - 3 subplots now
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    fig.suptitle(f'Game-Level ROI Analysis ({len(game_stats)} games)', fontsize=14)
    
    # Create labels using game_key (e.g., 25DEC04UTABKN)
    labels = []
    for _, row in game_stats.iterrows():
        if row['game_key']:
            labels.append(row['game_key'])
        else:
            labels.append('?')
    x_pos = range(len(game_stats))
    
    # Top chart: ROI % by game
    colors = ['green' if x >= 0 else 'red' for x in game_stats['roi_pct']]
    axes[0].bar(x_pos, game_stats['roi_pct'], color=colors, alpha=0.7)
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].axhline(avg_roi, color='blue', linewidth=1, linestyle='--', label=f'Avg ROI: {avg_roi:.1f}%')
    axes[0].set_ylabel('ROI %')
    axes[0].set_title('ROI by Game')
    axes[0].legend()
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Middle chart: Profit $ by game
    colors2 = ['green' if x >= 0 else 'red' for x in game_stats['profit']]
    axes[1].bar(x_pos, game_stats['profit'], color=colors2, alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Total Profit ($)')
    axes[1].set_title('Total Profit by Game')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Bottom chart: Profit per trade by game
    colors3 = ['green' if x >= 0 else 'red' for x in game_stats['profit_per_trade']]
    avg_ppt = game_stats['profit'].sum() / game_stats['num_trades'].sum()
    axes[2].bar(x_pos, game_stats['profit_per_trade'], color=colors3, alpha=0.7)
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].axhline(avg_ppt, color='blue', linewidth=1, linestyle='--', label=f'Avg: ${avg_ppt:.3f}/trade')
    axes[2].set_ylabel('Profit per Trade ($)')
    axes[2].set_title('Profit per Trade by Game')
    axes[2].legend()
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'game_roi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print top/bottom games
    print("\n  Top 5 Games by ROI:")
    top5 = game_stats.nlargest(5, 'roi_pct')
    for _, row in top5.iterrows():
        label = row['game_key']
        print(f"    {label}: {row['roi_pct']:+.1f}% (${row['profit']:+.2f} / ${row['investment']:.2f})")
    
    print("\n  Bottom 5 Games by ROI:")
    bottom5 = game_stats.nsmallest(5, 'roi_pct')
    for _, row in bottom5.iterrows():
        label = row['game_key']
        print(f"    {label}: {row['roi_pct']:+.1f}% (${row['profit']:+.2f} / ${row['investment']:.2f})")
    
    return {
        'total_games': len(game_stats),
        'total_investment': game_stats['investment'].sum(),
        'total_profit': game_stats['profit'].sum(),
        'avg_roi': avg_roi,
        'games': game_stats.to_dict('records')
    }


def analyze_edge_summary(df, output_dir):
    """Analyze edge demanded vs profitability (aggregated)."""
    print("\n📊 Edge Analysis (Aggregate)...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty:
        print("  No filled trades to analyze")
        return {}
    
    # Create granular edge buckets (in cents)
    filled['edge_bucket'] = pd.cut(
        filled['edge'],
        bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 100],
        labels=['0-2¢', '2-4¢', '4-6¢', '6-8¢', '8-10¢', '10-12¢', '12-14¢', '14-16¢', '16-18¢', '18-20¢', '20-25¢', '25-30¢', '30¢+']
    )
    
    edge_stats = filled.groupby('edge_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    edge_stats.columns = ['edge_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    print(edge_stats.to_string(index=False))
    
    # Plot - 2 subplots: Total P&L and Avg P&L per Trade
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('P&L by Edge Demanded (cents)', fontsize=14)
    
    # Left: Total P&L
    colors = ['green' if x >= 0 else 'red' for x in edge_stats['total_pnl']]
    axes[0].bar(range(len(edge_stats)), edge_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(edge_stats)))
    axes[0].set_xticklabels(edge_stats['edge_bucket'], rotation=45, ha='right')
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('Total P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Right: Avg P&L per Trade
    colors2 = ['green' if x >= 0 else 'red' for x in edge_stats['avg_pnl']]
    axes[1].bar(range(len(edge_stats)), edge_stats['avg_pnl'], color=colors2)
    axes[1].set_xticks(range(len(edge_stats)))
    axes[1].set_xticklabels(edge_stats['edge_bucket'], rotation=45, ha='right')
    axes[1].set_ylabel('Avg P&L per Trade ($)')
    axes[1].set_title('P&L per Trade')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_analysis.png', dpi=150)
    plt.close()
    
    # Edge distribution histogram (in cents)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(filled['edge'].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edge Demanded (cents)')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Distribution of Edge Demanded')
    ax.axvline(filled['edge'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {filled["edge"].median():.1f}¢')
    ax.axvline(filled['edge'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {filled["edge"].mean():.1f}¢')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_histogram.png', dpi=150)
    plt.close()
    
    return edge_stats.to_dict('records')


def analyze_time_summary(df, output_dir):
    """Analyze performance by time remaining (aggregated)."""
    print("\n📊 Time Remaining Analysis (Aggregate)...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty or filled['seconds_remaining'].isna().all():
        print("  No time data available")
        return {}
    
    filled['time_bucket'] = pd.cut(
        filled['mins_remaining'],
        bins=[0, 5, 12, 24, 36, 60],
        labels=['Crunch (<5m)', '4th Qtr (5-12m)', '2nd Half (12-24m)', '1st Half (24-36m)', 'Early (>36m)']
    )
    
    time_stats = filled.groupby('time_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    time_stats.columns = ['time_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    print(time_stats.to_string(index=False))
    
    # Plot - 2 subplots: Total P&L and Avg P&L per Trade
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('P&L by Game Phase', fontsize=14)
    
    # Left: Total P&L
    colors = ['green' if x >= 0 else 'red' for x in time_stats['total_pnl']]
    axes[0].bar(range(len(time_stats)), time_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(time_stats)))
    axes[0].set_xticklabels(time_stats['time_bucket'], rotation=45, ha='right')
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('Total P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Right: Avg P&L per Trade
    colors2 = ['green' if x >= 0 else 'red' for x in time_stats['avg_pnl']]
    axes[1].bar(range(len(time_stats)), time_stats['avg_pnl'], color=colors2)
    axes[1].set_xticks(range(len(time_stats)))
    axes[1].set_xticklabels(time_stats['time_bucket'], rotation=45, ha='right')
    axes[1].set_ylabel('Avg P&L per Trade ($)')
    axes[1].set_title('P&L per Trade')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_analysis.png', dpi=150)
    plt.close()
    
    return time_stats.to_dict('records')


def analyze_time_distribution(df, output_dir):
    """Generate histogram of time remaining for filled trades."""
    print("\n📊 Time Distribution Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty or filled['mins_remaining'].isna().all():
        print("  No time data available")
        return {}
        
    mean_val = filled['mins_remaining'].mean()
    median_val = filled['mins_remaining'].median()
    
    print(f"  Mean time remaining: {mean_val:.1f} mins")
    print(f"  Median time remaining: {median_val:.1f} mins")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(filled['mins_remaining'], bins=48, kde=True, color='purple', ax=ax)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}m')
    ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}m')
    
    ax.set_xlabel('Minutes Remaining')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Distribution of Filled Trades by Game Time')
    ax.set_xlim(48, 0) # Invert X to show game flow
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_distribution.png', dpi=150)
    plt.close()
    
    return {
        'mean_mins': mean_val,
        'median_mins': median_val
    }


def analyze_joint_time_edge_pnl(df, output_dir):
    """Analyze P&L across both time remaining and edge demanded using a heatmap."""
    print("\n📊 Joint Time-Edge Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty or filled['mins_remaining'].isna().all() or filled['edge'].isna().all():
        print("  Insufficient data for joint analysis")
        return
    
    # Create buckets for heatmap
    # Time: 4-minute chunks (12 buckets total, 3 per quarter)
    filled['time_chunk'] = pd.cut(
        filled['mins_remaining'],
        bins=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        labels=[
            'Q4: 0-4m', 'Q4: 4-8m', 'Q4: 8-12m', 
            'Q3: 12-16m', 'Q3: 16-20m', 'Q3: 20-24m',
            'Q2: 24-28m', 'Q2: 28-32m', 'Q2: 32-36m',
            'Q1: 36-40m', 'Q1: 40-44m', 'Q1: 44-48m'
        ]
    )
    
    # Edge: Granular 2-cent chunks
    filled['edge_chunk'] = pd.cut(
        filled['edge'],
        bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 100],
        labels=['0-2¢', '2-4¢', '4-6¢', '6-8¢', '8-10¢', '10-12¢', '12-14¢', '14-16¢', '16-18¢', '18-20¢', '20-25¢', '25-30¢', '30¢+']
    )
    
    # Pivot for heatmap: Mean realized_pnl
    pivot_table = filled.pivot_table(
        index='time_chunk', 
        columns='edge_chunk', 
        values='realized_pnl', 
        aggfunc='mean',
        observed=True
    )
    
    # Counter for volume
    pivot_counts = filled.pivot_table(
        index='time_chunk', 
        columns='edge_chunk', 
        values='trade_id', 
        aggfunc='count',
        observed=True
    )
    
    # Reverse index to show game chronologically (top to bottom)
    labels_rev = [
        'Q1: 44-48m', 'Q1: 40-44m', 'Q1: 36-40m',
        'Q2: 32-36m', 'Q2: 28-32m', 'Q2: 24-28m',
        'Q3: 20-24m', 'Q3: 16-20m', 'Q3: 12-16m',
        'Q4: 8-12m', 'Q4: 4-8m', 'Q4: 0-4m'
    ]
    pivot_table = pivot_table.reindex(labels_rev)
    pivot_counts = pivot_counts.reindex(labels_rev)
    
    # Increase height for more rows
    fig, ax = plt.subplots(figsize=(22, 12))

    # Create annotation string (Value + Count)
    annot = pivot_table.copy().astype(str)
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            val = pivot_table.iloc[i, j]
            count = pivot_counts.iloc[i, j]
            if pd.isna(val):
                annot.iloc[i, j] = ""
            else:
                annot.iloc[i, j] = f"${val:+.2f}\n(n={int(count)})"

    sns.heatmap(pivot_table, annot=annot, fmt="", cmap='RdYlGn', center=0, ax=ax, vmin=-0.2, vmax=0.25, cbar_kws={'label': 'Avg Realized P&L ($)'})
    
    ax.set_title('Avg P&L per Trade: Game Time vs Edge Demanded')
    ax.set_ylabel('Game Phase')
    ax.set_xlabel('Edge Demanded (cents)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'joint_time_edge_pnl.png', dpi=150)
    plt.close()


def analyze_market_spread_summary(df, output_dir):
    """Analyze market spread width vs performance (aggregated)."""
    print("\n📊 Market Spread Analysis (Aggregate)...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty or filled['market_spread'].isna().all():
        print("  No market spread data available")
        return {}
    
    filled['spread_bucket'] = pd.cut(
        filled['market_spread'],
        bins=[0, 5, 10, 15, 20, 100],
        labels=['Tight (0-5¢)', 'Normal (5-10¢)', 'Wide (10-15¢)', 'Very Wide (15-20¢)', 'Very Wide (>20¢)']
    )
    
    spread_stats = filled.groupby('spread_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    spread_stats.columns = ['spread_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    print(spread_stats.to_string(index=False))
    
    # Plot - 2 subplots: Total P&L and Avg P&L per Trade
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('P&L by Market Spread Width', fontsize=14)
    
    # Left: Total P&L
    colors = ['green' if x >= 0 else 'red' for x in spread_stats['total_pnl']]
    axes[0].bar(range(len(spread_stats)), spread_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(spread_stats)))
    axes[0].set_xticklabels(spread_stats['spread_bucket'], rotation=45, ha='right')
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('Total P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Right: Avg P&L per Trade
    colors2 = ['green' if x >= 0 else 'red' for x in spread_stats['avg_pnl']]
    axes[1].bar(range(len(spread_stats)), spread_stats['avg_pnl'], color=colors2)
    axes[1].set_xticks(range(len(spread_stats)))
    axes[1].set_xticklabels(spread_stats['spread_bucket'], rotation=45, ha='right')
    axes[1].set_ylabel('Avg P&L per Trade ($)')
    axes[1].set_title('P&L per Trade')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'market_spread_analysis.png', dpi=150)
    plt.close()
    
    return spread_stats.to_dict('records')


def analyze_spread_line_summary(df, output_dir):
    """Analyze performance by spread line magnitude (aggregated)."""
    print("\n📊 Spread Line Analysis (Aggregate)...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    filled = filled.dropna(subset=['spread_line'])
    
    if filled.empty:
        print("  No spread line data available")
        return {}
    
    filled['line_bucket'] = pd.cut(
        filled['spread_line'],
        bins=[0, 5, 10, 15, 100],
        labels=['Small (1-5)', 'Medium (6-10)', 'Large (11-15)', 'Huge (>15)']
    )
    
    line_stats = filled.groupby('line_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    line_stats.columns = ['line_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    print(line_stats.to_string(index=False))
    
    # Plot - 2 subplots: Total P&L and Avg P&L per Trade
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('P&L by Spread Line Magnitude', fontsize=14)
    
    # Left: Total P&L
    colors = ['green' if x >= 0 else 'red' for x in line_stats['total_pnl']]
    axes[0].bar(range(len(line_stats)), line_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(line_stats)))
    axes[0].set_xticklabels(line_stats['line_bucket'], rotation=45, ha='right')
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('Total P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Right: Avg P&L per Trade
    colors2 = ['green' if x >= 0 else 'red' for x in line_stats['avg_pnl']]
    axes[1].bar(range(len(line_stats)), line_stats['avg_pnl'], color=colors2)
    axes[1].set_xticks(range(len(line_stats)))
    axes[1].set_xticklabels(line_stats['line_bucket'], rotation=45, ha='right')
    axes[1].set_ylabel('Avg P&L per Trade ($)')
    axes[1].set_title('P&L per Trade')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_line_analysis.png', dpi=150)
    plt.close()
    
    return line_stats.to_dict('records')


def analyze_side_performance(df, output_dir):
    """Compare performance of longs (buy) vs shorts (sell)."""
    print("\n📊 Long vs Short Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty:
        print("  No filled trades to analyze")
        return {}
    
    side_stats = filled.groupby('side').agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    side_stats.columns = ['side', 'trades', 'total_pnl', 'avg_pnl']
    
    # Map side to readable names if needed (usually 'buy'/'sell')
    side_stats['label'] = side_stats['side'].apply(lambda x: 'Long (BUY)' if x.lower() == 'buy' else 'Short (SELL)')
    
    print(side_stats[['label', 'trades', 'total_pnl', 'avg_pnl']].to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance: Long (BUY) vs Short (SELL)', fontsize=14)
    
    # Left: Total P&L
    colors = ['green' if x >= 0 else 'red' for x in side_stats['total_pnl']]
    axes[0].bar(side_stats['label'], side_stats['total_pnl'], color=colors, alpha=0.7)
    axes[0].set_ylabel('Total realized P&L ($)')
    axes[0].set_title('Total P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Right: Avg P&L per Trade
    colors2 = ['green' if x >= 0 else 'red' for x in side_stats['avg_pnl']]
    axes[1].bar(side_stats['label'], side_stats['avg_pnl'], color=colors2, alpha=0.7)
    axes[1].set_ylabel('Avg Profit per Trade ($)')
    axes[1].set_title('Efficiency (Profit per Trade)')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'side_performance.png', dpi=150)
    plt.close()
    
    return side_stats.to_dict('records')


def analyze_side_phase_performance(df, output_dir):
    """Compare longs vs shorts across game phases."""
    print("\n📊 Long vs Short by Game Phase...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty or filled['mins_remaining'].isna().all():
        print("  No data for phase analysis")
        return {}
    
    # 4-minute chunks (12 buckets total, 3 per quarter)
    filled['time_bucket'] = pd.cut(
        filled['mins_remaining'],
        bins=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        labels=[
            'Q4: 0-4m', 'Q4: 4-8m', 'Q4: 8-12m', 
            'Q3: 12-16m', 'Q3: 16-20m', 'Q3: 20-24m',
            'Q2: 24-28m', 'Q2: 28-32m', 'Q2: 32-36m',
            'Q1: 36-40m', 'Q1: 40-44m', 'Q1: 44-48m'
        ]
    )
    
    side_phase = filled.groupby(['time_bucket', 'side'], observed=True).agg({
        'realized_pnl': 'mean',
        'trade_id': 'count'
    }).reset_index()
    side_phase.columns = ['time_bucket', 'side', 'avg_pnl', 'count']
    
    # Map side
    side_phase['side_label'] = side_phase['side'].apply(lambda x: 'LONG (BUY)' if x.lower() == 'buy' else 'SHORT (SELL)')
    
    # Set chronological order (Q1 -> Q4)
    chrono_order = [
        'Q1: 44-48m', 'Q1: 40-44m', 'Q1: 36-40m',
        'Q2: 32-36m', 'Q2: 28-32m', 'Q2: 24-28m',
        'Q3: 20-24m', 'Q3: 16-20m', 'Q3: 12-16m',
        'Q4: 8-12m', 'Q4: 4-8m', 'Q4: 0-4m'
    ]
    side_phase['time_bucket'] = pd.Categorical(side_phase['time_bucket'], categories=chrono_order, ordered=True)
    side_phase = side_phase.sort_values('time_bucket')
    
    # Plot
    plt.figure(figsize=(18, 8))
    sns.barplot(data=side_phase, x='time_bucket', y='avg_pnl', hue='side_label', palette='RdYlGn_r') # Inverted so SELL is more distinct if needed, or just standard
    
    plt.axhline(0, color='black', linewidth=1.0)
    plt.title('High-Res Strategy Performance: Longs vs Shorts by Game Phase (4min Buckets)')
    plt.ylabel('Avg Realized P&L ($)')
    plt.xlabel('Game Phase (Opening -> Finish)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add counts above bars
    # Using a slightly offset position for text is tricky with seaborn barplot, but we can try
    # or just keep it clean.
    
    plt.tight_layout()
    plt.savefig(output_dir / 'side_phase_performance.png', dpi=150)
    plt.close()
    
    return side_phase.to_dict('records')


def analyze_daily_summary(df, output_dir):
    """Analyze performance by day, including profit per game."""
    print("\n📊 Daily Performance...")
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty:
        print("  No filled trades")
        return {}
    
    # Drop rows without valid game_date
    filled = filled.dropna(subset=['game_date'])
    
    if filled.empty:
        print("  No trades with valid game dates")
        return {}
    
    # Count unique games per game date (actual game date, not trade date)
    games_per_day = filled.groupby('game_date')['game_id'].nunique().reset_index()
    games_per_day.columns = ['game_date', 'games']
    
    # Group by game date for P&L
    daily = filled.groupby('game_date').agg({
        'trade_id': 'count',
        'realized_pnl': 'sum',
    }).reset_index()
    daily.columns = ['game_date', 'trades', 'pnl']
    
    # Merge games count
    daily = daily.merge(games_per_day, on='game_date', how='left')
    daily['games'] = daily['games'].fillna(1).astype(int)
    
    # Calculate profit per game
    daily['profit_per_game'] = daily['pnl'] / daily['games']
    daily['cumulative_pnl'] = daily['pnl'].cumsum()
    
    print(daily.to_string(index=False))
    
    # Plot with 3 subplots now
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle('Daily Performance', fontsize=14)
    
    # Daily P&L
    colors = ['green' if x >= 0 else 'red' for x in daily['pnl']]
    axes[0].bar(range(len(daily)), daily['pnl'], color=colors)
    axes[0].set_xticks(range(len(daily)))
    axes[0].set_xticklabels([str(d) for d in daily['game_date']], rotation=45, ha='right')
    axes[0].set_ylabel('P&L ($)')
    axes[0].set_title('Daily P&L')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Profit per game
    colors2 = ['green' if x >= 0 else 'red' for x in daily['profit_per_game']]
    axes[1].bar(range(len(daily)), daily['profit_per_game'], color=colors2)
    axes[1].set_xticks(range(len(daily)))
    axes[1].set_xticklabels([str(d) for d in daily['game_date']], rotation=45, ha='right')
    axes[1].set_ylabel('Profit per Game ($)')
    axes[1].set_title('Profit per Game over Time')
    axes[1].axhline(0, color='black', linewidth=0.5)
    # Add game count labels
    for i, (_, row) in enumerate(daily.iterrows()):
        axes[1].annotate(f'{row["games"]}g', 
                        xy=(i, row['profit_per_game']),
                        ha='center', va='bottom' if row['profit_per_game'] >= 0 else 'top',
                        fontsize=8, alpha=0.7)
    
    # Cumulative P&L
    axes[2].plot(range(len(daily)), daily['cumulative_pnl'], marker='o', linewidth=2)
    axes[2].fill_between(range(len(daily)), 0, daily['cumulative_pnl'], 
                         where=daily['cumulative_pnl'] >= 0, alpha=0.3, color='green')
    axes[2].fill_between(range(len(daily)), 0, daily['cumulative_pnl'], 
                         where=daily['cumulative_pnl'] < 0, alpha=0.3, color='red')
    axes[2].set_xticks(range(len(daily)))
    axes[2].set_xticklabels([str(d) for d in daily['game_date']], rotation=45, ha='right')
    axes[2].set_ylabel('Cumulative P&L ($)')
    axes[2].set_title('Cumulative P&L')
    axes[2].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_performance.png', dpi=150)
    plt.close()
    
    # Print avg profit per game
    avg_ppg = daily['pnl'].sum() / daily['games'].sum()
    print(f"\n  Average profit per game: ${avg_ppg:.2f}")
    
    return daily.to_dict('records')


def generate_summary(start_date, end_date, df, analyses, output_dir):
    """Generate text summary."""
    summary_path = output_dir / 'summary.txt'
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])]
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PERFORMANCE SUMMARY: {start_date} to {end_date}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall stats
        f.write("📊 OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Trading Days: {df['date'].nunique()}\n")
        f.write(f"Total Orders: {len(df)}\n")
        f.write(f"Total Fills: {len(filled)}\n")
        f.write(f"Fill Rate: {len(filled)/len(df)*100:.1f}%\n")
        
        total_pnl = filled['realized_pnl'].sum()
        f.write(f"Total P&L: ${total_pnl:.2f}\n")
        f.write(f"Avg P&L per Trade: ${filled['realized_pnl'].mean():.3f}\n")
        f.write("\n")
        
        # Game ROI summary
        if 'game_roi' in analyses and analyses['game_roi']:
            roi_data = analyses['game_roi']
            f.write("📈 GAME ROI SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Games Traded: {roi_data['total_games']}\n")
            f.write(f"Total Investment: ${roi_data['total_investment']:.2f}\n")
            f.write(f"Total Profit: ${roi_data['total_profit']:.2f}\n")
            f.write(f"Overall ROI: {roi_data['avg_roi']:.1f}%\n")
            f.write("\n")
        
        f.write("📁 FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        for file in output_dir.glob('*.png'):
            f.write(f"• {file.name}\n")
        f.write(f"• summary.txt\n")
    
    print(f"\n✅ Summary saved to {summary_path}")


def main():
    args = parse_args()
    
    # Determine date range
    if args.all:
        start_date = None
        end_date = None
        date_label = "all_time"
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
        date_label = f"{start_date}_{end_date}"
    else:
        # Default: last 7 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        date_label = f"{start_date}_{end_date}"
    
    print("=" * 60)
    if start_date and end_date:
        print(f"PERFORMANCE SUMMARY: {start_date} to {end_date}")
    else:
        print("PERFORMANCE SUMMARY: All Time")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"summary_{date_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load data
    print("\n📥 Loading data from database...")
    df = load_trades_from_db(start_date, end_date, min_edge=args.min_edge)
    
    if df.empty:
        print("❌ No trades found!")
        return
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])]
    print(f"  ✓ Loaded {len(df)} trades ({len(filled)} filled)")
    print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  ✓ Trading days: {df['date'].nunique()}")
    
    # Run analyses
    analyses = {}
    analyses['game_roi'] = analyze_game_roi(df, output_dir)
    analyses['edge'] = analyze_edge_summary(df, output_dir)
    analyses['time'] = analyze_time_summary(df, output_dir)
    analyses['time_dist'] = analyze_time_distribution(df, output_dir)
    analyses['joint_time_edge'] = analyze_joint_time_edge_pnl(df, output_dir)
    analyses['market_spread'] = analyze_market_spread_summary(df, output_dir)
    # New analysis for longs vs shorts
    analyses['side'] = analyze_side_performance(df, output_dir)
    analyses['side_phase'] = analyze_side_phase_performance(df, output_dir)
    analyses['spread_line'] = analyze_spread_line_summary(df, output_dir)
    analyses['daily'] = analyze_daily_summary(df, output_dir)
    
    # Generate summary
    actual_start = str(df['date'].min())
    actual_end = str(df['date'].max())
    generate_summary(actual_start, actual_end, df, analyses, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ SUMMARY COMPLETE")
    print(f"📁 All files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
