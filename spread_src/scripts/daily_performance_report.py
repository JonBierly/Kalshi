#!/usr/bin/env python
"""
Daily Trade Performance Report Generator

Generates comprehensive analysis of trading performance for a specific date,
producing a dated report folder with visualizations and summary.

Analyses:
1. Edge Analysis - Edge demanded vs profitability
2. Market Spread Analysis - Spread width vs fill rate & P&L
3. Time Analysis - Seconds remaining vs performance
4. Position Analysis - What positions held at settlement
5. Liquidity Analysis - Taker vs maker performance
6. Spread Line Analysis - Spread line magnitude vs P&L
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
from data.kalshi import KalshiClient

# Configuration
DB_PATH = 'data/nba_data.db'
API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"

# Plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Generate daily trade performance report.")
    parser.add_argument("--date", type=str, required=True, 
                        help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Base output directory")
    return parser.parse_args()


def load_trades_from_db(date_str):
    """Load trades from local database for a specific date."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
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
        WHERE DATE(created_at) = ?
        ORDER BY created_at
    """
    
    df = pd.read_sql_query(query, conn, params=[date_str])
    conn.close()
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Derived fields
        df['edge'] = abs(df['model_fair_value'] - df['order_price'])
        df['edge_pct'] = df['edge'] / df['order_price'] * 100
        df['ci_width'] = df['model_ci_upper'] - df['model_ci_lower']
        df['mins_remaining'] = df['seconds_remaining'] / 60.0
        
        # Extract spread line from ticker
        df['spread_line'] = df['ticker'].apply(extract_spread_line)
        df['matchup'] = df['ticker'].apply(extract_matchup)
    
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
    match = re.search(r'-\d{2}[A-Z]{3}\d{2}([A-Z]+)([A-Z]+)-', ticker)
    if match:
        return f"{match.group(1)} vs {match.group(2)}"
    return None


def load_fills_from_api(client, date_str):
    """Load fills from Kalshi API for a specific date."""
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    start_dt = target_date
    end_dt = target_date + timedelta(days=1)
    
    min_ts = int(start_dt.timestamp() * 1000)
    max_ts = int(end_dt.timestamp() * 1000)
    
    fills = client.get_fills(min_ts=min_ts, max_ts=max_ts, limit=1000)
    
    if not fills:
        return pd.DataFrame()
    
    fill_data = []
    for fill in fills:
        ts_str = fill.get('created_time', '')
        if ts_str:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        else:
            continue
            
        fill_data.append({
            'fill_time': dt,
            'ticker': fill.get('ticker'),
            'side': fill.get('side'),
            'action': fill.get('action'),
            'count': fill.get('count', 0),
            'yes_price': fill.get('yes_price', 0),
            'no_price': fill.get('no_price', 0),
            'is_taker': fill.get('is_taker', False),
        })
    
    return pd.DataFrame(fill_data)


def load_settled_positions(client, date_str):
    """Load settled positions for markets from the target date."""
    settled = client.get_positions(settlement_status='settled', limit=1000)
    positions = settled.get('market_positions', [])
    
    # Filter to NBA spread markets for this date
    date_code = datetime.strptime(date_str, '%Y-%m-%d').strftime('%y%b%d').upper()
    
    result = []
    for pos in positions:
        ticker = pos.get('ticker', '')
        if f'-{date_code}' in ticker and 'KXNBASPREAD' in ticker:
            result.append({
                'ticker': ticker,
                'realized_pnl': pos.get('realized_pnl', 0) / 100.0,
                'fees_paid': pos.get('fees_paid', 0) / 100.0,
                'total_traded': pos.get('total_traded', 0),
            })
    
    return pd.DataFrame(result)


# ============================================================================
# DATA ENRICHMENT FUNCTIONS
# ============================================================================

def distribute_pnl_to_trades(trades_df, settled_df):
    """
    Distribute position-level P&L to individual trades.
    
    Since the database doesn't have per-trade realized P&L, we distribute 
    the position P&L proportionally based on trade size.
    """
    if settled_df.empty or trades_df.empty:
        return trades_df
    
    # Get filled trades only
    filled = trades_df[trades_df['status'].isin(['filled', 'settled'])].copy()
    
    # For each position, find trades that contributed to it
    for _, pos in settled_df.iterrows():
        ticker = pos['ticker']
        position_pnl = pos['realized_pnl']
        
        # Find all trades for this ticker
        mask = filled['ticker'] == ticker
        ticker_trades = filled[mask]
        
        if len(ticker_trades) == 0:
            continue
        
        # Distribute P&L proportionally by trade size
        total_size = ticker_trades['size'].sum()
        if total_size > 0:
            for idx in ticker_trades.index:
                trade_size = filled.loc[idx, 'size']
                trade_pnl = position_pnl * (trade_size / total_size)
                trades_df.loc[trades_df['trade_id'] == filled.loc[idx, 'trade_id'], 'realized_pnl'] = trade_pnl
    
    return trades_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_edge(df, output_dir):
    """Analyze edge demanded vs profitability."""
    print("\n📊 Edge Analysis...")
    
    # Only use filled trades with realized P&L
    filled = df[df['status'].isin(['filled', 'settled'])].copy()
    
    if filled.empty:
        print("  No filled trades to analyze")
        return {}
    
    # Create edge buckets
    filled['edge_bucket'] = pd.cut(
        filled['edge_pct'],
        bins=[0, 2, 5, 10, 100],
        labels=['Thin (0-2%)', 'Medium (2-5%)', 'Fat (5-10%)', 'Huge (>10%)']
    )
    
    # Stats by bucket
    edge_stats = filled.groupby('edge_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
        'fill_price': 'count'  # Proxy for fill count
    }).reset_index()
    edge_stats.columns = ['edge_bucket', 'orders', 'total_pnl', 'avg_pnl', 'fills']
    edge_stats['fill_rate'] = edge_stats['fills'] / edge_stats['orders'] * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Edge Analysis: Does Demanding More Edge Help?', fontsize=14)
    
    # P&L by edge
    colors = ['green' if x >= 0 else 'red' for x in edge_stats['total_pnl']]
    axes[0].bar(range(len(edge_stats)), edge_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(edge_stats)))
    axes[0].set_xticklabels(edge_stats['edge_bucket'], rotation=45)
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('P&L by Edge Demanded')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Fill rate by edge
    axes[1].bar(range(len(edge_stats)), edge_stats['fill_rate'], color='steelblue')
    axes[1].set_xticks(range(len(edge_stats)))
    axes[1].set_xticklabels(edge_stats['edge_bucket'], rotation=45)
    axes[1].set_ylabel('Fill Rate (%)')
    axes[1].set_title('Fill Rate by Edge Demanded')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_analysis.png', dpi=150)
    plt.close()
    
    print(edge_stats.to_string(index=False))
    return edge_stats.to_dict('records')


def analyze_market_spread(df, output_dir):
    """Analyze market spread width vs performance."""
    print("\n📊 Market Spread Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled'])].copy()
    
    if filled.empty or filled['market_spread'].isna().all():
        print("  No market spread data available")
        return {}
    
    # Create spread buckets
    filled['spread_bucket'] = pd.cut(
        filled['market_spread'],
        bins=[0, 5, 10, 15, 20, 100],
        labels=['Tight (0-5¢)', 'Normal (5-10¢)', 'Wide (10-15¢)', 'Very Wide (15-20¢)', 'Very Wide (>20¢)']
    )
    
    # Stats by bucket
    spread_stats = filled.groupby('spread_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    spread_stats.columns = ['spread_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Market Spread Analysis: Are Wider Spreads Better?', fontsize=14)
    
    # P&L by spread
    colors = ['green' if x >= 0 else 'red' for x in spread_stats['total_pnl']]
    axes[0].bar(range(len(spread_stats)), spread_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(spread_stats)))
    axes[0].set_xticklabels(spread_stats['spread_bucket'], rotation=45)
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('P&L by Market Spread Width')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Trade count by spread
    axes[1].bar(range(len(spread_stats)), spread_stats['trades'], color='steelblue')
    axes[1].set_xticks(range(len(spread_stats)))
    axes[1].set_xticklabels(spread_stats['spread_bucket'], rotation=45)
    axes[1].set_ylabel('Number of Trades')
    axes[1].set_title('Trade Volume by Market Spread')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_analysis.png', dpi=150)
    plt.close()
    
    print(spread_stats.to_string(index=False))
    return spread_stats.to_dict('records')


def analyze_time_remaining(df, output_dir):
    """Analyze performance by time remaining in game."""
    print("\n📊 Time Remaining Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled'])].copy()
    
    if filled.empty or filled['seconds_remaining'].isna().all():
        print("  No time data available")
        return {}
    
    # Create time buckets (NBA game is 48 minutes)
    filled['time_bucket'] = pd.cut(
        filled['mins_remaining'],
        bins=[0, 5, 12, 24, 36, 60],
        labels=['Crunch (<5m)', '4th Qtr (5-12m)', '2nd Half (12-24m)', '1st Half (24-36m)', 'Early (>36m)']
    )
    
    # Stats by bucket
    time_stats = filled.groupby('time_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    time_stats.columns = ['time_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Time Analysis: When Is Trading Most Profitable?', fontsize=14)
    
    # P&L by time
    colors = ['green' if x >= 0 else 'red' for x in time_stats['total_pnl']]
    axes[0].bar(range(len(time_stats)), time_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(time_stats)))
    axes[0].set_xticklabels(time_stats['time_bucket'], rotation=45)
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('P&L by Game Phase')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Average P&L by time
    colors2 = ['green' if x >= 0 else 'red' for x in time_stats['avg_pnl']]
    axes[1].bar(range(len(time_stats)), time_stats['avg_pnl'], color=colors2)
    axes[1].set_xticks(range(len(time_stats)))
    axes[1].set_xticklabels(time_stats['time_bucket'], rotation=45)
    axes[1].set_ylabel('Avg P&L per Trade ($)')
    axes[1].set_title('Average Profitability by Game Phase')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_analysis.png', dpi=150)
    plt.close()
    
    print(time_stats.to_string(index=False))
    return time_stats.to_dict('records')


def analyze_liquidity(fills_df, output_dir):
    """Analyze taker vs maker performance."""
    print("\n📊 Liquidity Analysis (Taker vs Maker)...")
    
    if fills_df.empty:
        print("  No fill data from API")
        return {}
    
    # Aggregate by ticker and taker status
    # Calculate profitability proxy (this is approximate without matching settlements)
    
    taker_stats = fills_df.groupby('is_taker').agg({
        'count': 'sum',
        'yes_price': 'mean',
        'no_price': 'mean',
    }).reset_index()
    taker_stats.columns = ['is_taker', 'contracts', 'avg_yes_price', 'avg_no_price']
    taker_stats['label'] = taker_stats['is_taker'].map({True: 'Taker', False: 'Maker'})
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Liquidity Analysis: Are Passive (Maker) Fills Better?', fontsize=14)
    
    ax.bar(taker_stats['label'], taker_stats['contracts'], color=['coral', 'steelblue'])
    ax.set_ylabel('Total Contracts')
    ax.set_title('Volume by Execution Type')
    
    # Add percentage labels
    total = taker_stats['contracts'].sum()
    for i, row in taker_stats.iterrows():
        pct = row['contracts'] / total * 100
        ax.annotate(f'{pct:.1f}%', 
                   xy=(row['label'], row['contracts']),
                   ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'liquidity_analysis.png', dpi=150)
    plt.close()
    
    print(taker_stats[['label', 'contracts', 'avg_yes_price', 'avg_no_price']].to_string(index=False))
    return taker_stats.to_dict('records')


def analyze_spread_lines(df, output_dir):
    """Analyze performance by spread line magnitude."""
    print("\n📊 Spread Line Analysis...")
    
    filled = df[df['status'].isin(['filled', 'settled'])].copy()
    filled = filled.dropna(subset=['spread_line'])
    
    if filled.empty:
        print("  No spread line data available")
        return {}
    
    # Create spread line buckets
    filled['line_bucket'] = pd.cut(
        filled['spread_line'],
        bins=[0, 5, 10, 15, 100],
        labels=['Small (1-5)', 'Medium (6-10)', 'Large (11-15)', 'Huge (>15)']
    )
    
    # Stats by bucket
    line_stats = filled.groupby('line_bucket', observed=True).agg({
        'trade_id': 'count',
        'realized_pnl': ['sum', 'mean'],
    }).reset_index()
    line_stats.columns = ['line_bucket', 'trades', 'total_pnl', 'avg_pnl']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Spread Line Analysis: Does Spread Magnitude Matter?', fontsize=14)
    
    # P&L by spread line
    colors = ['green' if x >= 0 else 'red' for x in line_stats['total_pnl']]
    axes[0].bar(range(len(line_stats)), line_stats['total_pnl'], color=colors)
    axes[0].set_xticks(range(len(line_stats)))
    axes[0].set_xticklabels(line_stats['line_bucket'], rotation=45)
    axes[0].set_ylabel('Total P&L ($)')
    axes[0].set_title('P&L by Spread Line Size')
    axes[0].axhline(0, color='black', linewidth=0.5)
    
    # Trade count by spread line
    axes[1].bar(range(len(line_stats)), line_stats['trades'], color='steelblue')
    axes[1].set_xticks(range(len(line_stats)))
    axes[1].set_xticklabels(line_stats['line_bucket'], rotation=45)
    axes[1].set_ylabel('Number of Trades')
    axes[1].set_title('Trade Volume by Spread Line')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_line_analysis.png', dpi=150)
    plt.close()
    
    print(line_stats.to_string(index=False))
    return line_stats.to_dict('records')


def analyze_positions(df, settled_df, output_dir):
    """Analyze what positions were held and their outcomes."""
    print("\n📊 Position Analysis...")
    
    if settled_df.empty:
        print("  No settled position data available")
        return {}
    
    # Sort by P&L to show winners and losers
    settled_df = settled_df.sort_values('realized_pnl')
    
    # Separate losers and winners
    losers = settled_df[settled_df['realized_pnl'] < 0]
    winners = settled_df[settled_df['realized_pnl'] > 0]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Position Analysis: Winners vs Losers', fontsize=14)
    
    # Losers
    if not losers.empty:
        y_pos = range(len(losers))
        axes[0].barh(y_pos, losers['realized_pnl'], color='red', alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([t.split('-')[-1] for t in losers['ticker']], fontsize=8)
        axes[0].set_xlabel('P&L ($)')
        axes[0].set_title(f'Losing Positions (n={len(losers)}, ${losers["realized_pnl"].sum():.2f})')
        axes[0].axvline(0, color='black', linewidth=0.5)
    
    # Winners
    if not winners.empty:
        y_pos = range(len(winners))
        axes[1].barh(y_pos, winners['realized_pnl'], color='green', alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([t.split('-')[-1] for t in winners['ticker']], fontsize=8)
        axes[1].set_xlabel('P&L ($)')
        axes[1].set_title(f'Winning Positions (n={len(winners)}, ${winners["realized_pnl"].sum():.2f})')
        axes[1].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_breakdown.png', dpi=150)
    plt.close()
    
    # Summary stats
    summary = {
        'total_positions': len(settled_df),
        'winners': len(winners),
        'losers': len(losers),
        'total_pnl': settled_df['realized_pnl'].sum(),
        'total_fees': settled_df['fees_paid'].sum(),
        'avg_winner': winners['realized_pnl'].mean() if not winners.empty else 0,
        'avg_loser': losers['realized_pnl'].mean() if not losers.empty else 0,
    }
    
    print(f"  Total Positions: {summary['total_positions']}")
    print(f"  Winners: {summary['winners']} (avg ${summary['avg_winner']:.2f})")
    print(f"  Losers: {summary['losers']} (avg ${summary['avg_loser']:.2f})")
    print(f"  Total P&L: ${summary['total_pnl']:.2f}")
    print(f"  Total Fees: ${summary['total_fees']:.2f}")
    
    return summary


def generate_summary(date_str, df, settled_df, analyses, output_dir):
    """Generate text summary of findings."""
    summary_path = output_dir / 'summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"DAILY PERFORMANCE REPORT: {date_str}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        # Overall stats
        f.write("📊 OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Orders Placed: {len(df)}\n")
        
        filled = df[df['status'].isin(['filled', 'settled'])]
        f.write(f"Total Fills: {len(filled)}\n")
        f.write(f"Fill Rate: {len(filled)/len(df)*100:.1f}%\n")
        
        if not settled_df.empty:
            total_pnl = settled_df['realized_pnl'].sum()
            total_fees = settled_df['fees_paid'].sum()
            f.write(f"Realized P&L: ${total_pnl:.2f}\n")
            f.write(f"Fees Paid: ${total_fees:.2f}\n")
            f.write(f"Net P&L: ${total_pnl - total_fees:.2f}\n")
        f.write("\n")
        
        # Key findings
        f.write("🔍 KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        # Edge findings
        if 'edge' in analyses and analyses['edge']:
            edge_data = analyses['edge']
            best_edge = max(edge_data, key=lambda x: x.get('total_pnl', 0))
            worst_edge = min(edge_data, key=lambda x: x.get('total_pnl', 0))
            f.write(f"• Best edge bucket: {best_edge['edge_bucket']} (${best_edge['total_pnl']:.2f})\n")
            f.write(f"• Worst edge bucket: {worst_edge['edge_bucket']} (${worst_edge['total_pnl']:.2f})\n")
        
        # Time findings
        if 'time' in analyses and analyses['time']:
            time_data = analyses['time']
            best_time = max(time_data, key=lambda x: x.get('total_pnl', 0))
            worst_time = min(time_data, key=lambda x: x.get('total_pnl', 0))
            f.write(f"• Best time phase: {best_time['time_bucket']} (${best_time['total_pnl']:.2f})\n")
            f.write(f"• Worst time phase: {worst_time['time_bucket']} (${worst_time['total_pnl']:.2f})\n")
        
        f.write("\n")
        
        # Files generated
        f.write("📁 FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        for file in output_dir.glob('*.png'):
            f.write(f"• {file.name}\n")
        f.write(f"• full_trades.csv\n")
        f.write(f"• summary.txt\n")
    
    print(f"\n✅ Summary saved to {summary_path}")


def main():
    args = parse_args()
    date_str = args.date
    
    print("=" * 60)
    print(f"DAILY PERFORMANCE REPORT: {date_str}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load data from database
    print("\n📥 Loading data from database...")
    df = load_trades_from_db(date_str)
    print(f"  ✓ Loaded {len(df)} trades from database")
    
    if df.empty:
        print("❌ No trades found for this date!")
        return
    
    # Load data from Kalshi API
    print("\n📥 Loading data from Kalshi API...")
    client = KalshiClient(API_KEY, KEY_PATH)
    
    fills_df = load_fills_from_api(client, date_str)
    print(f"  ✓ Loaded {len(fills_df)} fills from API")
    
    settled_df = load_settled_positions(client, date_str)
    print(f"  ✓ Loaded {len(settled_df)} settled positions for date")
    
    # Distribute position-level P&L to individual trades
    print("\n📊 Distributing P&L to trades...")
    df = distribute_pnl_to_trades(df, settled_df)
    trades_with_pnl = df['realized_pnl'].notna().sum()
    print(f"  ✓ {trades_with_pnl} trades now have P&L attribution")
    
    # Save raw trades
    df.to_csv(output_dir / 'full_trades.csv', index=False)
    print(f"\n✅ Saved raw trades to {output_dir / 'full_trades.csv'}")
    
    # Run analyses
    analyses = {}
    
    analyses['edge'] = analyze_edge(df, output_dir)
    analyses['spread'] = analyze_market_spread(df, output_dir)
    analyses['time'] = analyze_time_remaining(df, output_dir)
    analyses['liquidity'] = analyze_liquidity(fills_df, output_dir)
    analyses['spread_line'] = analyze_spread_lines(df, output_dir)
    analyses['positions'] = analyze_positions(df, settled_df, output_dir)
    
    # Generate summary
    generate_summary(date_str, df, settled_df, analyses, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ REPORT COMPLETE")
    print(f"📁 All files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
