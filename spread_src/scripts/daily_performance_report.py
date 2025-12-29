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
import pytz

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
    """Load trades from local database for a specific game date."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load all trades, we'll filter by game_date after parsing
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
    
    if df.empty:
        return df
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract game date from game_id (e.g., 25DEC12BOS -> 2025-12-12)
    df['game_date'] = df['game_id'].apply(extract_game_date)
    
    # Filter by game date
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    df = df[df['game_date'] == target_date]
    
    if df.empty:
        return df
    
    # Derived fields
    # Edge = difference between model fair value and fill price (in cents)
    df['edge'] = abs(df['model_fair_value'] - df['fill_price'])
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


def extract_game_date(game_id):
    """Extract actual game date from game_id (e.g., 25DEC11BOS -> 2025-12-11)."""
    if not game_id:
        return None
    try:
        # game_id format: 25DEC11BOS (YYMMMDD + team code)
        date_part = game_id[:7]
        year = 2000 + int(date_part[:2])  # 25 -> 2025
        month_str = date_part[2:5]  # DEC
        day = int(date_part[5:7])  # 11
        
        months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                  'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        month = months.get(month_str.upper(), 1)
        
        return datetime(year, month, day).date()
    except:
        return None


def get_game_day_time_range(date_str):
    """
    Get timestamp range for a game day.
    
    NBA games for a given date run from ~afternoon to late night,
    with West Coast games potentially settling after midnight.
    
    Time window: 10am target date to 4am next date (Eastern time)
    """
    eastern = pytz.timezone('US/Eastern')
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Start: 10am Eastern on target date
    start_dt = eastern.localize(datetime(target_date.year, target_date.month, target_date.day, 10, 0, 0))
    
    # End: 4am Eastern on next day
    next_date = target_date + timedelta(days=1)
    end_dt = eastern.localize(datetime(next_date.year, next_date.month, next_date.day, 4, 0, 0))
    
    # Convert to UTC timestamps in milliseconds
    min_ts = int(start_dt.astimezone(pytz.utc).timestamp() * 1000)
    max_ts = int(end_dt.astimezone(pytz.utc).timestamp() * 1000)
    
    return min_ts, max_ts


def load_fills_from_api(client, date_str):
    """Load fills from Kalshi API for a specific game day."""
    min_ts, max_ts = get_game_day_time_range(date_str)
    
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
        
        side = fill.get('side')
        yes_price = fill.get('yes_price', 0)
        no_price = fill.get('no_price', 0)
        price = yes_price if side == 'yes' else no_price
            
        fill_data.append({
            'fill_time': dt,
            'ticker': fill.get('ticker'),
            'side': side,
            'action': fill.get('action'),
            'count': fill.get('count', 0),
            'yes_price': yes_price,
            'no_price': no_price,
            'price': price,
            'is_taker': fill.get('is_taker', False),
        })
    
    return pd.DataFrame(fill_data)


def load_settled_positions(client, date_str):
    """Load settlements for markets from the target game day."""
    min_ts, max_ts = get_game_day_time_range(date_str)
    
    # Fetch settlements from Kalshi API
    settlements = client.get_settlements(min_ts=min_ts, max_ts=max_ts, limit=200)
    
    # Filter to NBA spread markets and calculate CORRECT P&L
    result = []
    for s in settlements:
        ticker = s.get('ticker', '')
        if 'KXNBASPREAD' in ticker:
            yes_count = s.get('yes_count', 0)
            no_count = s.get('no_count', 0)
            yes_cost = s.get('yes_total_cost', 0)  # in cents
            no_cost = s.get('no_total_cost', 0)    # in cents
            market_result = s.get('market_result', '')
            fees = float(s.get('fee_cost', '0'))
            
            # CORRECT P&L FORMULA:
            # Payout = winning_side_count × 100 cents
            # P&L = (payout - total_costs) / 100 - fees
            if market_result == 'yes':
                payout_cents = yes_count * 100
            else:  # 'no'
                payout_cents = no_count * 100
            
            pnl_cents = payout_cents - yes_cost - no_cost
            pnl_dollars = pnl_cents / 100.0 - fees
            
            result.append({
                'ticker': ticker,
                'event_ticker': s.get('event_ticker', ''),
                'market_result': market_result,
                'yes_count': yes_count,
                'no_count': no_count,
                'yes_cost': yes_cost / 100.0,
                'no_cost': no_cost / 100.0,
                'payout': payout_cents / 100.0,
                'realized_pnl': pnl_dollars,
                'fees_paid': fees,
                'settled_time': s.get('settled_time', ''),
            })
    
    return pd.DataFrame(result)


# ============================================================================
# UNIFIED P&L CALCULATION
# ============================================================================

def calculate_true_pnl(settled_df):
    """
    Calculate true P&L using settlement data as the SINGLE SOURCE OF TRUTH.
    
    The settlement API provides complete P&L for every ticker you traded,
    including positions you fully closed out (they show yes_count=0, no_count=0
    but still have cost data that reflects your trading activity).
    
    P&L formula (already calculated in load_settled_positions):
        payout = winning_side_count × 100 cents
        pnl = (payout - yes_cost - no_cost) / 100 - fees
    
    Returns:
        dict with:
        - 'ticker_pnl': {ticker: pnl} dict
        - 'total_pnl': float (sum of all ticker P&L)
        - 'total_fees': float (sum of all fees)
    """
    if settled_df.empty:
        return {'ticker_pnl': {}, 'total_pnl': 0.0, 'total_fees': 0.0}
    
    # Build ticker P&L lookup from settlement data
    ticker_pnl = {}
    total_fees = 0.0
    
    for _, row in settled_df.iterrows():
        ticker = row['ticker']
        pnl = row['realized_pnl']  # Already correct from load_settled_positions
        fees = row['fees_paid']
        
        ticker_pnl[ticker] = pnl
        total_fees += fees
    
    total_pnl = sum(ticker_pnl.values())
    
    return {
        'ticker_pnl': ticker_pnl,
        'total_pnl': total_pnl,
        'total_fees': total_fees
    }


# ============================================================================
# DATA ENRICHMENT FUNCTIONS
# ============================================================================

def distribute_pnl_to_trades(trades_df, ticker_pnl):
    """
    Distribute ticker-level P&L to individual trades proportionally.
    
    Args:
        trades_df: DataFrame of trades from database
        ticker_pnl: Dict of {ticker: realized_pnl} from calculate_true_pnl()
    
    Returns:
        trades_df with realized_pnl column populated
    """
    if trades_df.empty or not ticker_pnl:
        return trades_df
    
    # Get filled trades only
    filled = trades_df[trades_df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    # For each ticker, distribute P&L to trades proportionally by trade size
    for ticker, pnl in ticker_pnl.items():
        mask = filled['ticker'] == ticker
        ticker_trades = filled[mask]
        
        if len(ticker_trades) == 0:
            continue
        
        # Distribute P&L proportionally by trade size
        total_size = ticker_trades['size'].sum()
        if total_size > 0:
            for idx in ticker_trades.index:
                trade_size = filled.loc[idx, 'size']
                trade_pnl = pnl * (trade_size / total_size)
                trades_df.loc[trades_df['trade_id'] == filled.loc[idx, 'trade_id'], 'realized_pnl'] = trade_pnl
    
    return trades_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_edge(df, output_dir):
    """Analyze edge demanded vs profitability."""
    print("\n📊 Edge Analysis...")
    
    # Only use filled trades with realized P&L
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
    if filled.empty:
        print("  No filled trades to analyze")
        return {}
    
    # Create edge buckets (in cents)
    filled['edge_bucket'] = pd.cut(
        filled['edge'],
        bins=[0, 5, 10, 15, 1000],
        labels=['Small (0-5¢)', 'Medium (5-10¢)', 'Large (10-15¢)', 'Huge (>15¢)']
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
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
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
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
    
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
    
    filled = df[df['status'].isin(['filled', 'settled', 'closed'])].copy()
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
    """Analyze P&L by ticker using SETTLEMENT data (correct P&L source)."""
    print("\n📊 Position Analysis (from settlements)...")
    
    if settled_df.empty:
        print("  No settlement data available")
        return {}
    
    # Use settlement data which has correct P&L
    ticker_pnl = settled_df[['ticker', 'realized_pnl', 'yes_count', 'no_count']].copy()
    ticker_pnl['contracts'] = ticker_pnl['yes_count'] + ticker_pnl['no_count']
    ticker_pnl = ticker_pnl[['ticker', 'realized_pnl', 'contracts']]
    ticker_pnl = ticker_pnl.sort_values('realized_pnl')
    
    # Separate losers and winners
    losers = ticker_pnl[ticker_pnl['realized_pnl'] < 0]
    winners = ticker_pnl[ticker_pnl['realized_pnl'] > 0]
    
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
        'total_tickers': len(ticker_pnl),
        'winners': len(winners),
        'losers': len(losers),
        'total_pnl': ticker_pnl['realized_pnl'].sum(),
        'total_contracts': ticker_pnl['contracts'].sum(),
        'avg_winner': winners['realized_pnl'].mean() if not winners.empty else 0,
        'avg_loser': losers['realized_pnl'].mean() if not losers.empty else 0,
    }
    
    print(f"  Total Tickers: {summary['total_tickers']}")
    print(f"  Winners: {summary['winners']} (avg ${summary['avg_winner']:.2f})")
    print(f"  Losers: {summary['losers']} (avg ${summary['avg_loser']:.2f})")
    print(f"  Total P&L: ${summary['total_pnl']:.2f}")
    print(f"  Total Contracts: {summary['total_contracts']}")
    
    return summary


def generate_summary(date_str, df, pnl_result, settled_df, analyses, output_dir):
    """
    Generate text summary of findings.
    
    Args:
        date_str: Date string
        df: Trades DataFrame
        pnl_result: Dict from calculate_true_pnl() with total_pnl, trading_pnl, settlement_pnl
        settled_df: Settlements DataFrame (for fees)
        analyses: Dict of analysis results
        output_dir: Output directory path
    """
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
        
        filled = df[df['status'].isin(['filled', 'settled', 'closed'])]
        f.write(f"Total Fills: {len(filled)}\n")
        f.write(f"Fill Rate: {len(filled)/len(df)*100:.1f}%\n")
        
        # Use settlement-based P&L calculation (already includes fees)
        total_pnl = pnl_result.get('total_pnl', 0.0)
        total_fees = pnl_result.get('total_fees', 0.0)
        num_tickers = len(pnl_result.get('ticker_pnl', {}))
        
        f.write(f"Total P&L: ${total_pnl:.2f} (net of ${total_fees:.2f} fees)\n")
        f.write(f"Markets Traded: {num_tickers}\n")
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
    
    # Calculate TRUE P&L using settlements as single source of truth
    print("\n📊 Calculating TRUE P&L from settlements...")
    pnl_result = calculate_true_pnl(settled_df)
    print(f"  ✓ Total P&L: ${pnl_result['total_pnl']:.2f}")
    print(f"  ✓ Total Fees: ${pnl_result['total_fees']:.2f}")
    print(f"  ✓ Tickers: {len(pnl_result['ticker_pnl'])}")
    
    # Distribute P&L to individual trades for analysis charts
    df = distribute_pnl_to_trades(df, pnl_result['ticker_pnl'])
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
    
    # Generate summary with unified P&L
    generate_summary(date_str, df, pnl_result, settled_df, analyses, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ REPORT COMPLETE")
    print(f"📁 All files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
