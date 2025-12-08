#!/usr/bin/env python
"""
Analyze trading performance using Kalshi as source of truth.

Fetches realized P&L directly from Kalshi's settled positions API
and combines with database fill history for comprehensive analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import argparse
from data.kalshi import KalshiClient

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

DB_PATH = 'data/nba_data.db'


def get_day_cycle_bounds(date_str=None):
    """
    Get start and end timestamps for a trading day cycle (4am-4am).
    
    Args:
        date_str: Optional date string (YYYY-MM-DD). If None, uses most recent cycle.
        
    Returns:
        (start_timestamp_ms, end_timestamp_ms) as integers
    """
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        now = datetime.now()
        if now.hour < 4:
            target_date = now - timedelta(days=1)
        else:
            target_date = now
    
    start = target_date.replace(hour=4, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def fetch_kalshi_settled_positions(kalshi_client, start_ts=None, end_ts=None):
    """
    Fetch settled positions from Kalshi for P&L calculation.
    
    Args:
        kalshi_client: KalshiClient instance
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        
    Returns:
        DataFrame with settled positions and P&L
    """
    print("\n📊 Fetching settled positions from Kalshi...")
    
    try:
        response = kalshi_client.get_positions(
            settlement_status='settled',
            limit=1000
        )
        
        positions = response.get('market_positions', [])
        
        if not positions:
            print("  No settled positions found \n")
            return pd.DataFrame()
        
        # Parse into DataFrame
        data = []
        for pos in positions:
            # Get timestamp in ISO format
            last_updated = pos.get('last_updated_ts')
            #print(pos)
            
            # Parse timestamp for filtering
            if last_updated:
                settled_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                settled_ts_ms = int(settled_dt.timestamp() * 1000)
            else:
                settled_ts_ms = None
            
            # Filter by time if specified
            if start_ts and end_ts and settled_ts_ms:
                if not (start_ts <= settled_ts_ms < end_ts):
                    continue
            
            data.append({
                'ticker': pos.get('ticker'),
                'position': pos.get('position', 0),
                'total_cost': pos.get('total_cost', 0) / 100.0,  # Convert to dollars
                'realized_pnl': pos.get('realized_pnl', 0) / 100.0,  # Convert to dollars
                'settled_time': settled_dt if last_updated else None
            })
        
        df = pd.DataFrame(data)
        print(f"  ✓ Found {len(df)} settled positions")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching from Kalshi: {e}")
        return pd.DataFrame()


def load_fills_from_db(start_time=None, end_time=None):
    """
    Load fill history from database.
    
    Args:
        start_time: ISO timestamp string
        end_time: ISO timestamp string
        
    Returns:
        DataFrame with fill history
    """
    print("\n📥 Loading fills from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Build WHERE clause
    where_parts = ["fill_price IS NOT NULL"]
    if start_time and end_time:
        where_parts.append(f"timestamp >= '{start_time}' AND timestamp < '{end_time}'")
    
    where_clause = "WHERE " + " AND ".join(where_parts)
    
    fills = pd.read_sql_query(f"""
        SELECT ticker, side, fill_price, size, timestamp,
               datetime(timestamp) as dt
        FROM trades
        {where_clause}
        ORDER BY timestamp
    """, conn)
    
    conn.close()
    
    print(f"  ✓ Loaded {len(fills)} fills")
    
    return fills


def analyze_pnl(settled_df):
    """Analyze P&L from Kalshi settled positions."""
    print("\n" + "=" * 60)
    print("💰 P&L ANALYSIS (From Kalshi)")
    print("=" * 60)
    
    if len(settled_df) == 0:
        print("  No settled positions yet!")
        return
    
    total_pnl = settled_df['realized_pnl'].sum()
    num_positions = len(settled_df)
    wins = len(settled_df[settled_df['realized_pnl'] > 0])
    losses = len(settled_df[settled_df['realized_pnl'] < 0])
    breakeven = len(settled_df[settled_df['realized_pnl'] == 0])
    win_rate = wins / num_positions if num_positions > 0 else 0
    
    avg_win = settled_df[settled_df['realized_pnl'] > 0]['realized_pnl'].mean() if wins > 0 else 0
    avg_loss = settled_df[settled_df['realized_pnl'] < 0]['realized_pnl'].mean() if losses > 0 else 0
    
    print(f"\n📈 Overall Performance")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Settled Positions: {num_positions}")
    print(f"  Wins: {wins} | Losses: {losses} | Break-even: {breakeven}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    if avg_loss != 0:
        print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x")
    
    # Plot cumulative P&L
    settled_sorted = settled_df.sort_values('settled_time').copy()
    settled_sorted['cumulative_pnl'] = settled_sorted['realized_pnl'].cumsum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative P&L
    ax1.plot(range(len(settled_sorted)), settled_sorted['cumulative_pnl'], 
             linewidth=2, color='green' if total_pnl > 0 else 'red')
    ax1.fill_between(range(len(settled_sorted)), 0, settled_sorted['cumulative_pnl'], 
                      alpha=0.3, color='green' if total_pnl > 0 else 'red')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Cumulative P&L Over Time (Kalshi Data)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position Number')
    ax1.set_ylabel('Cumulative P&L ($)')
    ax1.grid(True, alpha=0.3)
    
    # Individual position P&L
    colors = ['green' if x > 0 else 'red' for x in settled_sorted['realized_pnl']]
    ax2.bar(range(len(settled_sorted)), settled_sorted['realized_pnl'], color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Individual Position P&L', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Position Number')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/kalshi_pnl.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/kalshi_pnl.png")


def analyze_fills(fills_df):
    """Analyze fill patterns from database."""
    print("\n" + "=" * 60)
    print("📦 FILL ANALYSIS (From Database)")
    print("=" * 60)
    
    if len(fills_df) == 0:
        print("  No fills yet!")
        return
    
    # Extract game from ticker
    fills_df['game'] = fills_df['ticker'].str.extract(r'25DEC\d{2}([A-Z]+)')
    
    # Side analysis
    side_counts = fills_df['side'].value_counts()
    print(f"\n📊 Fill Sides:")
    print(f"  Total Fills: {len(fills_df)}")
    for side, count in side_counts.items():
        print(f"  {side.upper()}: {count} ({count/len(fills_df):.1%})")
    
    # Game analysis
    game_counts = fills_df['game'].value_counts().head(10)
    print(f"\n📊 Top Games (by fill count):")
    for game, count in game_counts.items():
        print(f"  {game}: {count} fills")
    
    # Price distribution
    print(f"\n📊 Fill Price Distribution:")
    print(f"  Mean: {fills_df['fill_price'].mean():.1f}¢")
    print(f"  Median: {fills_df['fill_price'].median():.1f}¢")
    print(f"  Std: {fills_df['fill_price'].std():.1f}¢")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Side distribution
    side_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Buy vs Sell Fills', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Side')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Game distribution
    game_counts.plot(kind='barh', ax=axes[0, 1], color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Top 10 Games by Fill Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Count')
    
    # Fill price distribution
    axes[1, 0].hist(fills_df['fill_price'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=fills_df['fill_price'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {fills_df["fill_price"].mean():.1f}¢')
    axes[1, 0].set_title('Fill Price Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Price (cents)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Size distribution
    axes[1, 1].hist(fills_df['size'], bins=range(1, fills_df['size'].max()+2), 
                    alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Position Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Size (contracts)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data/kalshi_fills.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/kalshi_fills.png")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze trading performance using Kalshi as source of truth')
    parser.add_argument('--date', '-d', type=str, help='Date to analyze (YYYY-MM-DD), defaults to most recent 4am-4am cycle')
    parser.add_argument('--key-path', type=str, default='key.key', help='Path to private key file')
    args = parser.parse_args()
    
    # Get API key
    key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"  # Your hardcoded key
    
    # Get time bounds
    start_ts_ms, end_ts_ms = get_day_cycle_bounds(args.date)
    start_iso = datetime.fromtimestamp(start_ts_ms / 1000).isoformat()
    end_iso = datetime.fromtimestamp(end_ts_ms / 1000).isoformat()
    
    print("\n" + "=" * 60)
    print("🔬 KALSHI-BASED PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📅 Trading Day Cycle (4am-4am):")
    print(f"  Start: {start_iso}")
    print(f"  End:   {end_iso}")
    
    # Initialize Kalshi client
    print("\n🔌 Connecting to Kalshi...")
    kalshi = KalshiClient(key_id, args.key_path)
    
    # Fetch data
    settled_df = fetch_kalshi_settled_positions(kalshi, start_ts_ms, end_ts_ms)
    fills_df = load_fills_from_db(start_iso, end_iso)
    
    # Run analyses
    if len(settled_df) > 0:
        analyze_pnl(settled_df)
    else:
        print("\n⚠️  No settled positions in this time range")
    
    if len(fills_df) > 0:
        analyze_fills(fills_df)
    else:
        print("\n⚠️  No fills in this time range")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"\n💰 Kalshi Settled Positions: {len(settled_df)}")
    if len(settled_df) > 0:
        print(f"  Total Realized P&L: ${settled_df['realized_pnl'].sum():+.2f}")
    
    print(f"\n📊 Database Fills: {len(fills_df)}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/kalshi_pnl.png")
    print("  - data/kalshi_fills.png")
    print("\n")


if __name__ == "__main__":
    main()
