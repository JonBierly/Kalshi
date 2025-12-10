#!/usr/bin/env python
"""
Analyze all fills from Kalshi to understand intraday trading patterns.

Uses Kalshi's get_fills() API to analyze:
- Individual fill patterns
- Round-trip holding times
- Intraday P&L
- Fill rates and execution quality
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import argparse
from data.kalshi import KalshiClient

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 10)

API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"


def get_day_bounds(date_str=None):
    """
    Get Unix timestamp bounds for a trading day (4am-4am).
    
    Args:
        date_str: Optional date string (YYYY-MM-DD)
        
    Returns:
        (start_ts, end_ts) in Unix milliseconds
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
    
    # Convert to Unix milliseconds
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    
    return start_ts, end_ts, start, end


def fetch_fills(kalshi_client, start_ts, end_ts):
    """Fetch all fills from Kalshi for time range."""
    print("\n📦 Fetching fills from Kalshi...")
    
    fills = kalshi_client.get_fills(min_ts=start_ts, max_ts=end_ts, limit=1000)
    
    if not fills:
        print("  No fills found")
        return pd.DataFrame()
    
    # DEBUG: Show first fill to see structure
    if len(fills) > 0:
        print(f"\n  🔍 DEBUG - First fill structure:")
        import json
        print(f"  {json.dumps(fills[0], indent=4)}")
    
    # Parse into DataFrame
    data = []
    for fill in fills:
        # Parse created_time
        created_time = fill.get('created_time')
        if created_time:
            fill_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            # Convert to local timezone (Eastern Time)
            import pytz
            eastern = pytz.timezone('America/New_York')
            fill_dt = fill_dt.astimezone(eastern)
        else:
            continue
        
        data.append({
            'ticker': fill.get('ticker'),
            'order_id': fill.get('order_id'),
            'side': fill.get('side'),  # 'yes' or 'no'
            'action': fill.get('action'),  # 'buy' or 'sell'
            'count': fill.get('count', 0),
            'yes_price': fill.get('yes_price', 0),
            'no_price': fill.get('no_price', 0),
            'created_time': fill_dt,
            'trade_id': fill.get('trade_id')
        })
    
    df = pd.DataFrame(data)
    
    # Calculate fill price (use yes_price or no_price depending on side)
    df['fill_price'] = df.apply(
        lambda row: row['yes_price'] if row['side'] == 'yes' else row['no_price'],
        axis=1
    )
    
    # Sort by time
    df = df.sort_values('created_time').reset_index(drop=True)
    
    print(f"  ✓ Loaded {len(df)} fills")
    
    return df


def analyze_basic_stats(fills_df):
    """Analyze basic fill statistics."""
    print("\n" + "=" * 60)
    print("📊 BASIC FILL STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal Fills: {len(fills_df)}")
    print(f"Unique Tickers: {fills_df['ticker'].nunique()}")
    print(f"Unique Orders: {fills_df['order_id'].nunique()}")
    
    # Side breakdown
    print(f"\n📊 Fill Breakdown:")
    action_counts = fills_df['action'].value_counts()
    for action, count in action_counts.items():
        pct = count / len(fills_df) * 100
        print(f"  {action.upper()}: {count} ({pct:.1f}%)")
    
    # Average fill price
    print(f"\n💰 Fill Prices:")
    print(f"  Average: {fills_df['fill_price'].mean():.1f}¢")
    print(f"  Median: {fills_df['fill_price'].median():.1f}¢")
    print(f"  Std Dev: {fills_df['fill_price'].std():.1f}¢")
    
    # Size distribution
    print(f"\n📦 Fill Sizes:")
    print(f"  Average: {fills_df['count'].mean():.1f} contracts")
    print(f"  Median: {fills_df['count'].median():.0f} contracts")
    print(f"  Total Volume: {fills_df['count'].sum()} contracts")


def calculate_round_trips(fills_df):
    """
    Calculate round trips by normalizing everything to 'Net YES Exposure'.
    
    Kalshi Netting Logic:
    - Buy YES  -> Long YES (+1)
    - Sell NO  -> Long YES (+1) [Equivalent to Buy YES @ 100-price]
    - Sell YES -> Short YES (-1)
    - Buy NO   -> Short YES (-1) [Equivalent to Sell YES @ 100-price]
    """
    print("\n⏱️  Calculating round trips (using Net YES Exposure)...")
    
    round_trips = []
    
    for ticker in fills_df['ticker'].unique():
        ticker_fills = fills_df[fills_df['ticker'] == ticker].sort_values('created_time').copy()
        
        # Stack of open exposure [time, effective_price, original_fill]
        # We only track the current direction of exposure
        inventory = [] 
        current_direction = 0 # 1 for Long YES, -1 for Short YES (Long NO)
        
        for idx, fill in ticker_fills.iterrows():
            # 1. Normalize trade to Net YES terms
            fill_price = fill['fill_price']
            
            if fill['action'] == 'buy':
                if fill['side'] == 'yes':
                    # Buy YES -> Long YES
                    fill_eff_side = 1
                    fill_eff_price = fill_price
                else: # side == 'no'
                    # Buy NO -> Short YES (Selling YES @ 100-p)
                    fill_eff_side = -1
                    fill_eff_price = 100.0 - fill_price
            else: # sell
                if fill['side'] == 'yes':
                    # Sell YES -> Short YES
                    fill_eff_side = -1
                    fill_eff_price = fill_price
                else: # side == 'no'
                    # Sell NO -> Long YES (Buying YES @ 100-p)
                    fill_eff_side = 1
                    fill_eff_price = 100.0 - fill_price
            
            fill_contracts = fill['count']
            
            # 2. Match against inventory
            remaining_contracts = fill_contracts
            
            while remaining_contracts > 0:
                # Case A: Increasing position or opening new direction
                if len(inventory) == 0 or (current_direction == fill_eff_side):
                    inventory.append({
                        'time': fill['created_time'],
                        'price': fill_eff_price,
                        'count': remaining_contracts,
                        'raw_fill': fill
                    })
                    current_direction = fill_eff_side
                    remaining_contracts = 0
                
                # Case B: Closing/Reducing position
                else:
                    # Match against oldest in inventory (FIFO)
                    open_pos = inventory[0]
                    match_qty = min(remaining_contracts, open_pos['count'])
                    
                    # Calculate P&L
                    # If we were Long YES (dir=1), we are now Selling (price - cost)
                    # If we were Short YES (dir=-1), we are now Buying (cost - price)
                    # Simplified: (Exit Price - Entry Price) * Direction
                    
                    if current_direction == 1: # Closing Long YES
                        pnl_per_contract = (fill_eff_price - open_pos['price']) / 100.0
                    else: # Closing Short YES
                        pnl_per_contract = (open_pos['price'] - fill_eff_price) / 100.0
                        
                    total_pnl = pnl_per_contract * match_qty
                    holding_time = (fill['created_time'] - open_pos['time']).total_seconds() / 60
                    
                    round_trips.append({
                        'ticker': ticker,
                        'side': 'Long YES' if current_direction == 1 else 'Short YES',
                        'buy_time': open_pos['time'],
                        'sell_time': fill['created_time'],
                        'holding_minutes': holding_time,
                        'buy_price': open_pos['price'], # Entry price (normalized)
                        'sell_price': fill_eff_price,   # Exit price (normalized)
                        'size': match_qty,
                        'pnl': total_pnl
                    })
                    
                    remaining_contracts -= match_qty
                    open_pos['count'] -= match_qty
                    
                    if open_pos['count'] <= 0:
                        inventory.pop(0)
                        
                    # If inventory empty, direction is reset
                    if len(inventory) == 0:
                        current_direction = 0
                        
    rt_df = pd.DataFrame(round_trips)
    
    if len(rt_df) > 0:
        print(f"  ✓ Found {len(rt_df)} round trips")
        print(f"  Total P&L: ${rt_df['pnl'].sum():+.2f}")
        print(f"  Average P&L per round trip: ${rt_df['pnl'].mean():+.2f}")
        
        # Recalculate basic stats
        rt_df['cumulative_pnl'] = rt_df['pnl'].cumsum()
    else:
        print("  No round trips found")
    
    return rt_df


def analyze_round_trips(rt_df):
    """Analyze round trip patterns."""
    if len(rt_df) == 0:
        print("\n⚠️  No round trips to analyze")
        return
    
    print("\n" + "=" * 60)
    print("🔄 ROUND TRIP ANALYSIS")
    print("=" * 60)
    
    # Holding time stats
    print(f"\n⏱️  Holding Times:")
    print(f"  Average: {rt_df['holding_minutes'].mean():.1f} minutes ({rt_df['holding_minutes'].mean()/60:.2f} hours)")
    print(f"  Median: {rt_df['holding_minutes'].median():.1f} minutes")
    print(f"  Min: {rt_df['holding_minutes'].min():.1f} minutes")
    print(f"  Max: {rt_df['holding_minutes'].max():.1f} minutes")
    
    # P&L stats
    print(f"\n💰 P&L Statistics:")
    wins = rt_df[rt_df['pnl'] > 0]
    losses = rt_df[rt_df['pnl'] < 0]
    breakeven = rt_df[rt_df['pnl'] == 0]
    
    print(f"  Wins: {len(wins)} | Losses: {len(losses)} | Break-even: {len(breakeven)}")
    print(f"  Win Rate: {len(wins)/len(rt_df):.1%}")
    
    if len(wins) > 0:
        print(f"  Avg Win: ${wins['pnl'].mean():+.2f}")
    if len(losses) > 0:
        print(f"  Avg Loss: ${losses['pnl'].mean():+.2f}")
    if len(wins) > 0 and len(losses) > 0:
        print(f"  Win/Loss Ratio: {abs(wins['pnl'].mean() / losses['pnl'].mean()):.2f}x")
    
    # Time buckets
    rt_df['time_bucket'] = pd.cut(
        rt_df['holding_minutes'],
        bins=[0, 5, 15, 30, 60, 120, float('inf')],
        labels=['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']
    )
    
    print(f"\n📊 P&L by Holding Time:")
    for bucket in ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']:
        bucket_data = rt_df[rt_df['time_bucket'] == bucket]
        if len(bucket_data) > 0:
            total_pnl = bucket_data['pnl'].sum()
            avg_pnl = bucket_data['pnl'].mean()
            win_rate = len(bucket_data[bucket_data['pnl'] > 0]) / len(bucket_data)
            print(f"  {bucket:8s}: {len(bucket_data):3d} trades | Total: ${total_pnl:+7.2f} | Avg: ${avg_pnl:+6.2f} | WR: {win_rate:.1%}")


def plot_fill_analysis(fills_df, rt_df):
    """Create visualizations."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Fill timeline
    ax1 = fig.add_subplot(gs[0, :])
    fills_df['hour'] = fills_df['created_time'].dt.hour
    fills_by_hour = fills_df.groupby('hour').size()
    ax1.bar(fills_by_hour.index, fills_by_hour.values, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Fills')
    ax1.set_title('Fill Distribution by Hour', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Fill price distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(fills_df['fill_price'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(fills_df['fill_price'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {fills_df["fill_price"].mean():.1f}¢')
    ax2.set_xlabel('Fill Price (¢)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Fill Price Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Buy vs Sell sizes
    ax3 = fig.add_subplot(gs[1, 1])
    action_sizes = fills_df.groupby('action')['count'].sum()
    colors = ['green' if action == 'buy' else 'red' for action in action_sizes.index]
    ax3.bar(action_sizes.index, action_sizes.values, alpha=0.7, color=colors, edgecolor='black')
    ax3.set_ylabel('Total Contracts')
    ax3.set_title('Total Volume by Action', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    if len(rt_df) > 0:
        # 4. Round trip holding times
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(rt_df['holding_minutes'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(rt_df['holding_minutes'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {rt_df["holding_minutes"].mean():.1f}m')
        ax4.set_xlabel('Holding Time (minutes)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Round Trip Holding Times', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Round trip P&L scatter
        ax5 = fig.add_subplot(gs[2, 0])
        colors = rt_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red')
        ax5.scatter(rt_df['holding_minutes'], rt_df['pnl'], c=colors, alpha=0.6, edgecolors='black')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Holding Time (minutes)')
        ax5.set_ylabel('P&L ($)')
        ax5.set_title('Holding Time vs P&L', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative P&L
        ax6 = fig.add_subplot(gs[2, 1:])
        rt_sorted = rt_df.sort_values('sell_time')
        rt_sorted['cumulative_pnl'] = rt_sorted['pnl'].cumsum()
        ax6.plot(range(len(rt_sorted)), rt_sorted['cumulative_pnl'], linewidth=2, color='blue')
        ax6.fill_between(range(len(rt_sorted)), 0, rt_sorted['cumulative_pnl'], alpha=0.3,
                         color='green' if rt_sorted['cumulative_pnl'].iloc[-1] > 0 else 'red')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Round Trip Number')
        ax6.set_ylabel('Cumulative P&L ($)')
        ax6.set_title(f'Cumulative P&L Over Time (Final: ${rt_sorted["cumulative_pnl"].iloc[-1]:+.2f})',
                     fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.savefig('data/fills_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/fills_analysis.png")


def analyze_held_to_settlement(fills_df, kalshi_client, start_ts, end_ts):
    """Analyze positions held to settlement (not sold intraday)."""
    print("\n" + "=" * 60)
    print("📊 HELD-TO-SETTLEMENT ANALYSIS")
    print("=" * 60)
    print("(Positions you didn't sell - held until market closed)")
    
    try:
        # Get settled positions from Kalshi
        settled_positions = kalshi_client.get_positions(settlement_status='settled', limit=1000)
        
        # First, calculate raw total from Kalshi for this day
        raw_total = 0.0
        raw_count = 0
        for pos in settled_positions['market_positions']:
            last_updated = pos.get('last_updated_ts')
            if last_updated:
                settled_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                settled_ts = int(settled_dt.timestamp() * 1000)
                
                if start_ts <= settled_ts < end_ts:
                    raw_total += pos.get('realized_pnl', 0) / 100.0
                    raw_count += 1
        
        print(f"\n  🔍 DEBUG - Kalshi Raw Data:")
        print(f"     Total settled positions: {raw_count}")
        print(f"     Total realized P&L: ${raw_total:+.2f}")
        print(f"     (This is what Kalshi directly reports)")
        
        # Filter positions that settled today
        settled_data = []
        for pos in settled_positions['market_positions']:
            last_updated = pos.get('last_updated_ts')
            if last_updated:
                settled_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                settled_ts = int(settled_dt.timestamp() * 1000)
                
                if start_ts <= settled_ts < end_ts:
                    settled_data.append({
                        'ticker': pos.get('ticker'),
                        'realized_pnl': pos.get('realized_pnl', 0) / 100.0,
                        'settled_time': settled_dt,
                        'position': pos.get('position', 0),
                        'total_cost': pos.get('total_cost', 0) / 100.0
                    })
        
        if not settled_data:
            print("\n  No positions settled today")
            return pd.DataFrame(), 0.0
        
        settled_df = pd.DataFrame(settled_data)
        
        # Match with fills to get buy times and calc holding times
        for idx, row in settled_df.iterrows():
            ticker = row['ticker']
            ticker_fills = fills_df[fills_df['ticker'] == ticker]
            
            if len(ticker_fills) > 0:
                # First buy for this ticker
                buy_fills = ticker_fills[ticker_fills['action'] == 'buy']
                if len(buy_fills) > 0:
                    first_buy = buy_fills.iloc[0]['created_time']
                    import pytz
                    eastern = pytz.timezone('America/New_York')
                    settled_time_local = row['settled_time'].astimezone(eastern)
                    
                    holding_time = (settled_time_local - first_buy).total_seconds() / 60
                    settled_df.at[idx, 'holding_minutes'] = holding_time
                    settled_df.at[idx, 'buy_time'] = first_buy
        
        # Stats
        total_pnl = settled_df['realized_pnl'].sum()
        wins = settled_df[settled_df['realized_pnl'] > 0]
        losses = settled_df[settled_df['realized_pnl'] < 0]
        breakeven = settled_df[settled_df['realized_pnl'] == 0]
        
        print(f"\n  Settled Positions: {len(settled_df)}")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        print(f"  Wins: {len(wins)} | Losses: {len(losses)} | Break-even: {len(breakeven)}")
        
        if len(settled_df) > 0:
            print(f"  Win Rate: {len(wins)/len(settled_df):.1%}")
        
        if len(wins) > 0:
            print(f"  Avg Win: ${wins['realized_pnl'].mean():+.2f}")
        if len(losses) > 0:
            print(f"  Avg Loss: ${losses['realized_pnl'].mean():+.2f}")
        
        # Show holding times if available
        if 'holding_minutes' in settled_df.columns:
            valid_holdings = settled_df[settled_df['holding_minutes'].notna()]
            if len(valid_holdings) > 0:
                print(f"\n  Avg Holding Time: {valid_holdings['holding_minutes'].mean():.1f} minutes")
        
        # Show top losers
        if len(losses) > 0:
            print(f"\n  Top Losers:")
            top_losers = settled_df.nsmallest(3, 'realized_pnl')[['ticker', 'realized_pnl']]
            for idx, row in top_losers.iterrows():
                print(f"    {row['ticker'][-20:]}: ${row['realized_pnl']:+.2f}")
        
        return settled_df, total_pnl
        
    except Exception as e:
        print(f"  Error fetching settled positions: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0.0


def plot_settlement_analysis(settled_df):
    """Create visualization for held-to-settlement positions."""
    if len(settled_df) == 0:
        print("  No settled positions to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Held-to-Settlement Analysis (Didn\'t Sell Intraday)', fontsize=16, fontweight='bold')
    
    # 1. P&L distribution
    ax1 = axes[0, 0]
    colors = settled_df['realized_pnl'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray'))
    ax1.bar(range(len(settled_df)), settled_df['realized_pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Position Number')
    ax1.set_ylabel('P&L ($)')
    ax1.set_title('Individual Position P&L', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative P&L
    ax2 = axes[0, 1]
    cumulative = settled_df['realized_pnl'].cumsum()
    ax2.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
    ax2.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3,
                     color='green' if cumulative.iloc[-1] > 0 else 'red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position Number')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.set_title(f'Cumulative P&L (Final: ${cumulative.iloc[-1]:+.2f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss distribution
    ax3 = axes[1, 0]
    outcome_counts = [
        len(settled_df[settled_df['realized_pnl'] > 0]),
        len(settled_df[settled_df['realized_pnl'] < 0]),
        len(settled_df[settled_df['realized_pnl'] == 0])
    ]
    ax3.bar(['Wins', 'Losses', 'Break-even'], outcome_counts, 
           color=['green', 'red', 'gray'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Win/Loss Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Holding time distribution (if available)
    ax4 = axes[1, 1]
    if 'holding_minutes' in settled_df.columns:
        valid_holdings = settled_df[settled_df['holding_minutes'].notna()]
        if len(valid_holdings) > 0:
            ax4.hist(valid_holdings['holding_minutes'], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(valid_holdings['holding_minutes'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {valid_holdings["holding_minutes"].mean():.1f}m')
            ax4.set_xlabel('Holding Time (minutes)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Holding Time Distribution', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No holding time data', ha='center', va='center')
            ax4.set_title('Holding Time Distribution', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No holding time data', ha='center', va='center')
        ax4.set_title('Holding Time Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/settlement_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/settlement_analysis.png")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze fills from Kalshi')
    parser.add_argument('--date', '-d', type=str,
                       help='Date to analyze (YYYY-MM-DD), defaults to most recent 4am-4am cycle')
    args = parser.parse_args()
    
    # Get time bounds
    start_ts, end_ts, start_dt, end_dt = get_day_bounds(args.date)
    
    print("\n" + "=" * 60)
    print("🔬 KALSHI FILLS ANALYSIS")
    print("=" * 60)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📅 Trading Day (4am-4am):")
    print(f"  Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to Kalshi
    print("\n🔌 Connecting to Kalshi...")
    kalshi = KalshiClient(API_KEY, KEY_PATH)
    
    # Fetch fills
    fills_df = fetch_fills(kalshi, start_ts, end_ts)
    
    if len(fills_df) == 0:
        print("\n⚠️  No fills found for this time range")
        return
    
    # Analyze basic stats
    analyze_basic_stats(fills_df)
    
    # Analyze intraday round trips
    rt_df = calculate_round_trips(fills_df)
    analyze_round_trips(rt_df)
    
    # Analyze held-to-settlement positions
    settled_df, settlement_pnl = analyze_held_to_settlement(fills_df, kalshi, start_ts, end_ts)
    
    # Summary
    print("\n" + "=" * 60)
    print("💰 TOTAL P&L BREAKDOWN")
    print("=" * 60)
    
    round_trip_pnl = rt_df['pnl'].sum() if len(rt_df) > 0 else 0.0
    total_pnl = round_trip_pnl + settlement_pnl
    
    print(f"\n  Intraday Round Trips:  ${round_trip_pnl:+.2f}  ({len(rt_df)} trades)")
    print(f"  Held-to-Settlement:    ${settlement_pnl:+.2f}  ({len(settled_df)} positions)")
    print(f"  " + "-" * 45)
    print(f"  TOTAL:                 ${total_pnl:+.2f}")
    print(f"\n  Note: Does not include trading fees (~0.7% per trade)")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_fill_analysis(fills_df, rt_df)
    if len(settled_df) > 0:
        plot_settlement_analysis(settled_df)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/fills_analysis.png (intraday trading)")
    if len(settled_df) > 0:
        print("  - data/settlement_analysis.png (held-to-settlement)")
    print()


if __name__ == "__main__":
    main()
