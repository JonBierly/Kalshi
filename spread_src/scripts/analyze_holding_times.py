#!/usr/bin/env python
"""
Analyze holding times and their impact on profitability.

Calculates time between first fill and settlement for each position,
then correlates holding time with P&L to find optimal trading windows.
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
from data.kalshi import KalshiClient

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

DB_PATH = 'data/nba_data.db'
API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"


def get_yesterday_bounds():
    """Get 4am-4am bounds for yesterday."""
    now = datetime.now()
    if now.hour < 4:
        # Before 4am today = use day before yesterday
        target_date = now - timedelta(days=2)
    else:
        # After 4am = use yesterday
        target_date = now - timedelta(days=1)
    
    start = target_date.replace(hour=4, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    
    return start.isoformat(), end.isoformat(), int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def load_fills(start_iso, end_iso):
    """Load fills from database for date range."""
    conn = sqlite3.connect(DB_PATH)
    
    fills = pd.read_sql_query(f"""
        SELECT ticker, side, fill_price, size, timestamp
        FROM trades
        WHERE fill_price IS NOT NULL
        AND timestamp >= '{start_iso}'
        AND timestamp < '{end_iso}'
        ORDER BY timestamp
    """, conn)
    
    conn.close()
    
    if len(fills) > 0:
        fills['timestamp'] = pd.to_datetime(fills['timestamp']).dt.tz_localize('UTC')
    
    return fills


def load_settled_positions(kalshi_client, start_ms, end_ms):
    """Load settled positions from Kalshi."""
    response = kalshi_client.get_positions(settlement_status='settled', limit=1000)
    positions = response.get('market_positions', [])
    
    data = []
    for pos in positions:
        last_updated = pos.get('last_updated_ts')
        if last_updated:
            settled_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            settled_ms = int(settled_dt.timestamp() * 1000)
        else:
            continue
        
        # Filter by time range
        if not (start_ms <= settled_ms < end_ms):
            continue
        
        data.append({
            'ticker': pos.get('ticker'),
            'realized_pnl': pos.get('realized_pnl', 0) / 100.0,
            'settled_time': settled_dt
        })
    
    return pd.DataFrame(data)


def calculate_holding_times(fills_df, settled_df):
    """Calculate holding time for each position."""
    results = []
    
    for ticker in settled_df['ticker'].unique():
        # Get fills for this ticker
        ticker_fills = fills_df[fills_df['ticker'] == ticker].sort_values('timestamp')
        
        if len(ticker_fills) == 0:
            continue
        
        # Get settlement data
        settlement = settled_df[settled_df['ticker'] == ticker].iloc[0]
        
        # Calculate holding time
        first_fill = ticker_fills.iloc[0]['timestamp']
        settled_time = settlement['settled_time']
        holding_time = (settled_time - first_fill).total_seconds() / 60  # Minutes
        
        # Extract spread value from ticker
        ticker_str = ticker.split('-')[-1]  # e.g., "OKC26" or "PHI5"
        spread_val = ''.join(filter(str.isdigit, ticker_str))
        spread = int(spread_val) if spread_val else 0
        
        results.append({
            'ticker': ticker,
            'spread': spread,
            'first_fill': first_fill,
            'settled_time': settled_time,
            'holding_minutes': holding_time,
            'holding_hours': holding_time / 60,
            'num_fills': len(ticker_fills),
            'realized_pnl': settlement['realized_pnl']
        })
    
    return pd.DataFrame(results)


def analyze_holding_times(holdings_df):
    """Analyze holding time patterns."""
    print("\n" + "=" * 60)
    print("⏱️  HOLDING TIME ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Positions Analyzed: {len(holdings_df)}")
    print(f"  Avg Holding Time: {holdings_df['holding_minutes'].mean():.1f} min ({holdings_df['holding_hours'].mean():.2f} hrs)")
    print(f"  Median Holding Time: {holdings_df['holding_minutes'].median():.1f} min")
    print(f"  Min: {holdings_df['holding_minutes'].min():.1f} min")
    print(f"  Max: {holdings_df['holding_minutes'].max():.1f} min")
    
    # Create time buckets
    holdings_df['time_bucket'] = pd.cut(
        holdings_df['holding_minutes'],
        bins=[0, 5, 15, 30, 60, 120, float('inf')],
        labels=['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']
    )
    
    # Analyze by bucket
    print(f"\n📊 P&L by Holding Time:")
    bucket_stats = holdings_df.groupby('time_bucket').agg({
        'realized_pnl': ['count', 'sum', 'mean'],
        'ticker': 'count'
    }).round(2)
    
    for bucket in ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']:
        if bucket in bucket_stats.index:
            count = bucket_stats.loc[bucket, ('realized_pnl', 'count')]
            total_pnl = bucket_stats.loc[bucket, ('realized_pnl', 'sum')]
            avg_pnl = bucket_stats.loc[bucket, ('realized_pnl', 'mean')]
            print(f"  {bucket:8s}: {int(count):3d} trades | Total: ${total_pnl:+7.2f} | Avg: ${avg_pnl:+6.2f}")
    
    # Wins vs losses by holding time
    holdings_df['outcome'] = holdings_df['realized_pnl'].apply(
        lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Break-even')
    )
    
    print(f"\n📊 Win Rate by Holding Time:")
    for bucket in ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']:
        bucket_data = holdings_df[holdings_df['time_bucket'] == bucket]
        if len(bucket_data) > 0:
            wins = len(bucket_data[bucket_data['outcome'] == 'Win'])
            win_rate = wins / len(bucket_data)
            print(f"  {bucket:8s}: {win_rate:.1%} ({wins}/{len(bucket_data)})")
    
    # Correlation
    correlation = holdings_df['holding_minutes'].corr(holdings_df['realized_pnl'])
    print(f"\n📈 Correlation (holding time vs P&L): {correlation:.3f}")
    if abs(correlation) > 0.3:
        print(f"  {'📈 Positive' if correlation > 0 else '📉 Negative'} correlation detected!")
    else:
        print(f"  ➡️  Weak correlation (holding time doesn't strongly predict P&L)")
    
    return holdings_df


def plot_holding_time_analysis(holdings_df):
    """Create visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Histogram of holding times
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(holdings_df['holding_minutes'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(holdings_df['holding_minutes'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {holdings_df["holding_minutes"].mean():.1f}m')
    ax1.set_xlabel('Holding Time (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Holding Times', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: Holding time vs P&L
    ax2 = fig.add_subplot(gs[0, 1])
    colors = holdings_df['realized_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
    ax2.scatter(holdings_df['holding_minutes'], holdings_df['realized_pnl'], 
                c=colors, alpha=0.6, edgecolors='black')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Holding Time (minutes)')
    ax2.set_ylabel('Realized P&L ($)')
    ax2.set_title('Holding Time vs Profitability', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by time bucket
    ax3 = fig.add_subplot(gs[1, :])
    bucket_order = ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2h+']
    holdings_df_sorted = holdings_df.sort_values('time_bucket')
    sns.boxplot(data=holdings_df_sorted, x='time_bucket', y='realized_pnl', 
                order=bucket_order, ax=ax3, palette='Set2')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Holding Time Bucket')
    ax3.set_ylabel('Realized P&L ($)')
    ax3.set_title('P&L Distribution by Holding Time Bucket', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Win rate by bucket
    ax4 = fig.add_subplot(gs[2, 0])
    win_rates = []
    labels = []
    for bucket in bucket_order:
        bucket_data = holdings_df[holdings_df['time_bucket'] == bucket]
        if len(bucket_data) > 0:
            win_rate = len(bucket_data[bucket_data['realized_pnl'] > 0]) / len(bucket_data)
            win_rates.append(win_rate * 100)
            labels.append(f"{bucket}\n(n={len(bucket_data)})")
    
    bars = ax4.bar(range(len(win_rates)), win_rates, alpha=0.7, color='green', edgecolor='black')
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% Win Rate')
    ax4.set_xlabel('Holding Time Bucket')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate by Holding Time', fontweight='bold')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cumulative P&L over holding time
    ax5 = fig.add_subplot(gs[2, 1])
    holdings_sorted = holdings_df.sort_values('holding_minutes')
    holdings_sorted['cumulative_pnl'] = holdings_sorted['realized_pnl'].cumsum()
    ax5.plot(holdings_sorted['holding_minutes'], holdings_sorted['cumulative_pnl'], 
             linewidth=2, color='purple')
    ax5.fill_between(holdings_sorted['holding_minutes'], 0, holdings_sorted['cumulative_pnl'],
                      alpha=0.3, color='purple')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Holding Time (minutes)')
    ax5.set_ylabel('Cumulative P&L ($)')
    ax5.set_title('Cumulative P&L (sorted by holding time)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('data/holding_time_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/holding_time_analysis.png")


def main():
    """Main analysis function."""
    print("\n" + "=" * 60)
    print("⏱️  HOLDING TIME ANALYSIS - YESTERDAY")
    print("=" * 60)
    
    # Get yesterday's bounds
    start_iso, end_iso, start_ms, end_ms = get_yesterday_bounds()
    print(f"\n📅 Analyzing: {start_iso} to {end_iso}")
    
    # Connect to Kalshi
    print("\n🔌 Connecting to Kalshi...")
    kalshi = KalshiClient(API_KEY, KEY_PATH)
    
    # Load data
    print("📥 Loading fills from database...")
    fills_df = load_fills(start_iso, end_iso)
    print(f"  ✓ Loaded {len(fills_df)} fills")
    
    print("📊 Loading settled positions from Kalshi...")
    settled_df = load_settled_positions(kalshi, start_ms, end_ms)
    print(f"  ✓ Loaded {len(settled_df)} settled positions")
    
    if len(fills_df) == 0 or len(settled_df) == 0:
        print("\n⚠️  Insufficient data for analysis")
        return
    
    # Calculate holding times
    print("\n⏱️  Calculating holding times...")
    holdings_df = calculate_holding_times(fills_df, settled_df)
    print(f"  ✓ Analyzed {len(holdings_df)} positions")
    
    # Analyze
    holdings_df = analyze_holding_times(holdings_df)
    
    # Plot
    print("\n📊 Generating visualizations...")
    plot_holding_time_analysis(holdings_df)
    
    # Top performers
    print("\n" + "=" * 60)
    print("🏆 TOP PERFORMERS")
    print("=" * 60)
    
    print("\n💰 Most Profitable (by holding time):")
    top_profitable = holdings_df.nlargest(5, 'realized_pnl')[['ticker', 'holding_minutes', 'realized_pnl']]
    for idx, row in top_profitable.iterrows():
        print(f"  {row['ticker'][-15:]}: {row['holding_minutes']:.0f}m → ${row['realized_pnl']:+.2f}")
    
    print("\n⚡ Fastest Wins (< 30min):")
    fast_wins = holdings_df[(holdings_df['holding_minutes'] < 30) & (holdings_df['realized_pnl'] > 0)]
    if len(fast_wins) > 0:
        fast_wins_top = fast_wins.nlargest(5, 'realized_pnl')[['ticker', 'holding_minutes', 'realized_pnl']]
        for idx, row in fast_wins_top.iterrows():
            print(f"  {row['ticker'][-15:]}: {row['holding_minutes']:.0f}m → ${row['realized_pnl']:+.2f}")
    else:
        print("  None found")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
