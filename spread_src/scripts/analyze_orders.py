#!/usr/bin/env python
"""
Analyze order fill rates and execution quality.

Reads from the local SQLite database (data/nba_data.db) to analyze:
1. Overall Fill Rate
2. Fill Rate by Ticker/Game
3. Fill Rate vs Price Aggressiveness (Order Price vs Model Fair)
4. Fill Rate vs Market Spread Width
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

DB_PATH = 'data/nba_data.db'

import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Kalshi order execution quality.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()

def load_local_trades(start_date=None, end_date=None):
    """Load trades from local SQLite database."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Base query
        query = """
            SELECT 
                trade_id, timestamp, ticker, side, order_price, fill_price, size, 
                model_fair_value, market_spread, status, created_at, filled_at
            FROM trades
        """
        
        args = []
        conditions = []
        
        if start_date:
            conditions.append("DATE(created_at) >= ?")
            args.append(start_date)
        
        if end_date:
            conditions.append("DATE(created_at) <= ?")
            args.append(end_date)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY created_at DESC"
        
        df = pd.read_sql_query(query, conn, params=args)
        conn.close()
        
        if not df.empty:
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return pd.DataFrame()

def analyze_fill_rates(df):
    """Analyze fill rates and patterns."""
    print(f"\nTotal Orders Logged: {len(df)}")
    
    # 1. Overall Status Breakdown
    status_counts = df['status'].value_counts()
    print("\n📊 Order Status Breakdown:")
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"  {status.upper()}: {count} ({pct:.1f}%)")
    
    filled_orders = df[df['status'] == 'filled']
    fill_rate = len(filled_orders) / len(df) if len(df) > 0 else 0
    print(f"\n✅ Overall Fill Rate: {fill_rate:.1%}")
    
    # 2. Fill Rate by Side
    print("\n📊 Fill Rate by Side:")
    by_side = df.groupby('side')['status'].apply(lambda x: (x == 'filled').mean()).reset_index()
    by_side.columns = ['Side', 'Fill Rate']
    for _, row in by_side.iterrows():
        print(f"  {row['Side'].upper()}: {row['Fill Rate']:.1%}")
        
    # 3. Aggressiveness Analysis (Delta from Fair Value)
    df['edge_demanded'] = df.apply(
        lambda row: (row['model_fair_value'] - row['order_price']) if row['side'] == 'buy' 
        else (row['order_price'] - row['model_fair_value']),
        axis=1
    )
    
    # Bucket edge demanded
    df['edge_bucket'] = pd.cut(df['edge_demanded'], bins=[-10, 0, 2, 5, 8, 100], 
                              labels=['Negative (Reaching)', '0-2¢', '2-5¢', '5-8¢', '>8¢'])
    
    print("\n📊 Fill Rate by Edge Demanded (Model Edge):")
    edge_stats = df.groupby('edge_bucket', observed=True)['status'].apply(lambda x: (x == 'filled').mean()).reset_index()
    edge_stats.columns = ['Edge Bucket', 'Fill Rate']
    edge_counts = df['edge_bucket'].value_counts()
    
    for _, row in edge_stats.iterrows():
        count = edge_counts[row['Edge Bucket']]
        print(f"  {row['Edge Bucket']:<20}: {row['Fill Rate']:>6.1%}  (n={count})")

    # 4. Market Spread Analysis
    df['spread_bucket'] = pd.cut(df['market_spread'], bins=[0, 10, 25, 50, 100], 
                                labels=['Tight (0-10¢)', 'Normal (10-25¢)', 'Wide (25-50¢)', 'Very Wide (>50¢)'])
    
    print("\n📊 Fill Rate by Market Spread Width:")
    spread_stats = df.groupby('spread_bucket', observed=True)['status'].apply(lambda x: (x == 'filled').mean())
    spread_counts = df['spread_bucket'].value_counts()
    
    for bucket in spread_stats.index:
        fill_rate = spread_stats[bucket]
        count = spread_counts.get(bucket, 0)
        print(f"  {str(bucket):<20}: {fill_rate:>6.1%}  (n={count})")

    return df

def plot_analysis(df, filename="order_analysis_plots.png"):
    """Generate and save visualization plots."""
    if df.empty:
        return

    print(f"\n🎨 Generating plots to {filename}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Kalshi Order Execution Analysis', fontsize=20)
    
    # Plot 1: Order Status
    status_counts = df['status'].value_counts()
    axes[0, 0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    axes[0, 0].set_title('Order Status Distribution')
    
    # Plot 2: Fill Rate over Time (Daily)
    daily_stats = df.copy()
    daily_stats['date'] = daily_stats['created_at'].dt.date
    daily_fills = daily_stats.groupby('date')['status'].apply(lambda x: (x == 'filled').mean()).reset_index()
    
    sns.barplot(data=daily_fills, x='date', y='status', ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title('Daily Fill Rate')
    axes[0, 1].set_ylabel('Fill Rate')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Fill Rate by Edge Bucket
    edge_stats = df.groupby('edge_bucket', observed=True)['status'].apply(lambda x: (x == 'filled').mean()).reset_index()
    sns.barplot(data=edge_stats, x='edge_bucket', y='status', ax=axes[1, 0], hue='edge_bucket', legend=False, palette='viridis')
    axes[1, 0].set_title('Fill Rate by Demanded Edge')
    axes[1, 0].set_ylabel('Fill Rate')
    axes[1, 0].set_xlabel('Edge Demanded (cents)')
    
    # Plot 4: Fill Rate by Spread Bucket
    spread_stats = df.groupby('spread_bucket', observed=True)['status'].apply(lambda x: (x == 'filled').mean()).reset_index()
    sns.barplot(data=spread_stats, x='spread_bucket', y='status', ax=axes[1, 1], hue='spread_bucket', legend=False, palette='rocket')
    axes[1, 1].set_title('Fill Rate by Market Spread Width')
    axes[1, 1].set_ylabel('Fill Rate')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print("  ✓ Saved!")

def main():
    args = parse_args()
    
    print("🔎 Analyzing Order History...")
    if args.start:
        print(f"  Start Date: {args.start}")
    if args.end:
        print(f"  End Date: {args.end}")
        
    df = load_local_trades(args.start, args.end)
    
    if df.empty:
        print("No trades found for specified criteria.")
        return
        
    df_analyzed = analyze_fill_rates(df)
    plot_analysis(df_analyzed)
    
    print("\nNOTE: This analysis uses local database logs.")
    print("If 'market_bid' and 'market_ask' were not logged, we use 'market_spread' and 'model_fair_value' as proxies.")

if __name__ == "__main__":
    main()
