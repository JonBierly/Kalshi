#!/usr/bin/env python
"""
Analyze P&L drivers to identify profitability leaks.

Correlates realized P&L with:
1. Market Spread Width
2. Edge Demanded (Aggressiveness)
3. Time Remaining (Game Phase)
4. Model Confidence (CI Width)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)

DB_PATH = 'data/nba_data.db'

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze P&L drivers.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()

def load_closed_trades(start_date=None, end_date=None):
    """Load closed trades with P&L from database."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Query for trades that have a realized P&L (even if 0)
        # We also fetch model details logged at time of trade
        query = """
            SELECT 
                trade_id, timestamp, ticker, side, fill_price, size,
                model_fair_value, model_ci_lower, model_ci_upper,
                market_spread, seconds_remaining,
                realized_pnl, status, created_at, closed_at
            FROM trades
            WHERE realized_pnl IS NOT NULL
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
            query += " AND " + " AND ".join(conditions)
            
        df = pd.read_sql_query(query, conn, params=args)
        conn.close()
        
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate metrics
            # Model Edge at Entry = abs(Fill Price - Fair Value)
            # Note: We use fill_price now, not order_price
            df['entry_edge'] = abs(df['fill_price'] - df['model_fair_value'])
            
            # CI Width (Uncertainty)
            df['ci_width'] = df['model_ci_upper'] - df['model_ci_lower']
            
            # Game Phase (in minutes)
            df['mins_remaining'] = df['seconds_remaining'] / 60.0
            
        return df
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return pd.DataFrame()

def analyze_pnl_drivers(df):
    """Analyze what factors drive P&L."""
    if df.empty:
        print("No closed trades found with P&L data.")
        return
        
    print(f"\n📊 Analyzing {len(df)} closed trades...")
    print(f"Total Realized P&L: ${df['realized_pnl'].sum():.2f}")
    
    # --- 1. Daily Breakdown ---
    df['date'] = df['created_at'].dt.date
    daily_pnl = df.groupby('date')['realized_pnl'].sum().reset_index()
    print("\n📅 Daily P&L:")
    for _, row in daily_pnl.iterrows():
        print(f"  {row['date']}: ${row['realized_pnl']:>8.2f}")

    # --- Metrics Bucketing ---
    
    # Spread Buckets
    df['spread_bucket'] = pd.cut(df['market_spread'], 
                                bins=[0, 10, 25, 50, 200], 
                                labels=['Tight (0-10)', 'Normal (10-25)', 'Wide (25-50)', 'Very Wide (>50)'])
                                
    # Time Buckets (Game Phase)
    # NBA game is 48 mins long
    df['time_bucket'] = pd.cut(df['mins_remaining'], 
                              bins=[0, 5, 12, 24, 36, 48],
                              labels=['Crunch Time (0-5m)', '4th Qtr (5-12m)', '2nd Half', '1st Half', 'Early'])
                              
    # Entry Edge Buckets
    df['edge_bucket'] = pd.cut(df['entry_edge'], 
                              bins=[0, 2, 5, 8, 100],
                              labels=['Thin Edge (0-2)', 'Med Edge (2-5)', 'Fat Edge (5-8)', 'Huge Edge (>8)'])

    # --- Visualization ---
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Profitability Drivers Analysis', fontsize=20)
    
    # Plot 1: Daily P&L
    sns.barplot(data=daily_pnl, x='date', y='realized_pnl', ax=axes[0, 0], palette='RdBu')
    axes[0, 0].set_title('Total P&L by Date')
    axes[0, 0].axhline(0, color='black', linewidth=1)
    
    # Plot 2: P&L by Spread Width
    spread_pnl = df.groupby('spread_bucket', observed=True)['realized_pnl'].agg(['sum', 'count', 'mean']).reset_index()
    sns.barplot(data=spread_pnl, x='spread_bucket', y='sum', ax=axes[0, 1], palette='RdBu')
    axes[0, 1].set_title('Total P&L by Market Spread Width')
    axes[0, 1].set_ylabel('Total P&L ($)')
    axes[0, 1].axhline(0, color='black', linewidth=1)
    
    # Print Spread Stats
    print("\n📉 P&L by Spread Width:")
    for _, row in spread_pnl.iterrows():
        print(f"  {row['spread_bucket']:<15}: ${row['sum']:>8.2f} (n={row['count']}, avg=${row['mean']:.2f})")
    
    # Plot 3: P&L by Game Phase
    time_pnl = df.groupby('time_bucket', observed=True)['realized_pnl'].agg(['sum', 'count', 'mean']).reset_index()
    sns.barplot(data=time_pnl, x='time_bucket', y='sum', ax=axes[1, 0], palette='RdBu')
    axes[1, 0].set_title('Total P&L by Time Remaining')
    axes[1, 0].set_ylabel('Total P&L ($)')
    axes[1, 0].axhline(0, color='black', linewidth=1)
    
    # Print Time Stats
    print("\n⏱️ P&L by Game Phase:")
    for _, row in time_pnl.iterrows():
        print(f"  {row['time_bucket']:<15}: ${row['sum']:>8.2f} (n={row['count']}, avg=${row['mean']:.2f})")

    # Plot 4: P&L by Entry Edge
    edge_pnl = df.groupby('edge_bucket', observed=True)['realized_pnl'].agg(['sum', 'count', 'mean']).reset_index()
    sns.barplot(data=edge_pnl, x='edge_bucket', y='sum', ax=axes[1, 1], palette='RdBu')
    axes[1, 1].set_title('Total P&L by Entry Edge (Model vs Market)')
    axes[1, 1].set_ylabel('Total P&L ($)')
    axes[1, 1].axhline(0, color='black', linewidth=1)
    
    # Print Edge Stats
    print("\n🎯 P&L by Entry Edge:")
    for _, row in edge_pnl.iterrows():
        print(f"  {row['edge_bucket']:<15}: ${row['sum']:>8.2f} (n={row['count']}, avg=${row['mean']:.2f})")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pnl_drivers_analysis.png')
    print("\n✅ Saved plots to pnl_drivers_analysis.png")

def main():
    args = parse_args()
    print("🔎 Analyzing P&L Drivers...")
    if args.start: print(f"  Start: {args.start}")
    if args.end:   print(f"  End:   {args.end}")
    
    df = load_closed_trades(args.start, args.end)
    analyze_pnl_drivers(df)

if __name__ == "__main__":
    main()
