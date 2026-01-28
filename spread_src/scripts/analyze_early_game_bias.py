import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = 'data/nba_data.db'

def analyze_early_game_bias():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        seconds_remaining, 
        realized_pnl, 
        side,
        fill_price,
        model_fair_value
    FROM trades
    WHERE realized_pnl IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No trades with PnL found.")
        # Try finding why - status check
        conn = sqlite3.connect(DB_PATH)
        status_check = pd.read_sql_query("SELECT status, count(*) FROM trades GROUP BY status", conn)
        print("Status counts:")
        print(status_check)
        conn.close()
        return

    # Create buckets for game time (5-minute intervals)
    # NBA game has 2880 seconds (48 mins)
    df['mins_left'] = df['seconds_remaining'] / 60.0
    df['time_bucket'] = pd.cut(df['mins_left'], bins=np.arange(0, 50, 4), labels=np.arange(2, 50, 4))
    
    # Calculate Win Rate and PnL per bucket
    df['hit'] = 0
    df.loc[(df['side'] == 'buy') & (df['realized_pnl'] > 0), 'hit'] = 1
    df.loc[(df['side'] == 'sell') & (df['realized_pnl'] > 0), 'hit'] = 1
    # Note: Short side 'hit' logic might be more complex if realized_pnl is tricky, 
    # but generally realized_pnl > 0 means the trade was profitable.
    
    # Group by bucket
    stats = df.groupby('time_bucket', observed=False).agg({
        'realized_pnl': ['sum', 'count', 'mean'],
        'hit': 'mean'
    }).reset_index()
    stats.columns = ['mins_left', 'total_pnl', 'trade_count', 'avg_pnl', 'win_rate']

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Total PnL by Time Remaining
    sns.barplot(data=stats, x='mins_left', y='total_pnl', ax=ax1, palette='vlag')
    ax1.set_title('Total PnL vs. Time Remaining (Lower is Earlier Game)', fontsize=14)
    ax1.set_xlabel('Minutes Remaining', fontsize=12)
    ax1.set_ylabel('Total PnL ($)', fontsize=12)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.invert_xaxis() # 48m on left, 0m on right

    # Plot 2: Win Rate by Time Remaining
    sns.lineplot(data=stats, x=stats.index, y='win_rate', marker='o', ax=ax2, color='coral')
    ax2.set_xticks(stats.index)
    ax2.set_xticklabels(stats['mins_left'])
    ax2.set_title('Win Rate % vs. Time Remaining', fontsize=14)
    ax2.set_xlabel('Minutes Remaining', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('reports/game_time_bias.png')
    print("✅ Analysis complete. Chart saved to reports/game_time_bias.png")
    
    print("\nSummary Stats Table (Mins Remaining):")
    print(stats.sort_values('mins_left', ascending=False).to_string(index=False))

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    analyze_early_game_bias()
