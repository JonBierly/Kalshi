import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = 'data/nba_data.db'

def analyze_pnl_by_prob(min_edge=2.0):
    if not os.path.exists(DB_PATH):
        print("Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT trade_id, side, fill_price, realized_pnl, size, model_fair_value 
        FROM trades 
        WHERE realized_pnl IS NOT NULL
        AND size > 0
    """, conn)
    conn.close()
    
    if df.empty:
        print("No trades found")
        return

    # Calculate Edge to filter by min_edge (consistency with other reports)
    df['edge'] = abs(df['model_fair_value'] - df['fill_price'])
    df = df[df['edge'] >= min_edge]
    
    if df.empty:
        print(f"No trades found with edge >= {min_edge}¢")
        return

    # Calculate "Bet Probability"
    # model_fair_value is P(YES) in cents (0-100)
    # If side is 'buy', we bet on YES. Prob = model_fair_value.
    # If side is 'sell', we bet on NO. Prob = 100 - model_fair_value.
    df['bet_prob'] = df.apply(
        lambda row: row['model_fair_value'] if row['side'].lower() == 'buy' else (100 - row['model_fair_value']),
        axis=1
    )

    # Create probability bins
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    df['prob_bin'] = pd.cut(df['bet_prob'], bins=bins, labels=labels, include_lowest=True)

    # Group by probability bin AND side
    summary_side = df.groupby(['prob_bin', 'side'], observed=True).agg({
        'realized_pnl': ['mean', 'sum', 'count']
    }).reset_index()
    summary_side.columns = ['prob_bin', 'side', 'avg_pnl', 'total_pnl', 'trade_count']

    # Also get the global summary (all sides)
    summary_all = df.groupby('prob_bin', observed=True).agg({
        'realized_pnl': ['mean', 'sum', 'count']
    }).reset_index()
    summary_all.columns = ['prob_bin', 'avg_pnl', 'total_pnl', 'trade_count']

    print(f"\n📊 P&L Analysis by Predicted Probability and Side (Min Edge: {min_edge}¢)")
    print(summary_side.sort_values(['prob_bin', 'side']).to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))
    fig.suptitle(f'Performance vs. Model Confidence (Edge >= {min_edge}¢)', fontsize=16)

    # Plot 1: Avg P&L per Trade by Side
    sns.barplot(data=summary_side, x='prob_bin', y='avg_pnl', hue='side', palette={'buy': '#3498db', 'sell': '#e67e22'}, ax=axes[0])
    axes[0].set_title('Average P&L per Trade by Confidence & Side')
    axes[0].set_ylabel('Avg Realized P&L ($)')
    axes[0].axhline(0, color='black', linewidth=1.5)
    
    # Add counts below/above bars
    for i, p in enumerate(axes[0].patches):
        if p.get_height() == 0: continue
        val = p.get_height()
        axes[0].annotate(f"{int(val*100):+d}¢", 
                        (p.get_x() + p.get_width() / 2., val), 
                        ha='center', va='bottom' if val >= 0 else 'top', 
                        xytext=(0, 5 if val >= 0 else -5), 
                        textcoords='offset points', fontsize=8)

    # Plot 2: Total P&L by Side
    sns.barplot(data=summary_side, x='prob_bin', y='total_pnl', hue='side', palette={'buy': '#3498db', 'sell': '#e67e22'}, ax=axes[1])
    axes[1].set_title('Total Realized P&L by Confidence & Side')
    axes[1].set_ylabel('Total P&L ($)')
    axes[1].axhline(0, color='black', linewidth=1.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = getattr(args, 'output', 'pnl_by_prob.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Analysis plot saved to {output_path}")

    return summary_side

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=2.0, help="Min edge filter in cents")
    parser.add_argument("--output", type=str, default="pnl_by_prob.png", help="Output filename for plot")
    args = parser.parse_args()
    
    analyze_pnl_by_prob(min_edge=args.min_edge)
