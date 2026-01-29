import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

DB_PATH = 'data/nba_data.db'

def extract_side_from_ticker(ticker):
    # Ticker: KXNBASPREAD-26JAN15CHALAL-LAL4.5
    # The team in the ticker is the one the spread applies to.
    pass

def analyze_calibration(min_edge=0.0):
    if not os.path.exists(DB_PATH):
        print("Database not found")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Wait, size is in the DB.
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

    # Calculate Edge
    df['edge'] = abs(df['model_fair_value'] - df['fill_price'])
    
    # Filter by min_edge
    initial_count = len(df)
    df = df[df['edge'] >= min_edge]
    print(f"Filtered to edge >= {min_edge}¢: {len(df)} trades (from {initial_count})")

    if df.empty:
        print(f"No trades found with edge >= {min_edge}¢")
        return

    def get_outcome(row):
        if row['size'] == 0: return None
        if row['side'] == 'buy':
            # realized_pnl = (outcome - fill_price) * size / 100
            outcome = (100 * row['realized_pnl'] / row['size']) + row['fill_price']
        else:
            # realized_pnl = (fill_price - outcome) * size / 100
            outcome = row['fill_price'] - (100 * row['realized_pnl'] / row['size'])
        
        # Round to nearest 0 or 100
        return 100 if outcome > 50 else 0

    df['actual_outcome'] = df.apply(get_outcome, axis=1)
    df = df.dropna(subset=['actual_outcome'])

    print(f"\n📊 Calibration Check (Min Edge: {min_edge}¢)...")
    
    for side in ['buy', 'sell']:
        side_df = df[df['side'] == side]
        if side_df.empty:
            print(f"\n{side.upper()}: No data")
            continue
            
        avg_pred = side_df['model_fair_value'].mean()
        avg_actual = side_df['actual_outcome'].mean()
        bias = avg_pred - avg_actual
        
        print(f"\n{side.upper()}:")
        print(f"  Trades: {len(side_df)}")
        print(f"  Avg Predicted Prob: {avg_pred:.1f}%")
        print(f"  Avg Actual Outcome: {avg_actual:.1f}%")
        print(f"  Calibration Bias: {bias:+.1f}%")

    # Visualizing Calibration Curve
    df['pred_bucket'] = pd.cut(df['model_fair_value'], bins=np.linspace(0, 100, 11))
    calib = df.groupby(['side', 'pred_bucket'], observed=True)['actual_outcome'].mean().reset_index()
    
    plt.figure(figsize=(10, 8))
    for side in ['buy', 'sell']:
        data = calib[calib['side'] == side]
        if data.empty: continue
        # Get midpoints of buckets for plotting
        x = [b.mid for b in data['pred_bucket']]
        plt.plot(x, data['actual_outcome'], marker='o', label=f'Actual ({side})')
        
    plt.plot([0, 100], [0, 100], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability (Model Fair Value)')
    plt.ylabel('Actual Win Rate (%)')
    plt.title(f'Model Calibration (Edge >= {min_edge}¢)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('calibration_check.png')
    print("\n✅ Calibration plot saved to calibration_check.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=0.0, help="Min edge filter in cents")
    args = parser.parse_args()
    
    analyze_calibration(min_edge=args.min_edge)
