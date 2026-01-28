import sqlite3
import pandas as pd
import numpy as np
import os

def analyze_performance():
    db_path = 'data/nba_data.db'
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # Brier Score = mean( (prob - outcome)^2 )
    def brier_score(prob, outcome):
        return np.mean((prob - outcome)**2)

    # Load predictions that have been settled
    query = """
    SELECT 
        predicted_prob, 
        bid_price, 
        ask_price, 
        actual_outcome,
        seconds_remaining,
        ticker
    FROM model_predictions 
    WHERE actual_outcome IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No settled predictions found.")
        return

    # Calculate Market Probability (midpoint)
    df['market_mid'] = (df['bid_price'] + df['ask_price']) / 200.0
    
    # Side-Adjusted Maker Benchmark:
    # If model is Bullish (prob >= mid), benchmark is the price to buy (bid + 1)
    # If model is Bearish (prob < mid), benchmark is the price to sell (ask - 1)
    df['benchmark_price'] = np.where(
        df['predicted_prob'] >= df['market_mid'],
        (df['bid_price'] + 1) / 100.0,
        (df['ask_price'] - 1) / 100.0
    )
    
    # Calculate SIGNED Maker Edge
    # Long Edge: Model Prob - Buy Price
    # Short Edge: Sell Price - Model Prob
    df['maker_edge'] = np.where(
        df['predicted_prob'] >= df['market_mid'],
        df['predicted_prob'] - df['benchmark_price'],
        df['benchmark_price'] - df['predicted_prob']
    )
    
    # Manufacturer Execution Pricing (The price we actually get)
    # Long if model > midpoint, Short if model < midpoint
    df['exec_price'] = np.where(
        df['predicted_prob'] >= df['market_mid'],
        (df['bid_price'] + 1) / 100.0,
        (df['ask_price'] - 1) / 100.0
    )
    
    # 1. Manufacturer Directional Accuracy:
    # Did the outcome justify the side we took AT THE PRICE we took it?
    # Success means Outcome matched side AND we beat the entry price.
    # For longs: outcome=1 and price < 1.0 (always true if 1)
    # For shorts: outcome=0 and price > 0.0 (always true if 0)
    df['was_maker_correct'] = np.where(
        df['predicted_prob'] >= df['market_mid'],
        df['actual_outcome'] == 1,
        df['actual_outcome'] == 0
    )
    
    # 2. Manufacturer Profit (¢ per contract):
    # Profit = (Outcome - Exec Price) if Long, (Exec Price - Outcome) if Short
    df['maker_profit'] = np.where(
        df['predicted_prob'] >= df['market_mid'],
        df['actual_outcome'].astype(float) - df['exec_price'],
        df['exec_price'] - df['actual_outcome'].astype(float)
    )

    # Filter for valid market prices AND >= 5% "True" model edge
    # AND ignore trades with less than 3 minutes left (180s)
    valid_df = df.dropna(subset=['market_mid']).copy()
    valid_df = valid_df[valid_df['seconds_remaining'] >= 180]
    
    edge_df = valid_df[valid_df['maker_edge'] >= 0.05].copy()

    # Brier Score = mean( (prob - outcome)^2 )
    def brier_score(prob, outcome):
        return np.mean((prob - outcome)**2)

    if edge_df.empty:
        print("No predictions found with a 5% edge (>= 180s remaining).")
        return

    # Global Manufacturer performance
    global_maker_acc = valid_df['was_maker_correct'].mean()
    global_maker_profit = valid_df['maker_profit'].mean()
    
    # High-Edge Manufacturer performance
    model_brier = brier_score(edge_df['predicted_prob'], edge_df['actual_outcome'])
    benchmark_brier = brier_score(edge_df['benchmark_price'], edge_df['actual_outcome'])
    edge_maker_acc = edge_df['was_maker_correct'].mean()
    edge_maker_profit = edge_df['maker_profit'].mean()

    print("=" * 85)
    print(f"NBA SPREAD PREDICTION PERFORMANCE: MANUFACTURER-AWARE BENCHMARKS")
    print("=" * 85)
    print(f"Total Valid Predictions (>= 3m left): {len(valid_df):,}")
    print(f"Predictions with 5% Edge: {len(edge_df):,} ({len(edge_df)/len(valid_df):.1%})")
    print("-" * 85)
    print("GLOBAL MANUFACTURER PERFORMANCE (Every prediction, priced at bid+1/ask-1):")
    print(f"Win Rate:              {global_maker_acc:.1%}")
    print(f"Avg Profit/Contract:   {global_maker_profit*100:+.2f}¢")
    print("-" * 85)
    print("HIGH-CONFIDENCE PERFORMANCE (>= 5% EDGE over Maker Price):")
    print(f"Model Brier:           {model_brier:.5f}")
    print(f"Benchmark Brier:       {benchmark_brier:.5f}")
    print(f"Win Rate:              {edge_maker_acc:.1%}")
    print(f"Avg Profit/Contract:   {edge_maker_profit*100:+.2f}¢")
    
    print("-" * 85)
    rel_cal = (benchmark_brier - model_brier) / benchmark_brier
    if rel_cal > 0:
        print(f"VERDICT: Model IS more calibrated than its entry price ({rel_cal:.1%} better)")
    else:
        print(f"VERDICT: Entry Price is more calibrated than raw model ({abs(rel_cal):.1%} better)")
    
    if edge_maker_profit > 0:
        print(f"VERDICT: Manufacturer strategy is PROFITABLE at this edge level")
    else:
        print(f"VERDICT: Manufacturer strategy is LOSING at this edge level")
    
    # Stage breakdown
    # Bins for:
    # Q1: 12-6m (2880-2520), 6-0m (2520-2160)
    # Q2: 12-6m (2160-1800), 6-0m (1800-1440)
    # Q3: 12-6m (1440-1080), 6-0m (1080-720)
    # Q4: 12-6m (720-360), 6-3m (360-180)
    bins = [179, 360, 720, 1080, 1440, 1800, 2160, 2520, 2881]
    labels = [
        'Q4 6-3m', 'Q4 12-6m', 
        'Q3 6-0m', 'Q3 12-6m', 
        'Q2 6-0m', 'Q2 12-6m', 
        'Q1 6-0m', 'Q1 12-6m'
    ]
    
    print("\nManufacturer Performance by Game Stage (5% Edge Only):")
    edge_df['stage'] = pd.cut(edge_df['seconds_remaining'], bins=bins, labels=labels)
    
    stage_stats = edge_df.groupby('stage', observed=False).apply(
        lambda x: pd.Series({
            'WinRate': x['was_maker_correct'].mean(),
            'AvgProfit': x['maker_profit'].mean() * 100,
            'Count': len(x)
        })
    ).round(2)
    print(stage_stats)
    
    # Visualizations
    import matplotlib.pyplot as plt
    
    # Sort labels for the plot to be chronological (Q1 -> Q4)
    plot_stats = stage_stats.iloc[::-1] # Reverse to get Q1 first
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Avg Profit
    colors = ['green' if x > 0 else 'red' for x in plot_stats['AvgProfit']]
    plot_stats['AvgProfit'].plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Avg Profit per Contract (¢) by Game Stage')
    ax1.set_ylabel('Cents')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Plot 2: Trade Count
    plot_stats['Count'].plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Number of Trades (>= 5% Edge) by Game Stage')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plot_path = 'performance_analysis.png'
    plt.savefig(plot_path)
    print(f"\n📈 Saved visualization to {plot_path}")
    print("=" * 85)
    print("Note: All metrics use actual executable prices (bid+1 for Long, ask-1 for Short).")

if __name__ == "__main__":
    analyze_performance()
