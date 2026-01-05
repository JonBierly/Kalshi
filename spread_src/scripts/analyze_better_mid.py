import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_better_mid():
    try:
        df = pd.read_csv("data/model_vs_market_comparison.csv")
    except FileNotFoundError:
        print("Comparison results not found. Run compare_model_vs_market.py first.")
        return

    # Isolate Within-Spread observations
    # For many Kalshi markets, bid/ask are 0-1 (e.g. 0.35, 0.45)
    df['spread_width'] = df['ask'] - df['bid']
    
    # Filter for cases where the model is 'providing the fair price' within the gap
    within_df = df[(df['model_prob'] >= df['bid']) & (df['model_prob'] <= df['ask'])].copy()
    
    print(f"Total Observations: {len(df)}")
    print(f"Within-Spread Observations: {len(within_df)} ({len(within_df)/len(df)*100:.1f}%)")

    if within_df.empty:
        print("No within-spread observations found.")
        return

    # Bin by Spread Width
    # Common widths: 0.05, 0.10, 0.20, 0.30+
    bins = [0, 0.05, 0.10, 0.20, 0.40, 1.0]
    labels = ['0-5c', '5-10c', '10-20c', '20-40c', '40c+']
    within_df['width_bin'] = pd.cut(within_df['spread_width'], bins=bins, labels=labels)

    # Calculate Brier Scores for Model vs Market Mid
    # Market mid is stored as 'market_prob' in the CSV
    stats = within_df.groupby('width_bin', observed=True).agg({
        'model_brier': 'mean',
        'market_brier': 'mean',
        'ticker': 'count'
    }).rename(columns={'ticker': 'n_obs'})

    stats['improvement'] = (stats['market_brier'] - stats['model_brier']) / stats['market_brier'] * 100
    
    print("\n'Better Mid' Accuracy Comparison (Within-Spread Only)")
    print("=" * 60)
    print(stats)
    
    # Visualization
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    stats[['model_brier', 'market_brier']].plot(kind='bar', figsize=(10, 6))
    plt.title("Model vs. Market Mid Brier Score (Within-Spread Cases)")
    plt.xlabel("Market Spread Width")
    plt.ylabel("Mean Brier Score (Lower is Better)")
    plt.xticks(rotation=0)
    plt.legend(["NGBoost Fair Price", "Kalshi Mid-Price"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("reports/better_mid_analysis.png")
    print("\nSaved reports/better_mid_analysis.png")
    
    # Detailed Summary
    summary = f"""
'Better Mid' Analysis Summary
============================
This analysis isolates cases where the model FAIR PRICE sits BETWEEN the market bid and ask.
Goal: Determine if the model is a better estimator of the true probability than the simple mid-price.

Key Results:
{stats.to_string()}

Observations where Model is more accurate than Mid: { (within_df['model_brier'] < within_df['market_brier']).sum() } / {len(within_df)}
"""
    with open("reports/better_mid_summary.txt", "w") as f:
        f.write(summary)
    print("Saved reports/better_mid_summary.txt")

if __name__ == "__main__":
    analyze_better_mid()
