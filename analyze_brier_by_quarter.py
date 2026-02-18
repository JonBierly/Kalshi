
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_brier_by_stage():
    # Load comparison data which has Brier scores pre-calculated per row
    df = pd.read_csv('/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/model_vs_market_comparison.csv')
    
    # Bins for Quarters (Seconds Remaining)
    # Q1: 2880 - 2160
    # Q2: 2160 - 1440
    # Q3: 1440 - 720
    # Q4: 720 - 0
    bins = [0, 720, 1440, 2160, 2881]
    labels = ['Q4 (Late)', 'Q3', 'Q2', 'Q1 (Early)']
    
    df['stage'] = pd.cut(df['seconds_remaining'], bins=bins, labels=labels)
    
    # Group by stage
    stats = df.groupby('stage', observed=False)[['model_brier', 'market_brier']].mean()
    
    # Calculate difference (Positive = Market is Better, Negative = Model is Better)
    # Use pct difference relative to Market
    stats['Model_vs_Market'] = (stats['model_brier'] - stats['market_brier']) / stats['market_brier']
    
    print("\n--- Brier Score Analysis by Quarter ---")
    print("(Lower is Better. Negative 'Diff' means Model Wins)")
    print(stats)
    
    print("\n--- Interpretation ---")
    for stage in stats.index[::-1]: # Chronological
        m_brier = stats.loc[stage, 'model_brier']
        kt_brier = stats.loc[stage, 'market_brier']
        
        if m_brier < kt_brier:
            print(f"✅ {stage}: Model WINS (Brier {m_brier:.4f} vs {kt_brier:.4f})")
        else:
            print(f"❌ {stage}: Market WINS (Brier {m_brier:.4f} vs {kt_brier:.4f})")

    # Global Average
    print("\n--- Global Average ---")
    print(f"Model: {df['model_brier'].mean():.4f}")
    print(f"Market: {df['market_brier'].mean():.4f}")

if __name__ == "__main__":
    analyze_brier_by_stage()
