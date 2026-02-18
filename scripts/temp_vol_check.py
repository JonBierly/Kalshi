
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('reports/volatility/vol_comparison_data.csv')
    
    # Calculate absolute momentum (magnitude of the run)
    df['abs_momentum'] = df['momentum_2min'].abs()
    
    # Group by momentum buckets
    bins = [0, 2, 5, 10, 100]
    labels = ['Calm (0-2)', 'Active (3-5)', 'Run (6-10)', 'Big Run (>10)']
    df['mom_bucket'] = pd.cut(df['abs_momentum'], bins=bins, labels=labels)
    
    print("\n=== Volatility Ratio (Model / Market) by Momentum ===")
    print("If Ratio < 1.0, Market Volatility > Model Volatility (Market Overreacting?)")
    result = df.groupby('mom_bucket', observed=True)[['model_std', 'market_std', 'vol_ratio']].mean()
    result['count'] = df.groupby('mom_bucket', observed=True).size()
    print(result)
    
    print("\n=== Correlation Matrix ===")
    print(df[['abs_momentum', 'model_std', 'market_std', 'vol_ratio']].corr())

except Exception as e:
    print(e)
