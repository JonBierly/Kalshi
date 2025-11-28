from src.models.training import prepare_training_data
import pandas as pd
import numpy as np

def inspect():
    print("Loading training data...")
    X, y = prepare_training_data(num_games=50) # Load a sample
    
    df = X.copy()
    df['target'] = y
    
    print(f"\nData Shape: {df.shape}")
    
    # Check correlation
    corr = df[['score_diff', 'target']].corr().iloc[0, 1]
    print(f"\nCorrelation between score_diff and target (Home Win): {corr:.4f}")
    
    print("\nFeature Stats:")
    print(df[['seconds_remaining', 'log_seconds', 'score_diff']].describe())
    
    print("\n--- Sample Rows (Home Win) ---")
    home_wins = df[df['target'] == 1].sample(5)
    print(home_wins[['score_diff', 'target']])
    
    print("\n--- Sample Rows (Home Loss) ---")
    home_losses = df[df['target'] == 0].sample(5)
    print(home_losses[['score_diff', 'target']])
    
    # Check for anomalies
    print("\n--- Anomalies (Positive Score Diff but Target=0) ---")
    anomalies = df[(df['score_diff'] > 10) & (df['target'] == 0)]
    print(f"Count: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies[['score_diff', 'target']].head())

if __name__ == "__main__":
    inspect()
