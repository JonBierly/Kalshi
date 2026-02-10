import pandas as pd

def analyze_comparison(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Overall Brier
    print("--- Overall Brier ---")
    print(df.groupby('model')['brier'].mean())
    
    # 2. Tight Game Performance (abs(score_diff) <= 12)
    print("\n--- Tight Games (abs(diff) <= 12) ---")
    tight_df = df[df['score_diff'].abs() <= 12]
    print(tight_df.groupby('model')['brier'].mean())
    print(f"Num tight samples: {len(tight_df)}")
    
    # 3. Blowouts (abs(score_diff) > 20)
    print("\n--- Blowouts (abs(diff) > 20) ---")
    blowout_df = df[df['score_diff'].abs() > 20]
    print(blowout_df.groupby('model')['brier'].mean())
    print(f"Num blowout samples: {len(blowout_df)}")
    
    # 4. Late Game Tight (Tight & <10m left)
    print("\n--- Late Game Tight (<10m left & abs(diff) <= 12) ---")
    late_tight = tight_df[tight_df['seconds_remaining'] < 600]
    print(late_tight.groupby('model')['brier'].mean())

if __name__ == "__main__":
    analyze_comparison('reports/model_comparison_100.csv')
