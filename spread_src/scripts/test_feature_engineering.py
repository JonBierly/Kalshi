
from spread_src.features.engineering import add_interaction_features
import pandas as pd
import numpy as np

def test_feature_engineering():
    # Test dictionary case
    base_feats = {
        'score_diff': 10.0,
        'seconds_remaining': 1440.0, # 24 mins
        'home_team_season_win_margin': 5.0,
        'away_team_season_win_margin': -2.0,
        'home_roster_recent_pie': 0.12,
        'away_roster_recent_pie': 0.08
    }
    
    result = add_interaction_features(base_feats)
    
    print("Testing Feature Interactions (Dictionary):")
    time_prop = 1440.0 / 2880.0 # 0.5
    print(f"  Time Proportion: {time_prop}")
    print(f"  Home Season Margin x Time: {result['home_season_margin_x_time']} (Expected: {5.0 * time_prop})")
    print(f"  Away Season Margin x Time: {result['away_season_margin_x_time']} (Expected: {-2.0 * time_prop})")
    print(f"  Home PIE x Time: {result['home_pie_x_time']} (Expected: {0.12 * time_prop})")

    # Test DataFrame case
    df = pd.DataFrame([base_feats])
    result_df = add_interaction_features(df)
    
    print("\nTesting Feature Interactions (DataFrame):")
    print(f"  Home Season Margin x Time: {result_df.iloc[0]['home_season_margin_x_time']} (Expected: {5.0 * time_prop})")
    
    # Check 10s case
    base_feats['seconds_remaining'] = 10.0
    time_prop_10 = 10.0 / 2880.0
    result_10 = add_interaction_features(base_feats)
    print(f"\nTesting 10s case:")
    print(f"  Home Season Margin x Time: {result_10['home_season_margin_x_time']} (Expected: {5.0 * time_prop_10})")

if __name__ == "__main__":
    test_feature_engineering()
