import pandas as pd
from src.data.acquisition import MockHistoricalDataClient
from src.features.engineering import create_live_features, FeatureEngine

def test_feature_generation():
    print("Testing Feature Generation...")
    
    # Get mock data
    client = MockHistoricalDataClient()
    pbp_df = client.get_game_pbp('0022300001')
    
    print(f"Mock Data Shape: {pbp_df.shape}")
    
    # Generate features
    features_df = create_live_features(pbp_df)
    
    print(f"Features Shape: {features_df.shape}")
    print("Feature Columns:", features_df.columns.tolist())
    
    # Check for NaNs
    if features_df.isnull().values.any():
        print("WARNING: NaNs found in features.")
        print(features_df[features_df.isnull().any(axis=1)])
    else:
        print("No NaNs found.")
        
    # Check logic (e.g. score diff matches)
    # The mock data has score_diff, let's compare
    # Note: FeatureEngine takes the row's score_diff directly, so it should match.
    
    # Check eFG% logic
    # In mock, we only have Makes (event_type=1). So eFG should be 1.0 (or 1.5 for 3s) if we track FGA correctly.
    # Wait, mock data has event_type=1 (Make). 
    # FeatureEngine updates stats: Make -> FGM+1, FGA+1.
    # So eFG = FGM/FGA = 1.0.
    
    print("Sample Features:")
    print(features_df[['score_diff', 'home_efg', 'away_efg', 'seconds_remaining']].head())
    print(features_df[['score_diff', 'home_efg', 'away_efg', 'seconds_remaining']].tail())
    
    # Verify eFG is non-zero (since we have makes)
    if features_df['home_efg'].max() > 0 or features_df['away_efg'].max() > 0:
        print("eFG% is being calculated.")
    else:
        print("ERROR: eFG% is all zero.")

if __name__ == "__main__":
    test_feature_generation()
