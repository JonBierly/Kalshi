
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import mean_absolute_error

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST

def analyze_time_bias():
    print("Loading models and data for bias analysis...")
    try:
        model_data = joblib.load('models/nba_spread_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Prepare data (use the same logic as training but for analysis)
    from spread_src.scripts.train_spread_model import get_final_score_diffs
    print("Loading training data...")
    X, target = prepare_training_data()
    
    print("Getting final score differentials (Bulk)...")
    from data.database import DatabaseManager
    db = DatabaseManager()
    
    # Get the last row of pbp_events for ALL games in X
    game_ids = X['game_id'].unique().tolist()
    # Format for SQL
    game_id_str = ",".join([f"'{gid}'" for gid in game_ids])
    
    # Query to get the final score for each game efficiently
    query = f"""
        WITH LastEvents AS (
            SELECT game_id, home_score, away_score,
                   ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY period DESC, remaining_time ASC) as rn
            FROM pbp_events
            WHERE game_id IN ({game_id_str})
        )
        SELECT game_id, home_score - away_score as final_diff
        FROM LastEvents
        WHERE rn = 1
    """
    final_diff_df = pd.read_sql(query, db.engine)
    
    # Map to X
    mapping = final_diff_df.set_index('game_id')['final_diff'].to_dict()
    final_diffs = X['game_id'].map(mapping).values
    current_diffs = X['score_diff'].values
    
    # Filter to features
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    # Only use columns that exist in the trained model (handle mismatches if any)
    available_cols = [c for c in feature_cols if c in X.columns]
    X_features = X[available_cols]
    
    # Predict remainder
    print(f"Generating predictions for {len(X_features)} rows...")
    preds = []
    # Each model in ensemble
    for m in model_data:
        # Check if it has mean_model
        if isinstance(m, dict) and 'mean_model' in m:
            preds.append(m['mean_model'].predict(X_features))
        else:
            preds.append(m.predict(X_features))
    
    avg_remainder_pred = np.mean(preds, axis=0)
    reconstructed_final_diff = current_diffs + avg_remainder_pred
    
    # Calculate Error
    error = reconstructed_final_diff - final_diffs
    abs_error = np.abs(error)
    
    # Analyze by time buckets
    X_analysis = X.copy()
    X_analysis['abs_error'] = abs_error
    X_analysis['seconds_remaining'] = X_features['seconds_remaining']
    
    # Buckets
    def get_bucket(s):
        if s > 2160: return "Q1 (>36m)"
        if s > 1440: return "Q2 (24-36m)"
        if s > 720: return "Q3 (12-24m)"
        if s > 120: return "Q4 (2-12m)"
        return "Clutch (<2m)"
        
    X_analysis['bucket'] = X_analysis['seconds_remaining'].apply(get_bucket)
    
    summary = X_analysis.groupby('bucket')['abs_error'].agg(['mean', 'count']).reset_index()
    order = ["Q1 (>36m)", "Q2 (24-36m)", "Q3 (12-24m)", "Q4 (2-12m)", "Clutch (<2m)"]
    summary['bucket'] = pd.Categorical(summary['bucket'], categories=order, ordered=True)
    summary = summary.sort_values('bucket')
    
    print("\nMAE BY TIME PHASE:")
    print(summary.to_string(index=False))
    
    # Check "Stickiness"
    # Ideally, pred_move (predicted remainder) should be large early and small late
    X_analysis['pred_move'] = avg_remainder_pred
    X_analysis['actual_move'] = final_diffs - current_diffs
    
    summary_move = X_analysis.groupby('bucket')[['pred_move', 'actual_move']].agg(['mean', 'std']).reset_index()
    summary_move['bucket'] = pd.Categorical(summary_move['bucket'], categories=order, ordered=True)
    summary_move = summary_move.sort_values('bucket')
    
    print("\nEXPECTED VS ACTUAL MOVEMEMENT (FINAL - CURRENT):")
    print(summary_move.to_string(index=False))

if __name__ == "__main__":
    analyze_time_bias()
