"""
Quick Feature Test for NGBoost
Tries to validate if expanded features improve NLL/MAE without training a full ensemble.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Set non-interactive backend
plt.switch_backend('Agg')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ngboost import NGBRegressor
from ngboost.scores import LogScore
from spread_src.models.distributions import SafeT
from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import (
    BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, INTERACTION_FEATURES_LIST,
    add_interaction_features
)

def get_final_score_diffs_cached(X: pd.DataFrame) -> np.ndarray:
    """Optimized version of final score retrieval."""
    db = DatabaseManager()
    game_ids = X['game_id'].unique()
    
    # Batch query for all games
    ids_str = "', '".join(game_ids)
    query = f"""
        WITH LastEvents AS (
            SELECT game_id, home_score, away_score,
                   ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY period DESC, remaining_time ASC) as rn
            FROM pbp_events
            WHERE game_id IN ('{ids_str}')
        )
        SELECT game_id, (home_score - away_score) as final_diff
        FROM LastEvents
        WHERE rn = 1
    """
    df_final = pd.read_sql(query, db.engine)
    mapping = dict(zip(df_final['game_id'], df_final['final_diff']))
    
    return X['game_id'].map(mapping).fillna(0).values

def run_test(time_slice=None, slice_name="Full Game"):
    print("\n" + "=" * 60)
    print(f"Test: {slice_name}")
    print("=" * 60)
    
    # 1. Load a subset of data for speed
    print("\n[1/5] Loading data...")
    X, _ = prepare_training_data()
    # Sample subset of games to keep it quick
    all_game_ids = X['game_id'].unique()
    test_games = np.random.choice(all_game_ids, size=min(400, len(all_game_ids)), replace=False)
    X = X[X['game_id'].isin(test_games)].copy()
    
    # Filter by time if requested
    if time_slice == 'early':
        X = X[X['seconds_remaining'] > 1800] # 1st Half
    elif time_slice == 'mid':
        X = X[(X['seconds_remaining'] <= 1800) & (X['seconds_remaining'] > 600)] # 3rd Qtr + start of 4th
    elif time_slice == 'late':
        X = X[X['seconds_remaining'] <= 600] # Final 10 mins
        
    print(f"  Rows in slice: {len(X)}")
    if len(X) < 1000:
        print("  WARNING: Slice too small, skipping.")
        return None

    # 2. Prepare Targets
    print("[2/5] Preparing targets...")
    final_diffs = get_final_score_diffs_cached(X)
    y = final_diffs - X['score_diff'].values
    
    # 3. Add all features
    print("[3/5] Engineering features...")
    X = add_interaction_features(X)
    
    # Feature groups
    LIVE_STATS = [
        'home_efg', 'away_efg', 'turnover_diff', 'home_rebound_rate', 
        'live_pace', 'score_momentum', 'home_3p_reliance', 'away_3p_reliance',
        'home_steal_rate', 'away_steal_rate', 'home_block_rate', 'away_block_rate',
        'lead_changes', 'score_volatility'
    ]
    HISTORICAL_STATS = [c for c in ADVANCED_FEATURES_LIST if 'recent' in c]
    STATE_FEATURES = ['score_diff', 'seconds_remaining', 'period', 'log_time', 'time_proportion']
    INTERACTION = INTERACTION_FEATURES_LIST
    
    experimental_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST + INTERACTION_FEATURES_LIST
    experimental_cols = [c for c in experimental_cols if c in X.columns]
    
    # 4. Train Model
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train, X_val = X.iloc[train_idx][experimental_cols], X.iloc[val_idx][experimental_cols]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = NGBRegressor(
        Dist=SafeT,
        Score=LogScore,
        n_estimators=150,
        learning_rate=0.1,
        minibatch_frac=0.1,
        verbose=False,
        random_state=42
    )
    
    model.fit(X_train.values, y_train, X_val=X_val.values, Y_val=y_val)
    
    # 5. Extract Importance Aggregates
    try:
        importances = model.feature_importances_
        # Loc Importance
        loc_imp = importances[0]
        
        agg = {
            'Live Stats': sum(imp for f, imp in zip(experimental_cols, loc_imp) if f in LIVE_STATS),
            'Historical': sum(imp for f, imp in zip(experimental_cols, loc_imp) if f in HISTORICAL_STATS),
            'State (Time/Score)': sum(imp for f, imp in zip(experimental_cols, loc_imp) if f in STATE_FEATURES),
            'Interaction': sum(imp for f, imp in zip(experimental_cols, loc_imp) if f in INTERACTION)
        }
        
        print(f"\nImportance Aggregates for {slice_name} (LOC):")
        for group, val in agg.items():
            print(f"  {group:20}: {val:.4f}")
            
        return agg
    except Exception as e:
        print(f"  Error: {e}")
        return None

if __name__ == "__main__":
    results = {}
    for ts, name in [('early', 'Early Game (>1800s)'), ('mid', 'Mid Game (600-1800s)'), ('late', 'Late Game (<600s)')]:
        res = run_test(ts, name)
        if res: results[name] = res
        
    # Print final comparison table
    if results:
        print("\n" + "=" * 60)
        print("TEMPORAL COMPARISON: LIVE STATS VS HISTORICAL")
        print("=" * 60)
        stages = list(results.keys())
        groups = ['Live Stats', 'Historical', 'State (Time/Score)', 'Interaction']
        
        header = f"{'Group':20}" + "".join([f"{s:>20}" for s in stages])
        print(header)
        print("-" * len(header))
        for group in groups:
            row = f"{group:20}" + "".join([f"{results[s][group]:>20.4f}" for s in stages])
            print(row)
