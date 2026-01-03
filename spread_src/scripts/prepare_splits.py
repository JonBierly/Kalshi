"""
Prepare training data splits for model training.

Creates train/val/test CSVs that can be reused across different models.
This ensures consistent splits for fair comparison and stacking.

Split strategy:
- Train: 60% of games (for training base models)
- Val: 20% of games (for stacking weight optimization)
- Test: 20% of games (for final evaluation)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST


def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """Get final score differential for each game."""
    db = DatabaseManager()
    final_diffs = []
    
    for game_id in X['game_id'].unique():
        query = f"""
            SELECT home_score, away_score 
            FROM pbp_events 
            WHERE game_id = '{game_id}' 
            ORDER BY period DESC, remaining_time ASC 
            LIMIT 1
        """
        result = pd.read_sql(query, db.engine)
        
        if not result.empty:
            final_diff = result['home_score'].iloc[0] - result['away_score'].iloc[0]
        else:
            final_diff = 0
        
        game_rows = len(X[X['game_id'] == game_id])
        final_diffs.extend([final_diff] * game_rows)
    
    return np.array(final_diffs)


def prepare_splits(output_dir='data/splits', random_state=42):
    """
    Prepare and save train/val/test splits as CSVs.
    
    Returns:
        Dict with paths to saved files
    """
    print("=" * 80)
    print("PREPARING DATA SPLITS")
    print("=" * 80)
    
    # Load data
    print("\nLoading training data...")
    X, y_binary = prepare_training_data()
    
    print("\nGetting final score differentials...")
    final_diffs = get_final_score_diffs(X)
    
    # Compute score remainder (target)
    current_diffs = X['score_diff'].values
    score_remainders = final_diffs - current_diffs
    
    # Add targets to dataframe
    X['final_diff'] = final_diffs
    X['score_remainder'] = score_remainders
    
    print(f"\nTotal: {len(X)} events, {X['game_id'].nunique()} games")
    print(f"Score remainder stats: Mean={np.mean(score_remainders):.2f}, Std={np.std(score_remainders):.2f}")
    
    # Split by game ID (60/20/20)
    game_ids = X['game_id'].unique()
    
    # First split: 60% train, 40% temp
    train_ids, temp_ids = train_test_split(
        game_ids, test_size=0.4, random_state=random_state
    )
    
    # Second split: 50% of temp = 20% val, 50% of temp = 20% test
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=random_state
    )
    
    # Create masks
    train_mask = X['game_id'].isin(train_ids)
    val_mask = X['game_id'].isin(val_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train):,} events, {len(train_ids)} games ({len(train_ids)/len(game_ids):.0%})")
    print(f"  Val:   {len(X_val):,} events, {len(val_ids)} games ({len(val_ids)/len(game_ids):.0%})")
    print(f"  Test:  {len(X_test):,} events, {len(test_ids)} games ({len(test_ids)/len(game_ids):.0%})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"\nSaving to {output_dir}/...")
    X_train.to_csv(train_path, index=False)
    print(f"  ✓ train.csv ({len(X_train):,} rows)")
    
    X_val.to_csv(val_path, index=False)
    print(f"  ✓ val.csv ({len(X_val):,} rows)")
    
    X_test.to_csv(test_path, index=False)
    print(f"  ✓ test.csv ({len(X_test):,} rows)")
    
    # Save feature list for reference
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    with open(os.path.join(output_dir, 'features.txt'), 'w') as f:
        f.write('\n'.join(available_cols))
    print(f"  ✓ features.txt ({len(available_cols)} features)")
    
    print("\n" + "=" * 80)
    print("Done! Use these files for training:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path} (for stacking weight optimization)")
    print(f"  Test:  {test_path} (for final evaluation)")
    print("=" * 80)
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'features': available_cols
    }


if __name__ == "__main__":
    prepare_splits()
