"""
Train spread distribution models.

Strategy: Predict distribution parameters (mean, std) of final score differential.
Then use these to compute P(diff > threshold) for any threshold.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST


def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """
    For each row in X, get the final score differential for that game.
    
    Returns:
        Array of final score differentials (home - away)
    """
    db = DatabaseManager()
    final_diffs = []
    
    for game_id in X['game_id'].unique():
        # Get final score from database
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
            final_diff = 0  # Fallback
        
        # Replicate for all rows in this game
        game_rows = len(X[X['game_id'] == game_id])
        final_diffs.extend([final_diff] * game_rows)
    
    return np.array(final_diffs)


def train_spread_models(n_models=10):
    """
    Train ensemble of models to predict score differential distribution.
    
    Each model predicts:
        - mean: Expected final score differential
        - std: Uncertainty in score differential
    
    Returns:
        List of model dicts, each containing {'mean_model', 'std_model'}
    """
    print("=" * 80)
    print("Training Spread Distribution Models")
    print("=" * 80)
    
    # 1. Load training data (reuse existing function)
    print("\nLoading training data...")
    X, y_binary = prepare_training_data()
    
    # 2. Get final score differentials for each game
    print("Getting final score differentials...")
    final_diffs = get_final_score_diffs(X)
    
    print(f"Score diff stats:")
    print(f"  Mean: {np.mean(final_diffs):.2f}")
    print(f"  Std: {np.std(final_diffs):.2f}")
    print(f"  Min: {np.min(final_diffs):.0f}, Max: {np.max(final_diffs):.0f}")
    
    # 3. Split by game ID
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    train_mask = X['game_id'].isin(train_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    X_train_game_ids = X[train_mask]['game_id']
    X_test_game_ids = X[test_mask]['game_id']
    
    # 4. Filter to feature columns
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    X_train_features = X[train_mask][available_cols]
    X_test_features = X[test_mask][available_cols]
    y_train_diffs = final_diffs[train_mask]
    y_test_diffs = final_diffs[test_mask]
    
    print(f"\nTraining: {len(X_train_features)} events, {len(train_ids)} games")
    print(f"Test: {len(X_test_features)} events, {len(test_ids)} games")
    
    # 5. Train ensemble
    print(f"\nTraining {n_models} model pairs with time-weighted importance...")
    print("Weight scheme:")
    print("  Last 2 min (< 120s):  10x weight")
    print("  Last 5 min (< 300s):  5x weight")
    print("  Last 10 min (< 600s): 2x weight")
    print("  Rest of game:         1x weight")
    
    ensemble = []
    
    train_game_ids_unique = X_train_game_ids.unique()
    
    for i in range(n_models):
        print(f"\n  Model {i+1}/{n_models}...")
        
        # Game-level bootstrap
        boot_game_ids = np.random.choice(train_game_ids_unique, 
                                         len(train_game_ids_unique), 
                                         replace=True)
        boot_mask = X_train_game_ids.isin(boot_game_ids)
        
        X_boot = X_train_features[boot_mask]
        y_boot = y_train_diffs[boot_mask]
        
        # Calculate sample weights based on seconds_remaining
        # More weight = more important to get right
        seconds_remaining = X_boot['seconds_remaining'].values if 'seconds_remaining' in X_boot.columns else np.zeros(len(X_boot))
        
        sample_weights = np.ones(len(X_boot))
        sample_weights[seconds_remaining < 120] = 10.0  # Last 2 minutes: 10x
        sample_weights[(seconds_remaining >= 120) & (seconds_remaining < 300)] = 5.0  # 2-5 min: 5x
        sample_weights[(seconds_remaining >= 300) & (seconds_remaining < 600)] = 2.0  # 5-10 min: 2x
        
        print(f"    Sample weight distribution:")
        print(f"      10x weight: {np.sum(sample_weights == 10.0)} samples (<2 min)")
        print(f"      5x weight:  {np.sum(sample_weights == 5.0)} samples (2-5 min)")
        print(f"      2x weight:  {np.sum(sample_weights == 2.0)} samples (5-10 min)")
        print(f"      1x weight:  {np.sum(sample_weights == 1.0)} samples (>10 min)")
        
        # Train mean model (Ridge regression for score diff) with weights
        mean_model = Ridge(alpha=1.0, random_state=i)
        mean_model.fit(X_boot, y_boot, sample_weight=sample_weights)
        
        # Predict on bootstrap set to get residuals
        y_pred = mean_model.predict(X_boot)
        residuals = np.abs(y_boot - y_pred)
        
        # Train std model (predict absolute residuals) with weights
        std_model = Ridge(alpha=1.0, random_state=i)
        std_model.fit(X_boot, residuals, sample_weight=sample_weights)
        
        ensemble.append({
            'mean_model': mean_model,
            'std_model': std_model
        })
        
        # Evaluate on bootstrap set
        mean_mae = mean_absolute_error(y_boot, y_pred, sample_weight=sample_weights)
        print(f"    Weighted Mean MAE: {mean_mae:.2f} points")
    
    print(f"\nTrained {len(ensemble)} model pairs")
    
    # 6. Evaluate ensemble on test set
    print("\n" + "=" * 80)
    print("Ensemble Evaluation on Test Set")
    print("=" * 80)
    
    # Get predictions from all models
    test_mean_preds = []
    test_std_preds = []
    
    for model_pair in ensemble:
        mean_pred = model_pair['mean_model'].predict(X_test_features)
        std_pred = model_pair['std_model'].predict(X_test_features)
        test_mean_preds.append(mean_pred)
        test_std_preds.append(std_pred)
    
    test_mean_preds = np.array(test_mean_preds)
    test_std_preds = np.array(test_std_preds)
    
    # Ensemble mean predictions
    ensemble_mean = np.mean(test_mean_preds, axis=0)
    ensemble_std = np.mean(test_std_preds, axis=0)
    
    # Metrics
    mae = mean_absolute_error(y_test_diffs, ensemble_mean)
    rmse = np.sqrt(mean_squared_error(y_test_diffs, ensemble_mean))
    
    print(f"\nMean Prediction:")
    print(f"  MAE: {mae:.2f} points")
    print(f"  RMSE: {rmse:.2f} points")
    
    print(f"\nStd Prediction:")
    print(f"  Mean predicted std: {np.mean(ensemble_std):.2f}")
    print(f"  Actual std of errors: {np.std(y_test_diffs - ensemble_mean):.2f}")
    
    # Test probability calibration
    print("\n" + "=" * 80)
    print("Probability Calibration Check")
    print("=" * 80)
    
    # For each test point, compute P(diff > thresholds) and check calibration
    thresholds = [-10, -5, 0, 5, 10]
    
    from scipy import stats
    
    for threshold in thresholds:
        # Predicted probabilities
        pred_probs = []
        for mean, std in zip(ensemble_mean, ensemble_std):
            dist = stats.norm(loc=mean, scale=max(std, 1.0))
            prob = 1 - dist.cdf(threshold)
            pred_probs.append(prob)
        
        pred_probs = np.array(pred_probs)
        
        # Actual outcomes
        actual = (y_test_diffs > threshold).astype(float)
        
        # Calibration: predicted prob vs actual frequency
        mean_pred = np.mean(pred_probs)
        actual_freq = np.mean(actual)
        
        print(f"P(diff > {threshold:+.0f}): Predicted {mean_pred:.1%}, Actual {actual_freq:.1%}")
    
    # 7. Save ensemble
    print("\n" + "=" * 80)
    model_path = 'models/nba_spread_model.pkl'
    joblib.dump(ensemble, model_path)
    print(f"✓ Saved {len(ensemble)} models to '{model_path}'")
    print("=" * 80)
    
    return ensemble


if __name__ == "__main__":
    ensemble = train_spread_models(n_models=10)
