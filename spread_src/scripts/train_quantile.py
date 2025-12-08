"""
Train Quantile Regression models for spread prediction.

Directly predicts percentiles of score differential distribution.
No parametric assumptions - learns empirical quantiles.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST


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


def train_quantile_regression_models(n_models=10):
    """
    Train ensemble of quantile regression models.
    
    Each bootstrap iteration trains models for multiple quantiles:
    - 0.10 (10th percentile)
    - 0.25 (25th percentile / Q1)
    - 0.50 (median)
    - 0.75 (75th percentile / Q3)
    - 0.90 (90th percentile)
    """
    print("=" * 80)
    print("Training Quantile Regression Models")
    print("=" * 80)
    
    # Load data
    print("\nLoading training data...")
    X, y_binary = prepare_training_data()
    final_diffs = get_final_score_diffs(X)
    
    print(f"Score diff stats: Mean={np.mean(final_diffs):.2f}, Std={np.std(final_diffs):.2f}")
    
    # Split by game
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    train_mask = X['game_id'].isin(train_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    X_train_game_ids = X[train_mask]['game_id']
    
    # Feature columns
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    X_train_features = X[train_mask][available_cols]
    X_test_features = X[test_mask][available_cols]
    y_train_diffs = final_diffs[train_mask]
    y_test_diffs = final_diffs[test_mask]
    
    print(f"Training: {len(X_train_features)} events, {len(train_ids)} games")
    print(f"Test: {len(X_test_features)} events, {len(test_ids)} games")
    
    # Quantiles to predict
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    # Train ensemble
    print(f"\nTraining {n_models} quantile model sets (5 quantiles each)...")
    ensemble = []
    train_game_ids_unique = X_train_game_ids.unique()
    
    for i in range(n_models):
        print(f"\n  Model set {i+1}/{n_models}...")
        
        # Game-level bootstrap
        boot_game_ids = np.random.choice(train_game_ids_unique, 
                                         len(train_game_ids_unique), 
                                         replace=True)
        boot_mask = X_train_game_ids.isin(boot_game_ids)
        
        X_boot = X_train_features[boot_mask]
        y_boot = y_train_diffs[boot_mask]
        
        # Train a model for each quantile
        quantile_models = {}
        
        for q in quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=i
            )
            
            model.fit(X_boot, y_boot)
            quantile_models[q] = model
        
        ensemble.append(quantile_models)
        print(f"    Trained quantiles: {quantiles}")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Ensemble Evaluation")
    print("=" * 80)
    
    # Get quantile predictions
    quantile_preds = {q: [] for q in quantiles}
    
    for model_set in ensemble:
        for q in quantiles:
            pred = model_set[q].predict(X_test_features)
            quantile_preds[q].append(pred)
    
    # Average across ensemble
    ensemble_quantiles = {q: np.mean(quantile_preds[q], axis=0) for q in quantiles}
    
    # Check quantile ordering (should be monotonic)
    print("\nQuantile Ordering Check:")
    for i in range(len(y_test_diffs)):
        vals = [ensemble_quantiles[q][i] for q in quantiles]
        if not all(vals[j] <= vals[j+1] for j in range(len(vals)-1)):
            print(f"  Warning: Non-monotonic at row {i}: {vals}")
            break
    else:
        print("  ✓ All quantiles properly ordered")
    
    # Calibration: Check if quantiles match empirical frequencies
    print("\n" + "=" * 80)
    print("Calibration Check")
    print("=" * 80)
    
    for q in quantiles:
        # Predicted quantile
        pred_quantile = ensemble_quantiles[q]
        
        # Actual: what % of outcomes are below predicted quantile?
        actual_freq = np.mean(y_test_diffs < pred_quantile)
        
        print(f"Q{int(q*100):02d}: Predicted {q:.0%}, Actual {actual_freq:.1%} " +
              f"(diff: {abs(q - actual_freq):.1%})")
    
    # Median absolute error
    from sklearn.metrics import mean_absolute_error
    
    median_mae = mean_absolute_error(y_test_diffs, ensemble_quantiles[0.50])
    print(f"\nMedian Prediction MAE: {median_mae:.2f} points")
    
    # Probabilistic calibration
    print("\n" + "=" * 80)
    print("Probability Calibration (via interpolation)")
    print("=" * 80)
    
    thresholds = [-10, -5, 0, 5, 10]
    
    for threshold in thresholds:
        # For each test point, interpolate to find P(diff > threshold)
        pred_probs = []
        
        for i in range(len(y_test_diffs)):
            # Get quantile values at this point
            q_vals = [ensemble_quantiles[q][i] for q in quantiles]
            
            # Interpolate to find which quantile corresponds to threshold
            # If threshold is below q10, prob > threshold ≈ 1
            # If threshold is above q90, prob > threshold ≈ 0
            # Otherwise interpolate
            
            if threshold < q_vals[0]:  # Below 10th percentile
                prob = 0.95  # ~95% chance above
            elif threshold > q_vals[-1]:  # Above 90th percentile
                prob = 0.05  # ~5% chance above
            else:
                # Linear interpolation between quantiles
                for j in range(len(quantiles) - 1):
                    if q_vals[j] <= threshold <= q_vals[j+1]:
                        # Interp between quantiles[j] and quantiles[j+1]
                        frac = (threshold - q_vals[j]) / (q_vals[j+1] - q_vals[j] + 1e-8)
                        q_at_threshold = quantiles[j] + frac * (quantiles[j+1] - quantiles[j])
                        prob = 1 - q_at_threshold
                        break
                else:
                    prob = 0.5  # Fallback
            
            pred_probs.append(prob)
        
        mean_pred = np.mean(pred_probs)
        actual_freq = np.mean(y_test_diffs > threshold)
        
        print(f"P(diff > {threshold:+.0f}): Predicted {mean_pred:.1%}, Actual {actual_freq:.1%}")
    
    # Save
    model_path = 'models/nba_spread_quantile.pkl'
    joblib.dump(ensemble, model_path)
    print(f"\n✓ Saved to '{model_path}'")
    print("=" * 80)
    
    return ensemble


if __name__ == "__main__":
    train_quantile_regression_models(n_models=10)
