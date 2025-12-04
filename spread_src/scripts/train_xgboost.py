"""
Train XGBoost models for spread prediction.

Predicts:
- Mean: Expected score differential
- Std: Uncertainty in differential

Uses XGBoost instead of Ridge for better accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from src.data.database import DatabaseManager
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


def train_xgboost_spread_models(n_models=10):
    """
    Train ensemble of XGBoost models for spread prediction.
    """
    print("=" * 80)
    print("Training XGBoost Spread Models")
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
    
    # Train ensemble
    print(f"\nTraining {n_models} XGBoost model pairs...")
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
        
        # Split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_boot, y_boot, test_size=0.2, random_state=i
        )
        
        # Mean model
        mean_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            random_state=i,
            n_jobs=-1
        )
        
        mean_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict to get residuals
        y_pred = mean_model.predict(X_boot)
        residuals = np.abs(y_boot - y_pred)
        
        # Split residuals
        res_tr, res_val = residuals[X_boot.index.isin(X_tr.index)], residuals[X_boot.index.isin(X_val.index)]
        
        # Std model (predicts absolute residuals)
        std_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            random_state=i,
            n_jobs=-1
        )
        
        std_model.fit(
            X_tr, res_tr,
            eval_set=[(X_val, res_val)],
            verbose=False
        )
        
        ensemble.append({
            'mean_model': mean_model,
            'std_model': std_model
        })
        
        print(f"    Mean: {mean_model.best_iteration} iters, Std: {std_model.best_iteration} iters")
    
    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    print("\n" + "=" * 80)
    print("Ensemble Evaluation")
    print("=" * 80)
    
    test_mean_preds = []
    test_std_preds = []
    
    for model_pair in ensemble:
        mean_pred = model_pair['mean_model'].predict(X_test_features)
        std_pred = model_pair['std_model'].predict(X_test_features)
        test_mean_preds.append(mean_pred)
        test_std_preds.append(std_pred)
    
    ensemble_mean = np.mean(test_mean_preds, axis=0)
    ensemble_std = np.mean(test_std_preds, axis=0)
    
    mae = mean_absolute_error(y_test_diffs, ensemble_mean)
    rmse = np.sqrt(mean_squared_error(y_test_diffs, ensemble_mean))
    
    print(f"\nMean Prediction:")
    print(f"  MAE: {mae:.2f} points")
    print(f"  RMSE: {rmse:.2f} points")
    print(f"\nStd Prediction:")
    print(f"  Mean predicted std: {np.mean(ensemble_std):.2f}")
    print(f"  Actual std of errors: {np.std(y_test_diffs - ensemble_mean):.2f}")
    
    # Calibration
    from scipy import stats
    
    print("\n" + "=" * 80)
    print("Calibration Check")
    print("=" * 80)
    
    thresholds = [-10, -5, 0, 5, 10]
    for threshold in thresholds:
        pred_probs = []
        for mean, std in zip(ensemble_mean, ensemble_std):
            dist = stats.norm(loc=mean, scale=max(std, 1.0))
            prob = 1 - dist.cdf(threshold)
            pred_probs.append(prob)
        
        mean_pred = np.mean(pred_probs)
        actual_freq = np.mean(y_test_diffs > threshold)
        
        print(f"P(diff > {threshold:+.0f}): Predicted {mean_pred:.1%}, Actual {actual_freq:.1%}")
    
    # Save
    model_path = 'models/nba_spread_xgboost.pkl'
    joblib.dump(ensemble, model_path)
    print(f"\n✓ Saved to '{model_path}'")
    print("=" * 80)
    
    return ensemble


if __name__ == "__main__":
    train_xgboost_spread_models(n_models=10)
