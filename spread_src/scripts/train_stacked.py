"""
Train stacked spread model (Ridge + XGBoost weighted ensemble).

Uses pre-saved train/val/test splits from prepare_splits.py.

Flow:
1. Load train/val/test CSVs
2. Train Ridge on train set
3. Train XGBoost on train set  
4. Find optimal stacking weights on validation set
5. Evaluate stacked model on test set
6. Save models and stacking config
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.optimize import minimize_scalar
import xgboost as xgb
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST


def load_splits(splits_dir='data/splits'):
    """Load train/val/test splits from CSV files."""
    print("Loading data splits...")
    
    train = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
    test = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    
    # Load feature list
    with open(os.path.join(splits_dir, 'features.txt'), 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    print(f"  Train: {len(train):,} events, {train['game_id'].nunique()} games")
    print(f"  Val:   {len(val):,} events, {val['game_id'].nunique()} games")
    print(f"  Test:  {len(test):,} events, {test['game_id'].nunique()} games")
    print(f"  Features: {len(features)}")
    
    return train, val, test, features


def train_ridge_ensemble(X_train, y_train, feature_cols, n_models=10):
    """Train Ridge ensemble for mean and std prediction."""
    print(f"\nTraining Ridge ensemble ({n_models} models)...")
    
    ensemble = []
    game_ids = X_train['game_id'] if 'game_id' in X_train.columns else None
    
    # Use only the specified feature columns
    available_cols = [c for c in feature_cols if c in X_train.columns]
    X_features = X_train[available_cols].fillna(0)
    
    for i in range(n_models):
        # Game-level bootstrap
        if game_ids is not None:
            unique_games = game_ids.unique()
            boot_games = np.random.choice(unique_games, len(unique_games), replace=True)
            boot_mask = game_ids.isin(boot_games)
            X_boot = X_features[boot_mask]
            y_boot = y_train[boot_mask]
        else:
            idx = np.random.choice(len(X_features), len(X_features), replace=True)
            X_boot = X_features.iloc[idx]
            y_boot = y_train.iloc[idx]
        
        # Calculate sample weights (more weight on late game)
        if 'seconds_remaining' in X_boot.columns:
            seconds = X_boot['seconds_remaining'].values
            weights = np.ones(len(X_boot))
            weights[seconds < 120] = 10.0   # Last 2 min
            weights[(seconds >= 120) & (seconds < 300)] = 5.0  # 2-5 min
            weights[(seconds >= 300) & (seconds < 600)] = 2.0  # 5-10 min
        else:
            weights = None
        
        # Train mean model
        mean_model = Ridge(alpha=1.0, random_state=i)
        mean_model.fit(X_boot, y_boot, sample_weight=weights)
        
        # Train variance model on SQUARED residuals (predicts variance, not std)
        y_pred = mean_model.predict(X_boot)
        squared_residuals = (y_boot - y_pred) ** 2  # Variance estimation
        
        variance_model = Ridge(alpha=1.0, random_state=i)
        variance_model.fit(X_boot, squared_residuals, sample_weight=weights)
        
        ensemble.append({
            'mean_model': mean_model,
            'variance_model': variance_model  # Now predicts variance, not std
        })
        
        if (i + 1) % 5 == 0:
            print(f"    Trained {i+1}/{n_models} models")
    
    return ensemble, available_cols


def train_xgboost_ensemble(X_train, y_train, feature_cols, n_models=10):
    """Train XGBoost ensemble for mean and std prediction."""
    print(f"\nTraining XGBoost ensemble ({n_models} models)...")
    
    ensemble = []
    game_ids = X_train['game_id'] if 'game_id' in X_train.columns else None
    
    # Use only the specified feature columns
    available_cols = [c for c in feature_cols if c in X_train.columns]
    X_features = X_train[available_cols].fillna(0)
    
    for i in range(n_models):
        # Game-level bootstrap
        if game_ids is not None:
            unique_games = game_ids.unique()
            boot_games = np.random.choice(unique_games, len(unique_games), replace=True)
            boot_mask = game_ids.isin(boot_games)
            X_boot = X_features[boot_mask]
            y_boot = y_train[boot_mask]
        else:
            idx = np.random.choice(len(X_features), len(X_features), replace=True)
            X_boot = X_features.iloc[idx]
            y_boot = y_train.iloc[idx]
        
        # Split for early stopping
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_boot, y_boot, test_size=0.2, random_state=i)
        
        # Train mean model
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
        mean_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # Train variance model on SQUARED residuals
        y_pred = mean_model.predict(X_boot)
        squared_residuals = (y_boot - y_pred) ** 2
        
        # Split for variance model
        sq_res_tr = squared_residuals[X_boot.index.isin(X_tr.index)]
        sq_res_val = squared_residuals[X_boot.index.isin(X_val.index)]
        
        variance_model = xgb.XGBRegressor(
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
        variance_model.fit(X_tr, sq_res_tr, eval_set=[(X_val, sq_res_val)], verbose=False)
        
        ensemble.append({
            'mean_model': mean_model,
            'variance_model': variance_model  # Now predicts variance, not std
        })
        
        if (i + 1) % 5 == 0:
            print(f"    Trained {i+1}/{n_models} models")
    
    return ensemble, available_cols


def predict_ensemble(ensemble, X, current_diffs, feature_cols):
    """Get predictions from an ensemble, reconstructing final diff."""
    X_features = X[feature_cols].fillna(0)
    
    remainder_preds = []
    std_preds = []
    
    for model_pair in ensemble:
        remainder = model_pair['mean_model'].predict(X_features)
        # Predict variance, then take sqrt to get std
        variance = model_pair['variance_model'].predict(X_features)
        std = np.sqrt(np.maximum(variance, 1.0))  # Ensure positive, min std=1
        remainder_preds.append(remainder)
        std_preds.append(std)
    
    mean_remainder = np.mean(remainder_preds, axis=0)
    mean_std = np.mean(std_preds, axis=0)
    
    # Reconstruct final diff
    reconstructed = current_diffs + mean_remainder
    
    return reconstructed, mean_std, mean_remainder


def find_optimal_weights(ridge_preds, xgb_preds, y_true):
    """Find optimal stacking weight using validation set."""
    print("\nFinding optimal stacking weights...")
    
    def stacked_mae(alpha):
        stacked = alpha * ridge_preds + (1 - alpha) * xgb_preds
        return mean_absolute_error(y_true, stacked)
    
    result = minimize_scalar(stacked_mae, bounds=(0, 1), method='bounded')
    optimal_alpha = result.x
    
    # Test a few values
    print("\n  Weight | Stacked MAE")
    print("  -------|------------")
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0, optimal_alpha]:
        mae = stacked_mae(alpha)
        marker = " <-- optimal" if abs(alpha - optimal_alpha) < 0.01 else ""
        print(f"  {alpha:.2f}   | {mae:.4f}{marker}")
    
    return optimal_alpha


def evaluate_calibration(predictions, stds, y_true, label="Model"):
    """Evaluate probabilistic calibration."""
    thresholds = [-10, -5, 0, 5, 10]
    errors = []
    
    print(f"\n  {label} Calibration:")
    for threshold in thresholds:
        probs = []
        for pred, std in zip(predictions, stds):
            dist = stats.norm(loc=pred, scale=max(std, 1.0))
            probs.append(1 - dist.cdf(threshold))
        
        pred_prob = np.mean(probs)
        actual_prob = np.mean(y_true > threshold)
        error = abs(pred_prob - actual_prob)
        errors.append(error)
        
        print(f"    P(diff > {threshold:+.0f}): Pred {pred_prob:.1%}, Actual {actual_prob:.1%}, Error {error:.1%}")
    
    return np.mean(errors)


def train_stacked_model(n_models=10):
    """Train stacked Ridge + XGBoost model."""
    print("=" * 80)
    print("TRAINING STACKED SPREAD MODEL")
    print("=" * 80)
    
    # Load splits
    train, val, test, features = load_splits()
    
    # Prepare data
    y_train = train['score_remainder']
    y_val = val['score_remainder']
    y_test = test['score_remainder']
    
    val_current_diffs = val['score_diff'].values
    test_current_diffs = test['score_diff'].values
    
    val_final_diffs = val['final_diff'].values
    test_final_diffs = test['final_diff'].values
    
    # Train Ridge (pass feature list to ensure we use only numeric columns)
    ridge_ensemble, ridge_features = train_ridge_ensemble(train, y_train, features, n_models)
    
    # Train XGBoost
    xgb_ensemble, xgb_features = train_xgboost_ensemble(train, y_train, features, n_models)
    
    # Validate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)
    
    ridge_preds, ridge_stds, _ = predict_ensemble(ridge_ensemble, val, val_current_diffs, ridge_features)
    xgb_preds, xgb_stds, _ = predict_ensemble(xgb_ensemble, val, val_current_diffs, xgb_features)
    
    print(f"\n  Ridge MAE:   {mean_absolute_error(val_final_diffs, ridge_preds):.4f}")
    print(f"  XGBoost MAE: {mean_absolute_error(val_final_diffs, xgb_preds):.4f}")
    
    # Find optimal stacking weights
    optimal_alpha = find_optimal_weights(ridge_preds, xgb_preds, val_final_diffs)
    
    # Create stacked predictions
    stacked_preds = optimal_alpha * ridge_preds + (1 - optimal_alpha) * xgb_preds
    stacked_stds = optimal_alpha * ridge_stds + (1 - optimal_alpha) * xgb_stds
    
    print(f"\n  Optimal weight: {optimal_alpha:.2f} Ridge + {1-optimal_alpha:.2f} XGBoost")
    print(f"  Stacked MAE:   {mean_absolute_error(val_final_diffs, stacked_preds):.4f}")
    
    # Test set evaluation
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    
    ridge_test, ridge_test_stds, _ = predict_ensemble(ridge_ensemble, test, test_current_diffs, ridge_features)
    xgb_test, xgb_test_stds, _ = predict_ensemble(xgb_ensemble, test, test_current_diffs, xgb_features)
    stacked_test = optimal_alpha * ridge_test + (1 - optimal_alpha) * xgb_test
    stacked_test_stds = optimal_alpha * ridge_test_stds + (1 - optimal_alpha) * xgb_test_stds
    
    print(f"\n  Ridge MAE:   {mean_absolute_error(test_final_diffs, ridge_test):.4f}")
    print(f"  XGBoost MAE: {mean_absolute_error(test_final_diffs, xgb_test):.4f}")
    print(f"  Stacked MAE: {mean_absolute_error(test_final_diffs, stacked_test):.4f}")
    
    # Skip detailed calibration output for speed
    # evaluate_calibration(ridge_test, ridge_test_stds, test_final_diffs, "Ridge")
    # evaluate_calibration(xgb_test, xgb_test_stds, test_final_diffs, "XGBoost")
    # evaluate_calibration(stacked_test, stacked_test_stds, test_final_diffs, "Stacked")
    
    # Quick std calibration check
    print(f"\n  Std Calibration:")
    print(f"    Ridge:   Pred={np.mean(ridge_test_stds):.2f}, Actual={np.std(test_final_diffs - ridge_test):.2f}")
    print(f"    XGBoost: Pred={np.mean(xgb_test_stds):.2f}, Actual={np.std(test_final_diffs - xgb_test):.2f}")
    print(f"    Stacked: Pred={np.mean(stacked_test_stds):.2f}, Actual={np.std(test_final_diffs - stacked_test):.2f}")
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(ridge_ensemble, 'models/nba_spread_model.pkl')
    print("  ✓ models/nba_spread_model.pkl (Ridge)")
    
    joblib.dump(xgb_ensemble, 'models/nba_spread_xgboost.pkl')
    print("  ✓ models/nba_spread_xgboost.pkl (XGBoost)")
    
    stacking_config = {
        'ridge_weight': optimal_alpha,
        'xgb_weight': 1 - optimal_alpha,
        'feature_cols': ridge_features
    }
    joblib.dump(stacking_config, 'models/stacking_config.pkl')
    print(f"  ✓ models/stacking_config.pkl (weights: {optimal_alpha:.2f}/{1-optimal_alpha:.2f})")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    
    return ridge_ensemble, xgb_ensemble, stacking_config


if __name__ == "__main__":
    train_stacked_model(n_models=10)
