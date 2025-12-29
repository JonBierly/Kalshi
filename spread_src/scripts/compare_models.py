"""
Compare Ridge vs XGBoost spread prediction models.

Models now predict SCORE REMAINDER (final_diff - current_diff).
We reconstruct final diff at comparison time to evaluate calibration.

Metrics:
- MAE, RMSE (point prediction accuracy)
- Calibration (predicted prob vs actual frequency)
- Inference speed
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import time
import sys
import os
from scipy import stats

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


def load_models():
    """Load all trained models."""
    models = {}
    
    try:
        models['ridge'] = joblib.load('models/nba_spread_model.pkl')
        print("✓ Loaded Ridge models")
    except:
        print("✗ Ridge models not found")
    
    try:
        models['xgboost'] = joblib.load('models/nba_spread_xgboost.pkl')
        print("✓ Loaded XGBoost models")
    except:
        print("✗ XGBoost models not found")
    
    return models


def predict_ridge(models, X, current_diffs):
    """
    Predict using Ridge ensemble.
    
    Models predict REMAINDER. We add current_diff to get reconstructed final diff.
    Supports both variance_model (new) and std_model (legacy) formats.
    """
    remainder_preds = []
    std_preds = []
    
    for model_pair in models:
        remainder_pred = model_pair['mean_model'].predict(X)
        
        # Support both formats
        if 'variance_model' in model_pair:
            variance = model_pair['variance_model'].predict(X)
            std_pred = np.sqrt(np.maximum(variance, 1.0))
        else:
            std_pred = model_pair['std_model'].predict(X)
        
        remainder_preds.append(remainder_pred)
        std_preds.append(std_pred)
    
    mean_remainder = np.mean(remainder_preds, axis=0)
    mean_std = np.mean(std_preds, axis=0)
    
    # Reconstruct final diff
    reconstructed_final_diff = current_diffs + mean_remainder
    
    return reconstructed_final_diff, mean_std, mean_remainder


def predict_xgboost(models, X, current_diffs):
    """
    Predict using XGBoost ensemble.
    
    Models predict REMAINDER. We add current_diff to get reconstructed final diff.
    Supports both variance_model (new) and std_model (legacy) formats.
    """
    remainder_preds = []
    std_preds = []
    
    for model_pair in models:
        remainder_pred = model_pair['mean_model'].predict(X)
        
        # Support both formats
        if 'variance_model' in model_pair:
            variance = model_pair['variance_model'].predict(X)
            std_pred = np.sqrt(np.maximum(variance, 1.0))
        else:
            std_pred = model_pair['std_model'].predict(X)
        
        remainder_preds.append(remainder_pred)
        std_preds.append(std_pred)
    
    mean_remainder = np.mean(remainder_preds, axis=0)
    mean_std = np.mean(std_preds, axis=0)
    
    # Reconstruct final diff
    reconstructed_final_diff = current_diffs + mean_remainder
    
    return reconstructed_final_diff, mean_std, mean_remainder


def compute_prob_gt_threshold(mean, std, threshold):
    """Compute P(diff > threshold) assuming normal."""
    dist = stats.norm(loc=mean, scale=max(std, 1.0))
    return 1 - dist.cdf(threshold)


def compare_models():
    """Compare all models on test set."""
    print("=" * 80)
    print("SPREAD MODEL COMPARISON")
    print("=" * 80)
    
    # Load data
    print("\nLoading test data...")
    X, y_binary = prepare_training_data()
    final_diffs = get_final_score_diffs(X)
    
    # Split
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    test_mask = X['game_id'].isin(test_ids)
    
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    X_test = X[test_mask][available_cols]
    y_test_final_diffs = final_diffs[test_mask]  # Actual final diffs for evaluation
    
    # Get current diffs for reconstruction
    current_diffs = X[test_mask]['score_diff'].values
    
    # Also compute actual remainders for comparing raw model output
    y_test_remainder = y_test_final_diffs - current_diffs
    
    print(f"Test set: {len(X_test)} events, {len(test_ids)} games")
    
    # Load models
    print("\nLoading models...")
    models = load_models()
    
    if not models:
        print("No models found! Please train models first.")
        return
    
    results = {}
    
    # Evaluate each model
    print("\n" + "=" * 80)
    print("POINT PREDICTION ACCURACY")
    print("=" * 80)
    
    for name, model_ensemble in models.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        
        # Predict
        start = time.time()
        
        if name == 'xgboost':
            mean_pred, std_pred, remainder_pred = predict_xgboost(model_ensemble, X_test, current_diffs)
        else:  # ridge
            mean_pred, std_pred, remainder_pred = predict_ridge(model_ensemble, X_test, current_diffs)
        
        inference_time = (time.time() - start) * 1000  # ms
        
        # Metrics on REMAINDER (raw model output)
        remainder_mae = mean_absolute_error(y_test_remainder, remainder_pred)
        remainder_rmse = np.sqrt(mean_squared_error(y_test_remainder, remainder_pred))
        
        # Metrics on RECONSTRUCTED FINAL DIFF (what we actually care about)
        final_mae = mean_absolute_error(y_test_final_diffs, mean_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test_final_diffs, mean_pred))
        
        # Bias
        bias = np.mean(mean_pred - y_test_final_diffs)
        
        # Std calibration
        actual_std = np.std(y_test_final_diffs - mean_pred)
        predicted_std = np.mean(std_pred)
        
        print(f"Remainder MAE:     {remainder_mae:.2f} points (raw model output)")
        print(f"Final Diff MAE:    {final_mae:.2f} points (reconstructed)")
        print(f"Final Diff RMSE:   {final_rmse:.2f} points")
        print(f"Bias:              {bias:+.2f} points")
        print(f"Predicted Std:     {predicted_std:.2f}")
        print(f"Actual Std:        {actual_std:.2f}")
        print(f"Inference:         {inference_time:.1f} ms ({inference_time/len(X_test):.3f} ms/row)")
        
        results[name] = {
            'remainder_mae': remainder_mae,
            'mae': final_mae,
            'rmse': final_rmse,
            'bias': bias,
            'predicted_std': predicted_std,
            'actual_std': actual_std,
            'inference_ms': inference_time,
            'mean_pred': mean_pred,
            'std_pred': std_pred
        }
    
    # Calibration
    print("\n" + "=" * 80)
    print("PROBABILISTIC CALIBRATION")
    print("=" * 80)
    
    thresholds = [-10, -5, 0, 5, 10]
    
    for threshold in thresholds:
        print(f"\nP(diff > {threshold:+.0f}):")
        print(f"  Actual: {np.mean(y_test_final_diffs > threshold):.1%}")
        
        for name in results.keys():
            mean_pred = results[name]['mean_pred']
            std_pred = results[name]['std_pred']
            
            # Parametric probability using reconstructed final diff
            probs = [compute_prob_gt_threshold(m, s, threshold) 
                    for m, s in zip(mean_pred, std_pred)]
            pred_prob = np.mean(probs)
            
            error = abs(pred_prob - np.mean(y_test_final_diffs > threshold))
            print(f"  {name:8s}: {pred_prob:.1%} (error: {error:.1%})")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<12} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Std Cal':>10} {'Speed':>12}")
    print("-" * 70)
    
    for name in results.keys():
        r = results[name]
        std_ratio = r['predicted_std'] / r['actual_std']
        speed_per_row = r['inference_ms'] / len(X_test)
        
        print(f"{name:<12} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['bias']:>+8.2f} "
              f"{std_ratio:>9.2f}x {speed_per_row:>8.3f} ms/row")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best model for each metric
    best_mae = min(results.items(), key=lambda x: x[1]['mae'])
    best_speed = min(results.items(), key=lambda x: x[1]['inference_ms'])
    
    # Calibration score (avg absolute error across thresholds)
    cal_scores = {}
    for name in results.keys():
        errors = []
        for threshold in thresholds:
            mean_pred = results[name]['mean_pred']
            std_pred = results[name]['std_pred']
            
            probs = [compute_prob_gt_threshold(m, s, threshold) 
                    for m, s in zip(mean_pred, std_pred)]
            pred_prob = np.mean(probs)
            
            error = abs(pred_prob - np.mean(y_test_final_diffs > threshold))
            errors.append(error)
        
        cal_scores[name] = np.mean(errors)
    
    best_cal = min(cal_scores.items(), key=lambda x: x[1])
    
    print(f"\nBest Accuracy:    {best_mae[0].upper()} (MAE: {best_mae[1]['mae']:.2f})")
    print(f"Best Calibration: {best_cal[0].upper()} (Avg Error: {best_cal[1]:.1%})")
    print(f"Fastest:          {best_speed[0].upper()} ({best_speed[1]['inference_ms']:.1f} ms total)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_models()
