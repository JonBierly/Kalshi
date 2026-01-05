#!/usr/bin/env python
"""
Evaluate NGBoost model accuracy on the test set.
Calculates Brier Score, Log Loss, and calibration metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.model_selection import train_test_split
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from spread_src.features.engineering import (
    BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, INTERACTION_FEATURES_LIST,
    add_interaction_features
)
from spread_src.models.spread_model import SpreadDistributionModel

def calculate_brier_score(y_true, y_prob):
    """Calculate Brier Score for binary outcomes."""
    return np.mean((y_true - y_prob)**2)

def evaluate_accuracy():
    print("=" * 80)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 80)

    # 1. Load Data
    print("\nLoading data and recreating test split...")
    X, y_binary = prepare_training_data()
    
    # Identify unique games for split
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    test_mask = X['game_id'].isin(test_ids)
    X_test_all = X[test_mask]
    
    # 2. Add Interaction Features
    X_test_all = add_interaction_features(X_test_all)
    
    # 3. Get Ground Truth Final Differentials
    from spread_src.scripts.train_ngboost_model import get_final_score_diffs
    print("Fetching final score differentials...")
    final_diffs = get_final_score_diffs(X_test_all)
    
    # 4. Load Model
    model_path = 'models/nba_spread_ngboost.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading ensemble from {model_path}...")
    model_data = joblib.load(model_path)
    ensemble = model_data['ensemble']
    feature_cols = model_data['feature_order']
    
    X_test_features = X_test_all[feature_cols].values
    current_diffs = X_test_all['score_diff'].values
    
    # 5. Get Predictions
    print("\nGenerating predictions for test set...")
    all_remainders = []
    all_scales = []
    
    for model in ensemble:
        dist = model.pred_dist(X_test_features)
        all_remainders.append(dist.loc)
        all_scales.append(dist.scale)
    
    ensemble_remainder = np.mean(all_remainders, axis=0)
    ensemble_scale = np.mean(all_scales, axis=0) # Total aleatoric
    epistemic_std = np.std(all_remainders, axis=0)
    total_std = np.sqrt(ensemble_scale**2 + epistemic_std**2)
    
    predicted_final_diff = current_diffs + ensemble_remainder
    
    # 6. Metrics: Directional (Home Win)
    print("\n" + "-" * 40)
    print("Directional Metrics (Home Win)")
    print("-" * 40)
    
    # Actual winner
    actual_winner = (final_diffs > 0).astype(int)
    
    # Predicted probability of home win
    # Using the normal distribution approximation for the ensemble output
    from scipy.stats import norm
    # P(Final Diff > 0) = P(Current Diff + Remainder > 0) = P(Remainder > -Current Diff)
    # = 1 - CDF_remainder(-Current Diff)
    # We use norm.sf which is 1 - CDF
    z_scores = (-current_diffs - ensemble_remainder) / total_std
    prob_home_win = norm.sf(z_scores)
    
    brier_dir = brier_score_loss(actual_winner, prob_home_win)
    logloss_dir = log_loss(actual_winner, prob_home_win)
    acc_dir = np.mean((prob_home_win > 0.5) == actual_winner)
    
    print(f"Brier Score:  {brier_dir:.4f}")
    print(f"Log Loss:     {logloss_dir:.4f}")
    print(f"Accuracy:     {acc_dir:.2%}")
    
    # 7. Metrics: Spread Accuracy (Brier Score across thresholds)
    print("\n" + "-" * 40)
    print("Spread Threshold Metrics (Brier Score)")
    print("-" * 40)
    
    thresholds = [-17.5, -12.5, -7.5, -4.5, -2.5, 0.5, 2.5, 4.5, 7.5, 12.5, 17.5]
    print(f"{'Threshold':<12} {'Brier':<10} {'MAE (Prob)':<10}")
    
    for t in thresholds:
        # Actual: Did final_diff exceed threshold?
        actual_exceeded = (final_diffs > t).astype(int)
        
        # Predicted probability
        # P(Final Diff > t) = P(Remainder > t - Current Diff)
        z = (t - current_diffs - ensemble_remainder) / total_std
        prob_exceeded = norm.sf(z)
        
        b = brier_score_loss(actual_exceeded, prob_exceeded)
        mae_p = np.mean(np.abs(actual_exceeded - prob_exceeded))
        
        print(f"{t:<12.1f} {b:<10.4f} {mae_p:<10.4f}")

    # 8. MAE / RMSE (Continuous)
    print("\n" + "-" * 40)
    print("Continuous Metrics (Point Diff)")
    print("-" * 40)
    mae = mean_absolute_error(final_diffs, predicted_final_diff)
    rmse = np.sqrt(mean_squared_error(final_diffs, predicted_final_diff))
    print(f"MAE:  {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    evaluate_accuracy()
