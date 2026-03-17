"""
Script to Calibrate the NBA Spread Model using K-Fold Cross-Validation.
This allows us to calibrate the probabilities on the *entire* historical dataset 
without data leakage, bypassing the need for Kalshi market data.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from betacal import BetaCalibration
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import t

# Ensure we're running from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spread_src.models.distributions import SafeT
from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import add_interaction_features
from spread_src.scripts.train_ngboost_model import get_final_score_diffs

from ngboost import NGBRegressor
from ngboost.scores import LogScore

# Fast parameters for CV
N_ESTIMATORS = 200 # Faster training for CV calibration

def run_cv_calibration():
    print("=" * 80)
    print("Cross-Validation Beta Calibration (Historical Data)")
    print("=" * 80)
    
    # 1. Load full training data
    print("Loading historical data (this will take a moment)...")
    # Quick=True for testing, use False for full prod calibration
    X, _ = prepare_training_data(num_games=1000)
    
    print("Getting absolute final score differentials...")
    final_diffs = get_final_score_diffs(X)
    
    # Target is Score Remainder: (Final Home Score - Final Away Score) - Current Score Diff
    current_diffs = X['score_diff'].values
    score_remainders = final_diffs - current_diffs
    
    print("Adding interaction features...")
    X = add_interaction_features(X)
    
    # 2. Filter to Trading Window Only (Don't calibrate on garbage time)
    trading_window_mask = (X['seconds_remaining'] >= 120) & (X['seconds_remaining'] <= 2580)
    
    X_window = X[trading_window_mask]
    y_window = score_remainders[trading_window_mask]
    
    # We only care about calibrating the "clutch" high-impact probabilities
    # Let's filter to the target window (last 10 minutes)
    late_game_mask = (X_window['seconds_remaining'] <= 600)
    
    # Final data we'll use for CV
    game_ids = X_window['game_id'].unique()
    print(f"\nFiltered to {len(game_ids)} unique games in the dataset.")
    print(f"Total calibration events (last 10 mins): {late_game_mask.sum()}")
    
    from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, INTERACTION_FEATURES_LIST
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST + INTERACTION_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X_window.columns]

    print(f"\nSetting up 5-Fold Cross Validation... (Game-Level)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    out_of_sample_preds = []
    out_of_sample_scales = []
    out_of_sample_truths = []
    
    fold = 1
    for train_games_idx, test_games_idx in kf.split(game_ids):
        print(f"\n--- Fold {fold}/5 ---")
        train_ids = game_ids[train_games_idx]
        test_ids = game_ids[test_games_idx]
        
        # Mask for this fold
        fold_train_mask = X_window['game_id'].isin(train_ids)
        fold_test_mask = X_window['game_id'].isin(test_ids) & late_game_mask
        
        X_train = X_window[fold_train_mask][available_cols]
        y_train = y_window[fold_train_mask]
        
        X_test = X_window[fold_test_mask][available_cols]
        y_test = y_window[fold_test_mask]
        
        if len(X_test) == 0:
            print("  Skipping (No late game events in test fold)")
            continue
            
        print(f"  Training on {len(X_train)} events, predicting {len(X_test)} out-of-sample late-game events...")
        
        # Train a slightly faster single NGBoost model (we don't need a full ensemble just to calibrate the distribution shape)
        base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=30)
        model = NGBRegressor(
            Dist=SafeT,
            Score=LogScore,
            Base=base,
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            verbose=False,
            random_state=fold
        )
        
        # Fit on training fold
        model.fit(X_train.values, y_train.values)
        
        # Predict on holdout fold
        dist = model.pred_dist(X_test.values)
        out_of_sample_preds.extend(dist.loc)
        out_of_sample_scales.extend(dist.scale)
        
        # Did the home team outscore the away team from this point forward?
        target_binary = (y_test > 0).astype(int).values
        out_of_sample_truths.extend(target_binary)
        
        fold += 1
        
    print("\n" + "=" * 80)
    print("Cross-Validation Complete! Calculating raw probabilities...")
    
    # 3. Calculate raw probabilities for the entire "out of sample" dataset
    locs = np.array(out_of_sample_preds)
    scales = np.array(out_of_sample_scales)
    y_true = np.array(out_of_sample_truths)
    
    # CDF(0) is prob remainder < 0. So 1 - CDF is prob remainder > 0
    raw_probs = 1.0 - t.cdf(0, df=3, loc=locs, scale=scales)
    
    raw_brier = brier_score_loss(y_true, raw_probs)
    print(f"Raw NGBoost CV Brier Score: {raw_brier:.4f}")
    
    # 4. Fit Beta Calibrator
    print("\nFitting Beta Calibrator on ALL out-of-sample predictions...")
    calibrator = BetaCalibration(parameters="abm")
    
    # Avoid exact 0s and 1s
    safe_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    
    calibrator.fit(safe_probs, y_true)
    calibrated_probs = calibrator.predict(safe_probs)
    
    calibrated_brier = brier_score_loss(y_true, calibrated_probs)
    print(f"Calibrated CV Brier Score:  {calibrated_brier:.4f}")
    print(f"Improvement: {(raw_brier - calibrated_brier) / raw_brier * 100:.2f}%")
    
    a, b, m = calibrator.map_[-3:] if hasattr(calibrator, 'map_') else (0,0,0)
    print(f"\nRobust BetaCal Parameters (a, b, m): {a:.4f}, {b:.4f}, {m:.4f}")
    
    # 5. Plot and Save
    from sklearn.calibration import calibration_curve
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, safe_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=10)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    plt.plot(prob_pred_raw, prob_true_raw, marker='o', label=f'Raw NGBoost (Brier: {raw_brier:.3f})', color='salmon', linewidth=2)
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', label=f'Beta Calibrated (Brier: {calibrated_brier:.3f})', color='dodgerblue', linewidth=2)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Probability (Fraction of Positives)', fontsize=12)
    plt.title(f'Robust CV Reliability Diagram (N = {len(y_true)} events)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs('reports/calibration', exist_ok=True)
    plt.savefig('reports/calibration/cv_beta_calibration_curve.png', dpi=150, bbox_inches='tight')
    
    joblib.dump(calibrator, 'models/robust_beta_calibrator.pkl')
    print("\n✓ Saved robust Beta Calibrator to models/robust_beta_calibrator.pkl")
    print("✓ Saved CV calibration curve to reports/calibration/cv_beta_calibration_curve.png")
    
    print("\nRobust Calibration Adjustments (Extremes):")
    test_probs = np.array([0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95])
    cal_test_probs = calibrator.predict(test_probs)
    for r, c in zip(test_probs, cal_test_probs):
        print(f"  Raw: {r*100:5.1f}%  ->  Calibrated: {c*100:5.1f}%   (Diff: {(c-r)*100:+.1f}%)")

if __name__ == "__main__":
    run_cv_calibration()
