"""
Script to analyze the calibration of the Kalshi Spread Trading model
and fit a Beta Calibrator using recent (out-of-sample) games.

The current production model (`nba_spread_ngboost_v3_final.pkl`) was
trained on games prior to Feb 10th, 2026. This script analyzes the
model's performance on games played *after* this date and applies
Beta Calibration to adjust its probability outputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from betacal import BetaCalibration
from sklearn.metrics import brier_score_loss, log_loss

# Ensure we're running from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spread_src.models.distributions import SafeT
from src.data.database import DatabaseManager
from spread_src.features.engineering import add_advanced_features, create_live_features, add_interaction_features

# Dates for holdout set
MODEL_TRAINED_DATE = '2026-02-10'

def load_holdout_data():
    """Load the last 300 games from the current season."""
    db = DatabaseManager()
    
    # Get games from Feb 10th onwards
    query_games = """
        SELECT game_id, date
        FROM games 
        WHERE date >= '2026-02-10'
        ORDER BY date DESC
    """
    recent_games = pd.read_sql(query_games, db.engine)
    game_ids = recent_games['game_id'].tolist()
    
    print(f"Found {len(game_ids)} recent games in the dataset.")
    
    frames = []
    # Process game by game to avoid massive memory footprint
    for idx, g in enumerate(game_ids):
        query_pbp = f"""
            SELECT * FROM pbp_events 
            WHERE game_id = '{g}'
        """
        pbp_df = pd.read_sql(query_pbp, db.engine)
        
        if pbp_df.empty:
            continue
            
        # Get team IDs
        query_teams = f"SELECT home_team_id, away_team_id FROM games WHERE game_id = '{g}'"
        teams = pd.read_sql(query_teams, db.engine)
        pbp_df['home_team_id'] = teams['home_team_id'].iloc[0]
        pbp_df['away_team_id'] = teams['away_team_id'].iloc[0]
        
        # Calculate features (this mimics training prep)
        feat_df = create_live_features(pbp_df)
        
        # Determine actual outcome (did home team ultimately cover the spread required?)
        # For simplicity in this analysis, we'll try to predict if the home team wins outright
        # Note: In the actual spread model, we predict the *score remainder*, not a binary outcome.
        # But to calibrate probabilities, we need a binary outcome.
        
        # Let's define the binary outcome as: Does the Home Team increase its lead?
        # Target = 1 if (Final Home Score - Final Away Score) > Current Score Diff
        
        final_row = pbp_df.iloc[-1]
        final_home_score = final_row['home_score']
        final_away_score = final_row['away_score']
        final_diff = final_home_score - final_away_score
        
        feat_df['actual_final_diff'] = final_diff
        
        # We only want to analyze the core trading window (5 mins left in 1Q to 2 mins left in 4Q)
        # Total game is 2880 seconds.
        # 5 mins left in 1st quarter = 2580 seconds remaining
        # 2 mins left in 4th quarter = 120 seconds remaining
        trading_window_mask = (feat_df['seconds_remaining'] <= 2580) & (feat_df['seconds_remaining'] >= 120)
        feat_df = feat_df[trading_window_mask]
        
        frames.append(feat_df)
        
        if (idx+1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(game_ids)} games...")
            
    if not frames:
        return pd.DataFrame()
        
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nExtracted {len(combined)} core trading window situations for calibration.")
    
    # Add advanced features (like model training)
    print("Generating advanced features...")
    combined = add_advanced_features(combined)
    
    # Add interaction features
    print("Adding interaction features...")
    combined = add_interaction_features(combined)
    
    # Create the binary target: Did the score remainder end up being > 0 ?
    # i.e., did the home team perform better than the current tie from here to the end?
    # This roughly correlates to "does the current spread hold"
    combined['target_remainder'] = combined['actual_final_diff'] - combined['score_diff']
    
    return combined

def analyze_and_calibrate(df):
    """Run model predictions, analyze calibration, and fit Beta calibrator."""
    if df.empty:
        print("No holdout data available.")
        return
        
    try:
        model_data = joblib.load('models/nba_spread_ngboost_v3_final.pkl')
        ensemble = model_data['ensemble']
        feature_order = model_data['feature_order']
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print(f"\nModel loaded (Ensemble size: {len(ensemble)})")
    
    # Ensure all features exist
    available_cols = [c for c in feature_order if c in df.columns]
    X = df[available_cols].fillna(0)
    
    # Get predictions
    print("Generating predictions on holdout set...")
    all_locs = []
    all_scales = []
    
    for model in ensemble:
        dist = model.pred_dist(X.values)
        all_locs.append(dist.loc)
        all_scales.append(dist.scale)
        
    ensemble_loc = np.mean(all_locs, axis=0)
    ensemble_scale = np.mean(all_scales, axis=0)
    
    # Total uncertainty (aleatoric + epistemic)
    epistemic_var = np.var(all_locs, axis=0)
    total_scale = np.sqrt(ensemble_scale**2 + epistemic_var)
    
    # The model outputs a T-distribution with df=3
    # params[0] = loc, params[1] = scale
    locs = np.array(all_locs).mean(axis=0)
    scales = np.array(all_scales).mean(axis=0)
    epistemic_var = np.var(all_locs, axis=0)
    total_scale = np.sqrt(scales**2 + epistemic_var)
    
    print("\nCalculating raw probabilities across spread thresholds...")
    
    # Thresholds to evaluate
    thresholds = [-15.5, -10.5, -5.5, -1.5, 1.5, 5.5, 10.5, 15.5]
    
    plt.figure(figsize=(12, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    from scipy.stats import t
    
    overall_raw_briers = []
    
    # We will fit a single calibrator using all the target thresholds combined
    # to learn the fundamental overconfidence of the distribution shape
    all_raw_probs = []
    all_targets = []
    score_remainders = df['target_remainder'].values
    
    print("\nRaw Model Metrics per Threshold (Lower Brier is better):")
    for i, thresh in enumerate(thresholds):
        # Target: Did the home team cover this specific spread threshold?
        # i.e., Score Remainder > thresh
        target_binary = (score_remainders > thresh).astype(int)
        
        # Calculate raw probability from T-distribution for this threshold
        # CDF(thresh) is prob remainder < thresh. 
        # So 1 - CDF is prob remainder > thresh
        raw_probs = 1.0 - t.cdf(thresh, df=3, loc=locs, scale=total_scale)
        
        # Store for combined calibration tuning
        all_raw_probs.extend(raw_probs)
        all_targets.extend(target_binary)
        
        raw_brier = brier_score_loss(target_binary, raw_probs)
        overall_raw_briers.append(raw_brier)
        print(f"  >{thresh:>6}: {raw_brier:.4f} (Win Prob: {np.mean(raw_probs):.2%})")
        
        # Plot raw reliability curve for this threshold
        from sklearn.calibration import calibration_curve
        prob_true_raw, prob_pred_raw = calibration_curve(target_binary, raw_probs, n_bins=10)
        plt.plot(prob_pred_raw, prob_true_raw, marker='o', markersize=4, linestyle='-', alpha=0.6,
                 label=f'Raw >{thresh} (Brier: {raw_brier:.3f})', color=colors[i])

    print(f"\nAverage Raw Brier Score: {np.mean(overall_raw_briers):.4f}")
    
    # --- Fit a Unified Calibrator ---
    print("\nFitting unified Beta Calibrator across all thresholds...")
    calibrator = BetaCalibration(parameters="abm")
    
    all_raw_probs = np.array(all_raw_probs)
    all_targets = np.array(all_targets)
    all_raw_probs_clipped = np.clip(all_raw_probs, 1e-6, 1.0 - 1e-6)
    
    calibrator.fit(all_raw_probs_clipped, all_targets)
    all_cal_probs = calibrator.predict(all_raw_probs_clipped)
    
    overall_cal_brier = brier_score_loss(all_targets, all_cal_probs)
    overall_raw_brier = brier_score_loss(all_targets, all_raw_probs)
    print(f"\nUnified Calibrated Metrics:")
    print(f"Unified Brier Score: {overall_cal_brier:.4f} (was {overall_raw_brier:.4f})")
    print(f"Overall Improvement: {(overall_raw_brier - overall_cal_brier) / overall_raw_brier * 100:.2f}%")
    
    a, b, m = calibrator.map_[-3:] if hasattr(calibrator, 'map_') else (0,0,0)
    print(f"\nBetaCal Parameters (a, b, m): {a:.4f}, {b:.4f}, {m:.4f}")
    
    # Plot the unified calibrated curve (Avg behavior across all spreads)
    prob_true_cal, prob_pred_cal = calibration_curve(all_targets, all_cal_probs, n_bins=10)
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', markersize=8, linewidth=3,
             label=f'UNIFIED CALIBRATED (Brier: {overall_cal_brier:.3f})', color='black')
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Probability (Fraction of Positives)', fontsize=12)
    plt.title(f'Multi-Spread Reliability Diagram (50 Games)', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True, alpha=0.3)
    
    os.makedirs('reports/calibration', exist_ok=True)
    plt.savefig('reports/calibration/multi_spread_beta_calibration_curve.png', dpi=150, bbox_inches='tight')
    
    joblib.dump(calibrator, 'models/beta_calibrator_v1.pkl')
    print("\n✓ Saved unified calibration curve to reports/calibration/multi_spread_beta_calibration_curve.png")
    print("✓ Saved tuned Beta Calibrator to models/beta_calibrator_v1.pkl")

if __name__ == "__main__":
    print(f"--- Beta Calibration Analysis ---")
    df = load_holdout_data()
    analyze_and_calibrate(df)
