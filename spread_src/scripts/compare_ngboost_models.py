"""
Compare two NGBoost ensemble models for NBA spread prediction.

Focuses on:
- Point prediction accuracy (MAE, RMSE)
- Probabilistic accuracy (Brier Score, Log-Likelihood)
- Calibration (PIT histogram, Quantile coverage)

Supports both test set evaluation and table-based evaluation (model_predictions).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss
import joblib
import time
import sys
import os
import json
from scipy import stats
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import add_interaction_features, TeamStatsEngine, RosterEngine
from spread_src.models.distributions import SafeT
from spread_src.data.spread_markets import parse_spread_ticker

def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """Get final score differential for each game in X."""
    db = DatabaseManager()
    final_diffs_dict = {}
    unique_game_ids = X['game_id'].unique()
    
    for game_id in unique_game_ids:
        query = f"SELECT home_score, away_score FROM pbp_events WHERE game_id = '{game_id}' ORDER BY period DESC, remaining_time ASC LIMIT 1"
        result = pd.read_sql(query, db.engine)
        if not result.empty:
            final_diff = result['home_score'].iloc[0] - result['away_score'].iloc[0]
        else:
            final_diff = 0
        final_diffs_dict[game_id] = final_diff
    
    return X['game_id'].map(final_diffs_dict).values

def get_ensemble_predictions(models, X_values):
    """Get mean and scale predictions from an NGBoost ensemble."""
    all_means = []
    all_scales = []
    for model in models:
        dist = model.pred_dist(X_values)
        all_means.append(dist.loc)
        all_scales.append(dist.scale)
    
    mean_pred = np.mean(all_means, axis=0)
    ensemble_scale = np.mean(all_scales, axis=0)
    epistemic_std = np.std(all_means, axis=0)
    total_std = np.sqrt(ensemble_scale**2 + epistemic_std**2)
    return mean_pred, total_std

def calculate_brier_score(mean, std, actual_diff, threshold, dist_type='normal'):
    """Calculate Brier score for a specific threshold."""
    # P(actual_diff > threshold)
    # Using Normal approximation for the ensemble distribution
    # Note: If dist_type was SafeT, we'd ideally use that, but Normal is a good approximation for the ensemble spread.
    prob_gt = 1 - stats.norm.cdf(threshold, loc=mean, scale=np.maximum(std, 1.0))
    actual_gt = (actual_diff > threshold).astype(int)
    return (prob_gt - actual_gt)**2

def evaluate_on_table(model_paths, limit=None):
    """Evaluate models based on the model_predictions table."""
    print("\n" + "=" * 80)
    print("EVALUATING MODELS ON PREDICTIONS TABLE")
    print("=" * 80)
    
    db = DatabaseManager()
    query = "SELECT * FROM model_predictions"
    if limit:
        query += f" LIMIT {limit}"
    
    preds_df = pd.read_sql(query, db.engine)
    if preds_df.empty:
        print("No predictions found in the table.")
        return
    
    print(f"Loaded {len(preds_df)} predictions from the table.")
    
    # Get final scores for ground truth
    final_scores_query = "SELECT game_id, home_score, away_score FROM games"
    final_scores_df = pd.read_sql(final_scores_query, db.engine)
    final_scores_map = {row['game_id']: (row['home_score'], row['away_score']) for _, row in final_scores_df.iterrows()}
    
    # Initialize engines for missing features
    print("Initializing feature engines for recalculation...")
    team_engine = TeamStatsEngine()
    roster_engine = RosterEngine()
    
    results = {}
    
    # We'll evaluate one model at a time to save memory
    for name, path in model_paths.items():
        print(f"\nEvaluating {name}...")
        model_data = joblib.load(path)
        ensemble = model_data.get('ensemble')
        feature_order = model_data.get('feature_order')
        
        all_brier_scores = []
        all_processed_features = []
        all_ground_truth_outcomes = []
        
        for _, row in preds_df.iterrows():
            gid = row['game_id']
            if gid not in final_scores_map:
                continue
            
            # 1. Parse Ticker to get threshold
            try:
                parsed_ticker = parse_spread_ticker(row['ticker'])
                threshold = parsed_ticker['spread_value']
                # Ticker is usually for a specific team. 
                # "UTA3" means UTA wins by > 3.5. 
                # UTA is either home or away.
                home_score, away_score = final_scores_map[gid]
                
                # Re-calculate outcome relative to the ticker team
                # Default is usually Home - Away for the model's 'mean'
                # But Brier score on a ticker needs to know if THAT ticker hit.
                if parsed_ticker['spread_team'] == parsed_ticker['home_team']:
                    actual_spread = home_score - away_score
                else:
                    actual_spread = away_score - home_score
                
                actual_outcome = 1 if actual_spread > threshold else 0
                
            except Exception as e:
                # print(f"Error parsing ticker {row['ticker']}: {e}")
                continue
                
            # 2. Re-calculate Features
            # Start with existing features
            orig_features = json.loads(row['features_json'])
            
            # Fetch "fresh" advanced features from engine for this game
            home_id = orig_features.get('home_team_id')
            away_id = orig_features.get('away_team_id')
            
            # Use engines to get latest features (including potential new ones)
            # Both engines use pre-calculated caches internally for historical games
            adv_team_feats = team_engine.get_features(gid, home_id, away_id)
            adv_roster_feats = roster_engine.get_features(gid, home_id, away_id)
            
            # Merge: Situational (JSON) + Fresh Advanced (Engines)
            # Situational features are unique per row, Advanced are unique per game.
            full_feats = {**orig_features, **adv_team_feats, **adv_roster_feats}
            
            # Predicted prob for THIS row
            # We need the model's mean/std for the WHOLE game (Home-Away)
            # and then we map it to the specific threshold of the ticker.
            
            all_processed_features.append(full_feats)
            all_ground_truth_outcomes.append({'mean_target': home_score - away_score, 
                                             'ticker_threshold': threshold, 
                                             'is_home': parsed_ticker['spread_team'] == parsed_ticker['home_team'],
                                             'actual_outcome': actual_outcome})

        if not all_processed_features:
            continue
            
        # Batch Predict
        X_eval = pd.DataFrame(all_processed_features)
        X_eval = add_interaction_features(X_eval)
        
        # Align features
        missing = [f for f in feature_order if f not in X_eval.columns]
        for f in missing:
            X_eval[f] = 0
        X_eval = X_eval[feature_order].values
        
        means, stds = get_ensemble_predictions(ensemble, X_eval)
        
        # Calculate Brier Scores and group by stage
        row_brier_scores = []
        stage_brier = {'Early': [], 'Mid': [], 'Late': []}
        
        for i, gt in enumerate(all_ground_truth_outcomes):
            # Model predicts Home - Away
            mean_ha = means[i]
            std_ha = stds[i]
            
            # Map to ticker
            if not gt['is_home']:
                # Away - Home
                mean_ticker = -mean_ha
            else:
                mean_ticker = mean_ha
            
            # Predict P(ticker > threshold)
            # REFINEMENT: Use a much smaller floor for uncertainty (0.1 instead of 1.0)
            # to allow for more confidence late in games.
            prob_gt = 1 - stats.norm.cdf(gt['ticker_threshold'], loc=mean_ticker, scale=np.maximum(std_ha, 0.1))
            bs = (prob_gt - gt['actual_outcome'])**2
            row_brier_scores.append(bs)
            
            # Group by stage
            sec = all_processed_features[i]['seconds_remaining']
            if sec > 1800: stage = 'Early'
            elif sec > 600: stage = 'Mid'
            else: stage = 'Late'
            stage_brier[stage].append(bs)
            
        avg_brier = np.mean(row_brier_scores)
        results[name] = {
            'Brier Score': avg_brier, 
            'Count': len(row_brier_scores),
            'Stages': {k: np.mean(v) if v else 0 for k, v in stage_brier.items()}
        }
        print(f"  ✓ {name} Brier Score: {avg_brier:.4f}")

    # Calculate Market Brier Score by stage
    market_brier_scores = []
    market_stage_brier = {'Early': [], 'Mid': [], 'Late': []}
    
    for idx, gt, feats in zip(preds_df.index, all_ground_truth_outcomes, all_processed_features):
        mid_price = (preds_df.loc[idx, 'bid_price'] + preds_df.loc[idx, 'ask_price']) / 200.0
        bs = (mid_price - gt['actual_outcome'])**2
        market_brier_scores.append(bs)
        
        sec = feats['seconds_remaining']
        if sec > 1800: stage = 'Early'
        elif sec > 600: stage = 'Mid'
        else: stage = 'Late'
        market_stage_brier[stage].append(bs)
    
    avg_market_brier = np.mean(market_brier_scores)
    m_stages = {k: np.mean(v) if v else 0 for k, v in market_stage_brier.items()}
    
    # Print Summary
    model_names = list(results.keys())
    header = f"{'Metric':<20}"
    for name in model_names:
        header += f" | {name:>15}"
    header += f" | {'Market':>15}"
    
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    
    # Overall Brier
    row = f"{'Overall Brier':<20}"
    for name in model_names:
        row += f" | {results[name]['Brier Score']:>15.4f}"
    row += f" | {avg_market_brier:>15.4f}"
    print(row)
    
    # Stage Brier
    for s in ['Early', 'Mid', 'Late']:
        row = f"{s + ' Brier':<20}"
        for name in model_names:
            row += f" | {results[name]['Stages'][s]:>15.4f}"
        row += f" | {m_stages[s]:>15.4f}"
        print(row)
    
    print("-" * len(header))
    
    # Edge vs Market
    row = f"{'Edge vs Market':<20}"
    for name in model_names:
        edge = (avg_market_brier - results[name]['Brier Score']) / avg_market_brier
        row += f" | {edge:>15.2%}"
    row += f" | {'0.00%':>15}"
    print(row)
    
    print("=" * len(header))
    print("\nInterpretation:")
    print("A Brier Score of 0.25 is a random guess (50/50).")
    print("Lower is better. If the market is at 0.12, it's highly efficient.")
    print("Positive edge means the model is more accurate than the mid-market price.")

def compare_models_test_set(model_paths, split_method='temporal'):
    print("=" * 80)
    print(f"NGBOOST MODEL COMPARISON - TEST SET ({split_method.upper()} SPLIT)")
    print("=" * 80)
    
    # 1. Load Data
    print("\nLoading data...")
    X, y_binary = prepare_training_data()
    X = add_interaction_features(X)
    
    # Split
    if split_method == 'temporal':
        # Train on 2022-2024, Test on 2025-26
        train_mask = ~X['season'].isin(['2025-26'])
        test_mask = X['season'].isin(['2025-26'])
        print(f"  Split: Train on pre-2025, Test on 2025-26 games")
    else:
        # Random split by Game ID (leakage risk)
        game_ids = X['game_id'].unique()
        train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
        train_mask = X['game_id'].isin(train_ids)
        test_mask = X['game_id'].isin(test_ids)
        print(f"  Split: Random 80/20 by Game ID")
    
    # Get ground truth
    final_diffs = get_final_score_diffs(X)
    current_diffs = X['score_diff'].values
    
    X_test_full = X[test_mask]
    y_test_final = final_diffs[test_mask]
    test_current_diffs = current_diffs[test_mask]
    
    print(f"Test set: {len(X_test_full)} events, {X_test_full['game_id'].nunique()} games")
    
    results = {}
    # Use both wide and sharp thresholds
    thresholds_wide = [-10.5, -5.5, 0.5, 5.5, 10.5]
    thresholds_sharp = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    
    for name, path in model_paths.items():
        print(f"\nEvaluating {name}...")
        if not os.path.exists(path):
            continue
            
        model_data = joblib.load(path)
        ensemble = model_data.get('ensemble')
        feature_order = model_data.get('feature_order')
        
        # Align features
        missing = [f for f in feature_order if f not in X_test_full.columns]
        for f in missing: X_test_full[f] = 0
        X_test_aligned = X_test_full[feature_order].values
        
        start_time = time.time()
        mean_remainder, std_remainder = get_ensemble_predictions(ensemble, X_test_aligned)
        inf_time = (time.time() - start_time) * 1000
        
        # Point Prediction
        reconstructed_final = test_current_diffs + mean_remainder
        mae = mean_absolute_error(y_test_final, reconstructed_final)
        
        # Brier (Wide)
        bs_wide = []
        for t in thresholds_wide:
            prob = 1 - stats.norm.cdf(t, loc=reconstructed_final, scale=np.maximum(std_remainder, 1.0))
            bs_wide.append(np.mean((prob - (y_test_final > t))**2))
            
        # Brier (Sharp)
        bs_sharp = []
        for t in thresholds_sharp:
            prob = 1 - stats.norm.cdf(t, loc=reconstructed_final, scale=np.maximum(std_remainder, 1.0))
            bs_sharp.append(np.mean((prob - (y_test_final > t))**2))
            
        results[name] = {
            'MAE': mae,
            'BS (Wide)': np.mean(bs_wide),
            'BS (Sharp)': np.mean(bs_sharp),
            'Inf (ms)': inf_time
        }
        print(f"  ✓ MAE: {mae:.2f}, BS (Wide): {np.mean(bs_wide):.4f}, BS (Sharp): {np.mean(bs_sharp):.4f}")

    # Print Summary
    print("\n" + "=" * 80)
    print(f"{'Metric':<20} | {'Baseline':>15} | {'New Model':>15} | {'Improvement':>12}")
    print("-" * 80)
    b = results.get('Baseline')
    n = results.get('New Model')
    if b and n:
        for m in ['MAE', 'BS (Wide)', 'BS (Sharp)']:
            b_val, n_val = b[m], n[m]
            imp = (b_val - n_val) / b_val
            print(f"{m:<20} | {b_val:>15.4f} | {n_val:>15.4f} | {imp:+.2%}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare NGBoost Models')
    parser.add_argument('--table', action='store_true', help='Evaluate on model_predictions table')
    parser.add_argument('--limit', type=int, default=None, help='Limit rows for table evaluation')
    parser.add_argument('--split', type=str, default='temporal', choices=['temporal', 'random'], help='Test set split method')
    args = parser.parse_args()

    model_paths = {
        'Baseline': 'models/nba_spread_ngboost.pkl',
        'New Model': 'models/nba_spread_ngboost_new.pkl',
        'Unweighted': 'models/nba_spread_ngboost_unweighted.pkl'
    }

    if args.table:
        evaluate_on_table(model_paths, limit=args.limit)
    else:
        compare_models_test_set(model_paths, split_method=args.split)
