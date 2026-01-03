"""
Research and compare different variance estimation strategies for NBA spread models.
All strategies use a Weighted Ridge model as the Mean Model (fixed).

Strategies:
1. Linear Baseline: Ridge on squared residuals.
2. Log-Linear: Ridge on log(squared residuals).
3. XGBoost: Gradient boosted trees on squared residuals.
4. Lookup Table: Simple historical mean variance per time bucket.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, PERIOD_FEATURES_LIST, VOLATILITY_FEATURES_LIST

def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """Gets the final score differential for each game in X."""
    db = DatabaseManager()
    
    unique_games = X['game_id'].unique()
    print(f"Fetching final scores for {len(unique_games)} games...")
    
    game_final_scores = {}
    for game_id in unique_games:
        query = f"SELECT home_score, away_score FROM pbp_events WHERE game_id = '{game_id}' ORDER BY period DESC, remaining_time ASC LIMIT 1"
        result = pd.read_sql(query, db.engine)
        if not result.empty:
            game_final_scores[game_id] = result['home_score'].iloc[0] - result['away_score'].iloc[0]
        else:
            game_final_scores[game_id] = 0
            
    return X['game_id'].map(game_final_scores).values

def get_time_bucket(seconds_remaining, period):
    """Categorizes game time into buckets, including Overtime (Q5+)."""
    if period >= 5: return "Q5 (OT)"
    if seconds_remaining > 2160: return "Q1"
    if seconds_remaining > 1440: return "Q2"
    if seconds_remaining > 720: return "Q3"
    if seconds_remaining > 300: return "Q4 (>5m)"
    if seconds_remaining > 120: return "Last 5m-2m"
    return "Last 2m"

def evaluate_features(X, score_remainders, sample_weights, feature_cols, test_buckets, test_residuals_sq, label):
    """Trains and evaluates variance strategies for a given feature set."""
    print(f"\n--- Evaluating: {label} ---")
    
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    train_mask = X['game_id'].isin(train_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    X_train = X[train_mask][feature_cols]
    y_train = score_remainders[train_mask]
    w_train = sample_weights[train_mask]
    
    X_test = X[test_mask][feature_cols]
    y_test = score_remainders[test_mask]
    
    # Train mean model to get residuals (using current feature set)
    mean_model = Ridge(alpha=1.0)
    mean_model.fit(X_train, y_train, sample_weight=w_train)
    
    train_resid_sq = (y_train - mean_model.predict(X_train))**2
    test_resid_sq = (y_test - mean_model.predict(X_test))**2
    
    strategies = {}
    
    # Linear Ridge
    var_ridge = Ridge(alpha=1.0)
    var_ridge.fit(X_train, train_resid_sq, sample_weight=w_train)
    strategies["Ridge"] = np.maximum(var_ridge.predict(X_test), 1.0)
    
    # XGBoost
    var_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    var_xgb.fit(X_train, train_resid_sq, sample_weight=w_train)
    strategies["XGB"] = np.maximum(var_xgb.predict(X_test), 1.0)
    
    # Time Lookup (User Request: Sample Variance by time)
    # Group remains by 10s bins to stabilize
    train_df = pd.DataFrame({
        'secs': X[train_mask]['seconds_remaining'],
        'resid_sq': train_resid_sq
    })
    train_df['bin'] = (train_df['secs'] // 10) * 10
    lookup = train_df.groupby('bin')['resid_sq'].mean() # Expectation of squared residuals = Variance (if mean error is 0)
    
    # Add a fallback for bins missing in train (if any)
    global_variance = train_resid_sq.mean()
    
    def get_lookup_pred(secs):
        bin_val = (secs // 10) * 10
        return lookup.get(bin_val, global_variance)
    
    strategies["TimeLookup"] = np.maximum(np.array([get_lookup_pred(s) for s in X[test_mask]['seconds_remaining']]), 1.0)
    
    results = []
    bucket_order = ["Q1", "Q2", "Q3", "Q4 (>5m)", "Q5 (OT)"]
    
    for b in bucket_order:
        b_mask = np.array(test_buckets) == b
        if b_mask.sum() == 0: continue
        
        b_actual_sq_error = test_resid_sq[b_mask]
        row = {"Bucket": b, "Label": label}
        
        for name, preds in strategies.items():
            b_preds = preds[b_mask]
            bias = np.mean(b_actual_sq_error - b_preds)
            row[f"{name} Bias"] = bias
            
        results.append(row)
        
    return results

def run_variance_research():
    print("=" * 80)
    print("Variance Estimator Research: With vs Without Period One-Hot Encoding")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/5] Loading data...")
    X, _ = prepare_training_data()
    
    # 2. Prepare Targets and Weights
    print("\n[2/5] Preparing targets and weights...")
    final_diffs = get_final_score_diffs(X)
    score_remainders = final_diffs - X['score_diff'].values
    
    seconds_remaining = X['seconds_remaining'].values
    sample_weights = np.ones(len(X))
    sample_weights[seconds_remaining < 120] = 10.0
    sample_weights[(seconds_remaining >= 120) & (seconds_remaining < 300)] = 5.0
    sample_weights[(seconds_remaining >= 300) & (seconds_remaining < 600)] = 2.0
    
    # 3. Get feature columns
    feature_cols = [c for c in (BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST) if c in X.columns]
    
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    test_mask = X['game_id'].isin(test_ids)
    
    # Calculate buckets for the test set
    test_buckets_full = [get_time_bucket(s, p) for s, p in zip(X['seconds_remaining'], X['period'])]
    test_buckets = [test_buckets_full[i] for i in range(len(test_buckets_full)) if test_mask.iloc[i]]
    
    # 4. Compare Feature Sets
    # Baseline: No period or chaos features
    baseline_features = [c for c in feature_cols if c not in PERIOD_FEATURES_LIST and c not in VOLATILITY_FEATURES_LIST]
    # New: With period one-hot and chaos features
    new_features = feature_cols
    
    # We pass None for test_residuals_sq because evaluate_features calculates it per run
    results_baseline = evaluate_features(X, score_remainders, sample_weights, baseline_features, test_buckets, None, "Baseline")
    results_new = evaluate_features(X, score_remainders, sample_weights, new_features, test_buckets, None, "With Period & Chaos")
    
    # 5. Final Comparison
    print("\n[5/5] Feature Comparison Report (Bias: Actual - Predicted)")
    print("=" * 100)
    
    df_base = pd.DataFrame(results_baseline)
    df_new = pd.DataFrame(results_new)
    
    comparison = pd.concat([df_base, df_new]).sort_values(["Bucket", "Label"])
    print(comparison.to_string(index=False))
    print("\nNote: Positive values mean over-confident (underestimating risk).")
    print("=" * 100)

if __name__ == "__main__":
    run_variance_research()
