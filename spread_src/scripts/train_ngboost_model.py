"""
Train NGBoost Ensemble for Score Remainder prediction.

Uses an ensemble of NGBRegressor models with Student-T distribution for:
1. Fat tails (aleatoric uncertainty from each model's distribution)
2. Model disagreement (epistemic uncertainty from ensemble variance)

Output: List of models, each predicting P(ScoreRemainder | X) as Student-T.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless training
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ngboost import NGBRegressor
from ngboost.scores import LogScore  # CRPScore not supported for Student-T
from spread_src.models.distributions import SafeT

# Distribution priority: SafeT > Laplace > Normal
try:
    DISTRIBUTION = SafeT
    print("Using SafeT distribution (Student-T with clipped scale/df)")
except ImportError:
    try:
        from ngboost.distns import Laplace
        DISTRIBUTION = Laplace
        print("Using Laplace distribution (fat tails, fallback)")
    except ImportError:
        from ngboost.distns import Normal
        DISTRIBUTION = Normal
        print("WARNING: Using Normal distribution (thin tails)")

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import (
    BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, INTERACTION_FEATURES_LIST,
    add_interaction_features
)


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


def generate_pit_histogram(models, feature_order, X_test, y_test, current_diffs, save_path='reports/pit_histogram.png'):
    """
    Generate Probability Integral Transform (PIT) histogram using ensemble mean.
    
    If well-calibrated, the histogram should be approximately uniform.
    """
    print("\nGenerating PIT Histogram...")
    
    # Get ensemble mean predictions
    all_pit_values = []
    
    for model in models:
        dist = model.pred_dist(X_test)
        pit_values = dist.cdf(y_test)
        all_pit_values.append(pit_values)
    
    # Use ensemble mean PIT
    mean_pit = np.mean(all_pit_values, axis=0)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(mean_pit, bins=20, density=True, 
                                edgecolor='black', alpha=0.7, color='steelblue')
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (Uniform)')
    
    ax.set_xlabel('PIT Value (Predicted CDF at Observation)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'PIT Histogram - Ensemble of {len(models)} NGBoost Models\n(Should be approximately uniform)', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    
    # Check calibration
    edge_density = (n[0] + n[1] + n[-2] + n[-1]) / 4
    center_density = np.mean(n[8:12])
    
    if edge_density > center_density * 1.3:
        calibration_note = "⚠️ U-SHAPED: Ensemble is under-dispersed"
        ax.text(0.5, 0.95, calibration_note, transform=ax.transAxes, 
                fontsize=10, ha='center', color='orange', fontweight='bold')
    elif center_density > edge_density * 1.3:
        calibration_note = "⚠️ INVERSE-U: Ensemble is over-dispersed"
        ax.text(0.5, 0.95, calibration_note, transform=ax.transAxes,
                fontsize=10, ha='center', color='orange', fontweight='bold')
    else:
        calibration_note = "✓ Approximately uniform - well calibrated!"
        ax.text(0.5, 0.95, calibration_note, transform=ax.transAxes,
                fontsize=10, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ Saved PIT histogram to '{save_path}'")
    
    # plt.show()  # Removed - blocks in headless mode
    
    return mean_pit


def evaluate_ensemble_calibration(models, feature_order, X_test, y_test, current_diffs, final_diffs):
    """
    Check calibration at specific quantiles using ensemble.
    """
    print("\nEnsemble Quantile Calibration Check:")
    print("-" * 50)
    
    # Get predictions from all models
    all_remainders = []
    all_scales = []
    
    for model in models:
        dist = model.pred_dist(X_test)
        all_remainders.append(dist.loc)
        all_scales.append(dist.scale)
    
    # Ensemble mean
    ensemble_remainder = np.mean(all_remainders, axis=0)
    ensemble_scale = np.mean(all_scales, axis=0)
    
    # Reconstruct final diff
    predicted_final = current_diffs + ensemble_remainder
    
    # For each quantile, check coverage
    from scipy import stats
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    for q in quantiles:
        # Use normal approximation for quantile bounds
        z = stats.norm.ppf(q)
        predicted_quantile = predicted_final + z * ensemble_scale
        
        actual_coverage = np.mean(final_diffs <= predicted_quantile)
        deviation = actual_coverage - q
        
        status = "✓" if abs(deviation) < 0.05 else "⚠️"
        print(f"  {q*100:.0f}% quantile: Target={q:.1%}, Actual={actual_coverage:.1%}, "
              f"Deviation={deviation:+.1%} {status}")


def train_ngboost_ensemble(n_models=5):
    """
    Train ensemble of NGBoost models to predict score remainder distribution.
    
    Each model is trained on a bootstrap sample of games (not events).
    """
    print("=" * 80)
    print(f"Training NGBoost Ensemble ({n_models} models)")
    print(f"Distribution: {DISTRIBUTION.__name__}")
    print("=" * 80)
    
    # 1. Load training data
    print("\nLoading training data...")
    X, y_binary = prepare_training_data()
    
    # 2. Get final score differentials for each game
    print("Getting final score differentials...")
    final_diffs = get_final_score_diffs(X)
    
    # 3. Compute Score Remainder target
    current_diffs = X['score_diff'].values
    score_remainders = final_diffs - current_diffs
    
    print(f"\nScore remainder stats (TARGET):")
    print(f"  Mean: {np.mean(score_remainders):.2f}")
    print(f"  Std: {np.std(score_remainders):.2f}")
    print(f"  Min: {np.min(score_remainders):.0f}, Max: {np.max(score_remainders):.0f}")
    
    # 3b. Add interaction features using centralized function
    print("\nAdding interaction features for variance learning...")
    X = add_interaction_features(X)
    print(f"  Added {len(INTERACTION_FEATURES_LIST)} interaction features")
    
    # 4. Split by game ID (same as before)
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    train_mask = X['game_id'].isin(train_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    # 5. Filter to feature columns (including interaction features)
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST + INTERACTION_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    print(f"\nUsing {len(available_cols)} features (including interactions)")
    
    X_train = X[train_mask][available_cols]
    X_test = X[test_mask][available_cols]
    
    y_train = score_remainders[train_mask]
    y_test = score_remainders[test_mask]
    
    train_game_ids = X[train_mask]['game_id'].values
    train_game_ids_unique = np.unique(train_game_ids)
    
    # Keep current_diffs for test set reconstruction
    test_current_diffs = current_diffs[test_mask]
    y_test_final_diffs = final_diffs[test_mask]
    
    print(f"\nTraining: {len(X_train)} events, {len(train_ids)} games")
    print(f"Test: {len(X_test)} events, {len(test_ids)} games")
    
    # 6. Train ensemble with game-level bootstrapping
    print(f"\nTraining {n_models} NGBoost models with game-level bootstrapping...")
    print("Each model trains on ~80% of games (bootstrap sample)")
    print("Estimated time: ~15-20 min per model")
    
    ensemble = []
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"Model {i+1}/{n_models}")
        print(f"{'='*60}")
        
        # Game-level bootstrap
        np.random.seed(42 + i)
        boot_game_ids = np.random.choice(
            train_game_ids_unique, 
            size=len(train_game_ids_unique), 
            replace=True
        )
        
        # Create mask for bootstrapped games
        boot_mask = pd.Series(train_game_ids).isin(boot_game_ids).values
        
        X_boot = X_train.values[boot_mask]
        y_boot = y_train.values[boot_mask] if hasattr(y_train, 'values') else y_train[boot_mask]
        
        # Create validation set (10% of bootstrap) for early stopping
        val_size = int(0.1 * len(X_boot))
        X_train_boot = X_boot[:-val_size]
        y_train_boot = y_boot[:-val_size]
        X_val_boot = X_boot[-val_size:]
        y_val_boot = y_boot[-val_size:]
        
        print(f"  Bootstrap: {len(X_train_boot)} train, {len(X_val_boot)} val")
        
        # Custom base learner with full feature visibility
        # Note: DecisionTreeRegressor doesn't support n_jobs (single-threaded)
        # For faster training, we rely on smaller minibatch_frac instead
        from sklearn.tree import DecisionTreeRegressor
        base_learner = DecisionTreeRegressor(
            max_depth=3,           # Shallower trees for speed
            min_samples_leaf=50,   # Allow finer granularity
            max_features=1.0       # Use ALL features (no subsampling)
        )
        
        # Train model with Student-T + LogScore
        print(f"  Training NGBoost (max 300 iters, early stopping enabled)...")
        model = NGBRegressor(
            Dist=DISTRIBUTION,
            Score=LogScore,          # LogScore required for Student-T
            Base=base_learner,
            n_estimators=300,        # Max iterations
            learning_rate=0.05,      # Higher LR for faster convergence
            minibatch_frac=0.01,     # Small minibatch for speed
            tol=1e-4,                # Early stopping threshold
            col_sample=1.0,          # Use all features
            verbose=True,
            verbose_eval=10,
            random_state=42 + i
        )
        # Fit with validation set for early stopping
        model.fit(X_train_boot, y_train_boot, X_val=X_val_boot, Y_val=y_val_boot)
        
        ensemble.append(model)
        
        # Quick evaluation
        dist = model.pred_dist(X_test.values)
        preds = dist.loc
        mae = mean_absolute_error(y_test, preds)
        print(f"  Model {i+1} Test MAE: {mae:.2f} points")
    
    print(f"\n✓ Trained {len(ensemble)} NGBoost models")
    
    # 7. Evaluate ensemble on test set
    print("\n" + "=" * 80)
    print("Ensemble Evaluation on Test Set")
    print("=" * 80)
    
    # Get predictions from all models
    all_remainders = []
    all_scales = []
    
    for model in ensemble:
        dist = model.pred_dist(X_test.values)
        all_remainders.append(dist.loc)
        all_scales.append(dist.scale)
    
    all_remainders = np.array(all_remainders)
    all_scales = np.array(all_scales)
    
    # Ensemble mean
    ensemble_remainder = np.mean(all_remainders, axis=0)
    ensemble_scale = np.mean(all_scales, axis=0)
    
    # Epistemic uncertainty from ensemble spread
    epistemic_std = np.std(all_remainders, axis=0)
    
    # Reconstruct final diff
    reconstructed_final = test_current_diffs + ensemble_remainder
    
    # Metrics
    remainder_mae = mean_absolute_error(y_test, ensemble_remainder)
    remainder_rmse = np.sqrt(mean_squared_error(y_test, ensemble_remainder))
    final_mae = mean_absolute_error(y_test_final_diffs, reconstructed_final)
    
    print(f"\nRemainder Prediction:")
    print(f"  MAE: {remainder_mae:.2f} points")
    print(f"  RMSE: {remainder_rmse:.2f} points")
    
    print(f"\nReconstructed Final Diff:")
    print(f"  MAE: {final_mae:.2f} points")
    
    print(f"\nUncertainty Stats:")
    print(f"  Mean aleatoric std (from distributions): {np.mean(ensemble_scale):.2f}")
    print(f"  Mean epistemic std (from ensemble): {np.mean(epistemic_std):.2f}")
    print(f"  Total uncertainty: {np.mean(np.sqrt(ensemble_scale**2 + epistemic_std**2)):.2f}")
    print(f"  Actual std of errors: {np.std(y_test - ensemble_remainder):.2f}")
    
    # 8. PIT Histogram
    generate_pit_histogram(ensemble, available_cols, X_test.values, y_test, test_current_diffs)
    
    # 9. Quantile calibration
    evaluate_ensemble_calibration(ensemble, available_cols, X_test.values, y_test, test_current_diffs, y_test_final_diffs)
    
    # 10. Save ensemble
    print("\n" + "=" * 80)
    
    model_data = {
        'ensemble': ensemble,
        'feature_order': available_cols,
        'distribution': DISTRIBUTION.__name__,
        'n_models': n_models
    }
    
    model_path = 'models/nba_spread_ngboost.pkl'
    joblib.dump(model_data, model_path)
    print(f"✓ Saved {n_models}-model NGBoost ensemble to '{model_path}'")
    print("=" * 80)
    
    return ensemble, available_cols


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NGBoost Ensemble')
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in ensemble')
    args = parser.parse_args()
    
    ensemble, features = train_ngboost_ensemble(n_models=args.n_models)
