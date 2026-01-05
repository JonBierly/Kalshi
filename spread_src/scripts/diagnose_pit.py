"""
Diagnose PIT U-shape by analyzing residuals across time and game state.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager
from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST, INTERACTION_FEATURES_LIST, add_interaction_features
from sklearn.model_selection import train_test_split

from spread_src.models.distributions import SafeT


def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """Get final score diffs for each game."""
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


def diagnose_pit_issue(model_path='models/nba_spread_ngboost.pkl'):
    """Analyze why PIT is U-shaped."""
    
    print("=" * 80)
    print("Diagnosing U-shaped PIT Histogram")
    print("=" * 80)
    
    # Load model
    model_data = joblib.load(model_path)
    if 'ensemble' in model_data:
        ensemble = model_data['ensemble']
    else:
        ensemble = [model_data['model']]
    feature_order = model_data['feature_order']
    
    # Load test data
    print("\nLoading data...")
    X, _ = prepare_training_data()
    final_diffs = get_final_score_diffs(X)
    score_remainders = final_diffs - X['score_diff'].values
    
    # Add interaction features (same as training)
    X = add_interaction_features(X)
    
    # Split same as training
    game_ids = X['game_id'].unique()
    _, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    test_mask = X['game_id'].isin(test_ids)
    
    X_test = X[test_mask][feature_order]
    y_test = score_remainders[test_mask]
    seconds_test = X[test_mask]['seconds_remaining'].values
    
    # Get predictions from ensemble
    all_locs = []
    all_scales = []
    
    for model in ensemble:
        dist = model.pred_dist(X_test.values)
        all_locs.append(dist.loc)
        all_scales.append(dist.scale)
    
    mean_pred = np.mean(all_locs, axis=0)
    mean_scale = np.mean(all_scales, axis=0)
    
    # Compute residuals
    residuals = y_test - mean_pred
    standardized_residuals = residuals / mean_scale
    
    print(f"\nResidual Stats:")
    print(f"  Mean: {np.mean(residuals):.2f}")
    print(f"  Std: {np.std(residuals):.2f}")
    print(f"  Mean predicted scale: {np.mean(mean_scale):.2f}")
    print(f"  Ratio (actual/predicted): {np.std(residuals) / np.mean(mean_scale):.2f}")
    
    # Analyze by time bucket
    time_buckets = [
        (2400, 2880, "1Q (48-40 min)"),
        (1920, 2400, "2Q (40-32 min)"),
        (1440, 1920, "3Q (32-24 min)"),
        (960, 1440, "Half (24-16 min)"),
        (480, 960, "4Q early (16-8 min)"),
        (120, 480, "4Q late (8-2 min)"),
        (0, 120, "Clutch (<2 min)"),
    ]
    
    print("\n" + "=" * 80)
    print("Residual Analysis by Time Bucket")
    print("=" * 80)
    print(f"\n{'Bucket':<20} {'N':<10} {'Actual Std':<12} {'Pred Scale':<12} {'Ratio':<10}")
    print("-" * 64)
    
    ratios = []
    for low, high, label in time_buckets:
        mask = (seconds_test >= low) & (seconds_test < high)
        if mask.sum() > 100:
            actual_std = np.std(residuals[mask])
            pred_scale = np.mean(mean_scale[mask])
            ratio = actual_std / pred_scale if pred_scale > 0 else 0
            ratios.append((label, ratio))
            
            status = "✓" if 0.8 < ratio < 1.2 else "⚠️ UNDER" if ratio > 1.2 else "⚠️ OVER"
            print(f"{label:<20} {mask.sum():<10} {actual_std:<12.2f} {pred_scale:<12.2f} {ratio:<10.2f} {status}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Standardized residuals histogram
    ax1 = axes[0, 0]
    ax1.hist(standardized_residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(-4, 4, 100)
    from scipy import stats
    ax1.plot(x, stats.norm.pdf(x), 'r--', label='Standard Normal')
    ax1.set_xlabel('Standardized Residual (r / scale)')
    ax1.set_ylabel('Density')
    ax1.set_title('Standardized Residuals\n(Should match red line if well-calibrated)')
    ax1.legend()
    
    # 2. Actual vs Predicted std by time
    ax2 = axes[0, 1]
    labels = [r[0] for r in ratios]
    ratio_vals = [r[1] for r in ratios]
    colors = ['green' if 0.8 < r < 1.2 else 'red' for r in ratio_vals]
    ax2.bar(range(len(labels)), ratio_vals, color=colors, alpha=0.7)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=2)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Actual Std / Predicted Scale')
    ax2.set_title('Calibration by Time Bucket\n(Should be ~1.0 everywhere)')
    
    # 3. Predicted scale vs actual residual std (scatter)
    ax3 = axes[1, 0]
    # Sample for visibility
    idx = np.random.choice(len(mean_scale), min(5000, len(mean_scale)), replace=False)
    ax3.scatter(mean_scale[idx], np.abs(residuals[idx]), alpha=0.1, s=5)
    ax3.plot([0, 20], [0, 20], 'r--', label='Perfect calibration')
    ax3.set_xlabel('Predicted Scale')
    ax3.set_ylabel('Absolute Residual')
    ax3.set_title('Scale vs Actual Error\n(Points should cluster around red line)')
    ax3.legend()
    
    # 4. Scale distribution
    ax4 = axes[1, 1]
    ax4.hist(mean_scale, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax4.axvline(np.std(residuals), color='red', linestyle='--', label=f'Actual Std: {np.std(residuals):.1f}')
    ax4.set_xlabel('Predicted Scale')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Predicted Scale\n(Red line = Actual Std of Residuals)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('reports/pit_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plot to 'reports/pit_diagnosis.png'")
    
    # Suggestions
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    overall_ratio = np.std(residuals) / np.mean(mean_scale)
    
    if overall_ratio > 1.3:
        print(f"\n⚠️ Overall ratio = {overall_ratio:.2f} > 1.3")
        print("   The model CONSISTENTLY UNDER-PREDICTS uncertainty.")
        print("\n   Possible fixes:")
        print("   1. Add a variance multiplier based on seconds_remaining")
        print("   2. Add time-interaction features (margin × time)")
        print("   3. Train separate models for early/late game")
    elif overall_ratio < 0.7:
        print(f"\n⚠️ Overall ratio = {overall_ratio:.2f} < 0.7")
        print("   The model OVER-PREDICTS uncertainty.")
    else:
        print(f"\n✓ Overall ratio = {overall_ratio:.2f} is reasonable.")
        print("   The U-shape might be from specific time buckets.")
    
    # Check time-dependent miscalibration
    early_ratios = [r[1] for r in ratios[:3]]  # 1Q, 2Q, 3Q
    late_ratios = [r[1] for r in ratios[-3:]]  # 4Q late, clutch
    
    if np.mean(early_ratios) > 1.3 and np.mean(late_ratios) < 1.1:
        print("\n⚠️ Early game is under-dispersed, late game is OK")
        print("   Consider: time-dependent variance scaling")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/nba_spread_ngboost.pkl')
    args = parser.parse_args()
    
    diagnose_pit_issue(args.model)
