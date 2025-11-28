"""
Comprehensive Model Analysis Script

Generates feature importance reports, SHAP visualizations, 
partial dependence plots, and correlation analyses.

Usage:
    python scripts/analyze_model.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_analysis import (
    FeatureImportanceAnalyzer,
    SHAPAnalyzer,
    PartialDependenceAnalyzer,
    FeatureCorrelationAnalyzer
)
from src.models.training import prepare_training_data
from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST

def main():
    print("=" * 80)
    print("NBA Model Analysis")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('reports/model_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutputs will be saved to: {output_dir}")
    
    # Load models
    print("\n[1/6] Loading models...")
    model_path = 'models/nba_live_model_ensemble.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python src/models/training.py")
        return
    
    models = joblib.load(model_path)
    print(f"  Loaded {len(models)} models in ensemble")
    
    # Prepare data
    print("\n[2/6] Loading training data...")
    X, y = prepare_training_data()
    
    if X.empty:
        print("Error: No training data found")
        return
    
    print(f"  Data shape: {X.shape}")
    print(f"  Target distribution: {np.mean(y):.2%} home wins")
    
    # Extract features
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    X_features = X[available_cols]
    
    print(f"  Using {len(available_cols)} features")
    
    # Use a sample for faster analysis
    sample_size = min(5000, len(X_features))
    sample_indices = np.random.choice(len(X_features), sample_size, replace=False)
    X_sample = X_features.iloc[sample_indices]
    y_sample = y[sample_indices]
    
    print(f"  Using sample of {sample_size} instances for analysis")
    
    # =========================================================================
    # Feature Importance Analysis
    # =========================================================================
    print("\n[3/6] Analyzing feature importance...")
    importance_analyzer = FeatureImportanceAnalyzer(models, available_cols)
    
    # Native importance
    print("  Computing XGBoost native importance...")
    native_importance = importance_analyzer.get_native_importance()
    
    # Save to CSV
    native_importance.to_csv(output_dir / 'feature_importance_native.csv', index=False)
    
    # Plot
    fig = importance_analyzer.plot_importance(
        native_importance, 
        top_n=25,
        title='XGBoost Native Feature Importance (Top 25)'
    )
    fig.savefig(output_dir / 'importance_native.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n  Top 10 Most Important Features (Native):")
    for i, row in native_importance.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")
    
    # Permutation importance (on sample for speed)
    print("\n  Computing permutation importance...")
    perm_importance = importance_analyzer.get_permutation_importance(
        X_sample, y_sample, n_repeats=5
    )
    
    perm_importance.to_csv(output_dir / 'feature_importance_permutation.csv', index=False)
    
    fig = importance_analyzer.plot_importance(
        perm_importance,
        top_n=25,
        importance_col='permutation_importance',
        title='Permutation Feature Importance (Top 25)'
    )
    fig.savefig(output_dir / 'importance_permutation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # =========================================================================
    # SHAP Analysis
    # =========================================================================
    print("\n[4/6] Computing SHAP values...")
    
    # Use smaller background for SHAP
    shap_background_size = min(500, len(X_sample))
    X_shap_background = X_sample.iloc[:shap_background_size]
    
    shap_analyzer = SHAPAnalyzer(models, X_shap_background, available_cols)
    
    print("  Creating SHAP summary plot...")
    # Use a sample for SHAP visualization
    shap_viz_size = min(1000, len(X_sample))
    X_shap_viz = X_sample.iloc[:shap_viz_size]
    
    fig = shap_analyzer.plot_summary(X_shap_viz, max_display=25)
    fig.savefig(output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Create waterfall plots for a few examples
    print("  Creating example SHAP waterfall plots...")
    for i in range(min(3, len(X_sample))):
        fig = shap_analyzer.plot_waterfall(X_sample, instance_idx=i)
        fig.savefig(output_dir / f'shap_waterfall_example_{i+1}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # =========================================================================
    # Partial Dependence Analysis
    # =========================================================================
    print("\n[5/6] Creating partial dependence plots...")
    
    pd_analyzer = PartialDependenceAnalyzer(models, available_cols)
    
    # Select key features for PDP
    key_features = [
        'score_diff',
        'seconds_remaining',
        'home_efg',
        'home_team_season_win_pct',
        'home_team_recent_win_pct',
        'home_roster_season_pie'
    ]
    
    # Filter to available features
    key_features = [f for f in key_features if f in available_cols]
    
    if key_features:
        fig = pd_analyzer.plot_partial_dependence(X_sample, key_features)
        fig.savefig(output_dir / 'partial_dependence.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # =========================================================================
    # Feature Correlation Analysis
    # =========================================================================
    print("\n[6/6] Analyzing feature correlations...")
    
    corr_analyzer = FeatureCorrelationAnalyzer(available_cols)
    
    # Correlation matrix
    corr_matrix = corr_analyzer.get_correlation_matrix(X_sample)
    corr_matrix.to_csv(output_dir / 'feature_correlations.csv')
    
    # Plot
    fig = corr_analyzer.plot_correlation_matrix(X_sample, threshold=0.7)
    fig.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Find redundant features
    redundant = corr_analyzer.find_redundant_features(X_sample, threshold=0.95)
    if redundant:
        print(f"\n  Found {len(redundant)} highly redundant feature pairs (|r| > 0.95):")
        for feat1, feat2, r in redundant:
            print(f"    {feat1} <-> {feat2}: {r:.3f}")
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nGenerated files in {output_dir}:")
    print("  - feature_importance_native.csv")
    print("  - feature_importance_permutation.csv")
    print("  - importance_native.png")
    print("  - importance_permutation.png")
    print("  - shap_summary.png")
    print("  - shap_waterfall_example_*.png")
    print("  - partial_dependence.png")
    print("  - correlation_matrix.png")
    print("  - feature_correlations.csv")
    
    # Create summary report
    summary_path = output_dir / 'analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("NBA Model Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Models: {len(models)} in ensemble\n")
        f.write(f"Features: {len(available_cols)}\n")
        f.write(f"Training samples: {len(X)}\n")
        f.write(f"Analysis sample: {sample_size}\n\n")
        
        f.write("Top 15 Most Important Features (Native):\n")
        f.write("-" * 80 + "\n")
        for i, row in native_importance.head(15).iterrows():
            f.write(f"{i+1:2d}. {row['feature']:40s} {row['mean_importance']:.4f} ± {row['std_importance']:.4f}\n")
        
        f.write("\n\nTop 15 Most Important Features (Permutation):\n")
        f.write("-" * 80 + "\n")
        for i, row in perm_importance.head(15).iterrows():
            f.write(f"{i+1:2d}. {row['feature']:40s} {row['permutation_importance']:.4f} ± {row['std']:.4f}\n")
        
        if redundant:
            f.write("\n\nHighly Correlated Features (|r| > 0.95):\n")
            f.write("-" * 80 + "\n")
            for feat1, feat2, r in redundant:
                f.write(f"  {feat1} <-> {feat2}: {r:.3f}\n")
    
    print(f"  - analysis_summary.txt")
    print("\n✓ Done!\n")


if __name__ == '__main__':
    main()
