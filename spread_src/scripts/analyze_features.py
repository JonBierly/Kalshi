"""
Analyze NGBoost feature importances.

Shows which features are most important for predicting score remainder.
NGBoost has separate importances for location (mean) and scale (uncertainty).
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_feature_importance(model_path='models/nba_spread_ngboost_normal.pkl'):
    """Analyze and display feature importances from NGBoost ensemble."""
    
    print("=" * 80)
    print("NGBoost Feature Importance Analysis")
    print("=" * 80)
    
    # Load model
    model_data = joblib.load(model_path)
    
    if 'ensemble' in model_data:
        ensemble = model_data['ensemble']
    else:
        ensemble = [model_data['model']]
    
    feature_order = model_data['feature_order']
    n_models = len(ensemble)
    n_features = len(feature_order)
    
    print(f"\nModel: {model_path}")
    print(f"Ensemble size: {n_models}")
    print(f"Features: {n_features}")
    
    # Aggregate feature importances across ensemble
    # NGBoost returns (n_params, n_features) - row 0 is loc, row 1 is scale
    loc_importances = np.zeros(n_features)
    scale_importances = np.zeros(n_features)
    
    for model in ensemble:
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            if fi.ndim == 2:
                loc_importances += fi[0]  # Location (mean)
                scale_importances += fi[1]  # Scale (uncertainty)
            else:
                loc_importances += fi
    
    # Normalize
    loc_importances = loc_importances / loc_importances.sum() if loc_importances.sum() > 0 else loc_importances
    scale_importances = scale_importances / scale_importances.sum() if scale_importances.sum() > 0 else scale_importances
    combined = (loc_importances + scale_importances) / 2
    
    # Create DataFrame for display
    importance_df = pd.DataFrame({
        'Feature': feature_order,
        'Mean (loc)': loc_importances,
        'Uncertainty (scale)': scale_importances,
        'Combined': combined
    }).sort_values('Combined', ascending=False)
    
    # Print top features
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES (by combined importance)")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Feature':<35} {'Mean':<12} {'Uncert':<12} {'Combined':<12}")
    print("-" * 76)
    
    cumulative = 0
    for rank, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        cumulative += row['Combined']
        print(f"{rank:<5} {row['Feature']:<35} {row['Mean (loc)']:.4f}       {row['Uncertainty (scale)']:.4f}       {row['Combined']:.4f}")
    
    print(f"\n  Top 20 features explain {cumulative:.1%} of variance")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    top_n = 20
    top_df = importance_df.head(top_n)
    
    # Mean importance
    ax1 = axes[0]
    ax1.barh(range(top_n), top_df['Mean (loc)'].values[::-1], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_df['Feature'].values[::-1])
    ax1.set_xlabel('Importance')
    ax1.set_title('LOCATION (Mean) Importance\nWhich features predict WHERE the score lands?')
    for i, v in enumerate(top_df['Mean (loc)'].values[::-1]):
        ax1.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
    
    # Scale importance
    ax2 = axes[1]
    ax2.barh(range(top_n), top_df['Uncertainty (scale)'].values[::-1], color='coral', alpha=0.8)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_df['Feature'].values[::-1])
    ax2.set_xlabel('Importance')
    ax2.set_title('SCALE (Uncertainty) Importance\nWhich features predict HOW UNCERTAIN we are?')
    for i, v in enumerate(top_df['Uncertainty (scale)'].values[::-1]):
        ax2.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved feature importance plot to 'reports/feature_importance.png'")
    
    # Show bottom features (potentially useless)
    print("\n" + "=" * 80)
    print("BOTTOM 10 FEATURES (least important - consider removing)")
    print("=" * 80)
    for _, row in importance_df.tail(10).iterrows():
        print(f"  {row['Feature']:<35} {row['Combined']:.4f}")
    
    return importance_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze NGBoost feature importance')
    parser.add_argument('--model', type=str, default='models/nba_spread_ngboost_normal.pkl',
                        help='Path to model file')
    args = parser.parse_args()
    
    analyze_feature_importance(args.model)
