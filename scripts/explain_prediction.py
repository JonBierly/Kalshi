"""
Explain Individual Predictions

Interactive script to analyze and explain specific game predictions.

Usage:
    python scripts/explain_prediction.py [game_id]
    
If game_id is not provided, analyzes a random sample from test set.
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
    SHAPAnalyzer,
    PredictionExplainer
)
from src.models.training import prepare_training_data
from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST
from src.data.database import DatabaseManager

def explain_game_prediction(game_id: str, models, feature_cols):
    """Explain prediction for a specific game."""
    print("\n" + "=" * 80)
    print(f"Analyzing Game: {game_id}")
    print("=" * 80)
    
    # Load game data
    db = DatabaseManager()
    
    # Get game info
    query = f"""
    SELECT game_id, home_team_id, away_team_id, date
    FROM games 
    WHERE game_id = '{game_id}'
    """
    game_info = pd.read_sql(query, db.engine)
    
    if game_info.empty:
        print(f"Error: Game {game_id} not found in database")
        return
    
    print(f"\nGame Info:")
    print(f"  Date: {game_info.iloc[0]['date']}")
    print(f"  Home Team ID: {game_info.iloc[0]['home_team_id']}")
    print(f"  Away Team ID: {game_info.iloc[0]['away_team_id']}")
    
    # Get play-by-play and features
    from src.features.engineering import create_live_features, add_advanced_features
    
    pbp_query = f"""
    SELECT * FROM pbp_events 
    WHERE game_id = '{game_id}' 
    ORDER BY period, remaining_time DESC
    """
    pbp_df = pd.read_sql(pbp_query, db.engine)
    
    if pbp_df.empty:
        print(f"Error: No play-by-play data found for game {game_id}")
        return
    
    pbp_df['home_team_id'] = game_info.iloc[0]['home_team_id']
    pbp_df['away_team_id'] = game_info.iloc[0]['away_team_id']
    
    # Create features
    print("\nGenerating features...")
    features_df = create_live_features(pbp_df)
    X = add_advanced_features(features_df)
    X = X.fillna(0)
    
    # Filter to available features
    available_cols = [c for c in feature_cols if c in X.columns]
    X_features = X[available_cols]
    
    # Get final game state
    final_idx = len(X_features) - 1
    X_final = X_features.iloc[[final_idx]]
    
    # Get actual outcome
    final_score = pbp_df.iloc[-1]
    actual_home_win = 1 if final_score['home_score'] > final_score['away_score'] else 0
    
    print(f"\nFinal Score:")
    print(f"  Home: {final_score['home_score']}")
    print(f"  Away: {final_score['away_score']}")
    print(f"  Result: {'Home Win' if actual_home_win else 'Away Win'}")
    
    # Make prediction
    print("\n" + "-" * 80)
    print("Model Prediction (Final State)")
    print("-" * 80)
    
    explainer = PredictionExplainer(models, available_cols)
    
    # Create SHAP analyzer for detailed explanation
    print("\nCreating SHAP explainer...")
    shap_analyzer = SHAPAnalyzer(models, X_features.iloc[:100], available_cols)
    
    result = explainer.explain_prediction(X_final, shap_analyzer)
    
    print(f"\nPredicted Home Win Probability: {result['probability']:.2%}")
    print(f"Ensemble Std Dev: {result['std']:.4f}")
    print(f"Min/Max across models: {result['min_prob']:.2%} - {result['max_prob']:.2%}")
    print(f"Actual Outcome: {'Home Win' if actual_home_win else 'Away Win'}")
    
    correct = (result['probability'] > 0.5) == actual_home_win
    print(f"Prediction: {'✓ Correct' if correct else '✗ Incorrect'}")
    
    # Show top feature contributions
    print("\n" + "-" * 80)
    print("Top 15 Feature Contributions (SHAP values)")
    print("-" * 80)
    
    contributions = result['feature_contributions']
    print(f"\n{'Feature':<40s} {'Value':>12s} {'SHAP':>12s} {'Effect':>10s}")
    print("-" * 80)
    
    for i, row in contributions.head(15).iterrows():
        effect = "→ Home" if row['shap_value'] > 0 else "→ Away"
        print(f"{row['feature']:<40s} {row['value']:>12.4f} {row['shap_value']:>12.4f} {effect:>10s}")
    
    # Create visualizations
    output_dir = Path('reports/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving visualizations to {output_dir}/...")
    
    # SHAP waterfall
    fig = shap_analyzer.plot_waterfall(X_final, instance_idx=0)
    fig.savefig(output_dir / f'shap_waterfall_{game_id}.png', 
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ SHAP waterfall plot")
    
    # Feature contributions bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_contrib = contributions.head(20)
    colors = ['red' if x < 0 else 'blue' for x in top_contrib['shap_value']]
    ax.barh(range(len(top_contrib)), top_contrib['shap_value'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_contrib)))
    ax.set_yticklabels(top_contrib['feature'])
    ax.set_xlabel('SHAP Value (← Away Win | Home Win →)')
    ax.set_title(f'Feature Contributions - Game {game_id}')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / f'contributions_{game_id}.png', 
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Feature contributions chart")
    
    # Analyze prediction over time
    print("\nAnalyzing prediction trajectory over game...")
    
    # Sample every N events for efficiency
    sample_interval = max(1, len(X_features) // 50)
    time_indices = list(range(0, len(X_features), sample_interval))
    time_indices.append(len(X_features) - 1)  # Always include final
    
    time_preds = []
    for idx in time_indices:
        X_t = X_features.iloc[[idx]]
        probs = [m.predict_proba(X_t)[0][1] for m in models]
        time_preds.append({
            'event': idx,
            'probability': np.mean(probs),
            'std': np.std(probs)
        })
    
    time_df = pd.DataFrame(time_preds)
    
    # Plot trajectory
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_df['event'], time_df['probability'], linewidth=2)
    ax.fill_between(time_df['event'], 
                    time_df['probability'] - time_df['std'],
                    time_df['probability'] + time_df['std'],
                    alpha=0.3)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, label='Even Odds')
    ax.axhline(y=actual_home_win, color='green', linestyle='--', 
              linewidth=1.5, alpha=0.5, label='Actual Outcome')
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Home Win Probability')
    ax.set_title(f'Prediction Trajectory - Game {game_id}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f'trajectory_{game_id}.png', 
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Prediction trajectory")
    
    print("\n✓ Analysis complete!\n")


def main():
    # Load models
    print("Loading models...")
    model_path = 'models/nba_live_model_ensemble.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python src/models/training.py")
        return
    
    models = joblib.load(model_path)
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    
    # Get game_id from command line or use sample
    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        explain_game_prediction(game_id, models, feature_cols)
    else:
        # Analyze a few random test games
        print("No game_id provided. Analyzing sample games from test set...")
        
        # Get test games
        db = DatabaseManager()
        query = """
        SELECT DISTINCT game_id 
        FROM pbp_events 
        ORDER BY RANDOM() 
        LIMIT 3
        """
        sample_games = pd.read_sql(query, db.engine)
        
        if sample_games.empty:
            print("No games found in database")
            return
        
        for game_id in sample_games['game_id']:
            explain_game_prediction(game_id, models, feature_cols)
            print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
