import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.database import DatabaseManager, Game
from spread_src.features.engineering import FeatureEngine, add_interaction_features
from spread_src.models.spread_model import SpreadDistributionModel

def calculate_brier_score(prob, outcome):
    """Brier Score = (prob - outcome)^2"""
    return (prob - outcome) ** 2

def evaluate_performance(num_games=100, model_paths=['models/nba_spread_ngboost_new.pkl', 'models/nba_spread_ngboost_v2.pkl']):
    # Use absolute path for DB and read-only mode
    db_path = '/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/nba_data.db'
    db_url = f'sqlite:///{db_path}?mode=ro'
    db = DatabaseManager(db_url=db_url)
    session = db.get_session()
    
    # 1. Get last N games that are finished
    print(f"Retrieving last {num_games} games...")
    games = session.query(Game).order_by(Game.date.desc()).limit(num_games).all()
    session.close()
    
    if not games:
        print("No games found in database.")
        return

    # 2. Load models and engines
    models = {}
    for path in model_paths:
        if os.path.exists(path):
            name = os.path.basename(path).replace('.pkl', '')
            models[name] = SpreadDistributionModel(path)
        else:
            print(f"Warning: Model not found at {path}")
    
    if not models:
        print("No models loaded for evaluation.")
        return
    
    from spread_src.features.engineering import TeamStatsEngine, RosterEngine
    print("Initializing feature engines...")
    team_engine = TeamStatsEngine()
    roster_engine = RosterEngine()
    
    results = []
    
    # Thresholds to evaluate (common Kalshi-like spreads)
    thresholds = [-15.5, -10.5, -5.5, -0.5, 0.5, 5.5, 10.5, 15.5]
    
    print(f"Evaluating models on {len(games)} games...")
    
    for game in tqdm(games):
        game_id = game.game_id
        home_margin = game.home_score - game.away_score
        
        # Get PBP events for this game
        query = f"SELECT * FROM pbp_events WHERE game_id = '{game_id}' ORDER BY period, remaining_time DESC"
        pbp_df = pd.read_sql(query, db.engine)
        
        if pbp_df.empty:
            continue
            
        pbp_df['home_team_id'] = game.home_team_id
        pbp_df['away_team_id'] = game.away_team_id
        
        engine = FeatureEngine()
        
        # Fetch advanced features once per game
        team_feats = team_engine.get_features(game_id, game.home_team_id, game.away_team_id)
        roster_feats = roster_engine.get_features(game_id, game.home_team_id, game.away_team_id)
        advanced_context = {**team_feats, **roster_feats}
        
        # Evaluate every 2 minutes of game time
        last_eval_time = 3000 # Start before game
        
        for _, row in pbp_df.iterrows():
            engine.lightweight_update(row)
            
            seconds_remaining = row['remaining_time']
            if row['period'] <= 4:
                seconds_remaining += (4 - row['period']) * 720
            
            if last_eval_time - seconds_remaining >= 120:
                # Build features
                live_features = engine.calculate_current_features(
                    row['score_diff'], seconds_remaining, row['period'], 
                    game_id, game.home_team_id, game.away_team_id
                )
                
                full_features = {**live_features, **advanced_context}
                
                outcome_vec = []
                for threshold in thresholds:
                    outcome_vec.append(1 if home_margin > threshold else 0)
                
                # Model predictions for all models
                for name, spread_model in models.items():
                    preds = spread_model.predict_spread_probabilities(full_features, thresholds)
                    
                    for i, threshold in enumerate(thresholds):
                        prob = preds['probabilities'][i]
                        outcome = outcome_vec[i]
                        brier = calculate_brier_score(prob, outcome)
                        
                        results.append({
                            'game_id': game_id,
                            'model': name,
                            'seconds_remaining': seconds_remaining,
                            'threshold': threshold,
                            'prob': prob,
                            'outcome': outcome,
                            'brier': brier,
                            'score_diff': row['score_diff'],
                            'period': row['period'],
                            'home_margin': home_margin
                        })
                
                last_eval_time = seconds_remaining

    if not results:
        print("No evaluation results were generated.")
        return

    results_df = pd.DataFrame(results)
    
    # 3. Analyze Results
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    model_stats = results_df.groupby('model')['brier'].mean()
    print("Overall Brier Scores:")
    print(model_stats)
    
    # Comparison by Time Bucket
    results_df['time_bucket'] = pd.cut(results_df['seconds_remaining'], 
                                      bins=[0, 300, 600, 1200, 1800, 2880],
                                      labels=['<5m', '5-10m', '10-20m', '20-30m', '30m+'])
    
    print("\nBrier Score by Time Remaining (Competitive Comparison):")
    time_comparison = results_df.pivot_table(index='time_bucket', columns='model', values='brier', aggfunc='mean')
    print(time_comparison)
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/model_comparison_100.csv', index=False)
    print(f"\nDetailed comparison saved to 'reports/model_comparison_100.csv'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--models', nargs='+', default=['models/nba_spread_ngboost_new.pkl', 'models/nba_spread_ngboost_v2.pkl', 'models/nba_spread_ngboost_v3.pkl'])
    args = parser.parse_args()
    
    evaluate_performance(args.num_games, args.models)
