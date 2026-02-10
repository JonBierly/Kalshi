
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.database import DatabaseManager
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.features.engineering import FeatureEngine, TeamStatsEngine, RosterEngine

def diagnose():
    model_path = 'models/nba_spread_ngboost_new.pkl'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found")
        return
        
    model = SpreadDistributionModel(model_path)
    model.conformal_q = 1.0
    
    db = DatabaseManager()
    # Get a recent game that exists
    query = "SELECT game_id, home_team_id, away_team_id FROM games LIMIT 1"
    game_row = pd.read_sql(query, db.engine).iloc[0]
    game_id, home_id, away_id = game_row['game_id'], game_row['home_team_id'], game_row['away_team_id']
    
    query = f"SELECT * FROM pbp_events WHERE game_id = '{game_id}' ORDER BY period, remaining_time DESC"
    pbp_df = pd.read_sql(query, db.engine)
    pbp_df['home_team_id'] = home_id
    pbp_df['away_team_id'] = away_id
    
    engine = FeatureEngine()
    team_engine = TeamStatsEngine()
    roster_engine = RosterEngine()
    
    pre_game_feats = {**team_engine.get_features(game_id, home_id, away_id), 
                      **roster_engine.get_features(game_id, home_id, away_id)}
    
    # Pre-calculate time
    pbp_df['total_seconds'] = pbp_df['remaining_time'] + (4 - pbp_df['period']).clip(lower=0) * 720
    
    print(f"Diagnosing game {game_id}...")
    
    events = pbp_df.to_dict('records')
    last_sampled_time = 9999
    
    for event in events:
        secs = event['total_seconds']
        engine.lightweight_update(event)
        
        if last_sampled_time - secs >= 300: # Every 5 minutes for diagnostics
            if event['period'] > 1:
                live_feats = engine.calculate_current_features(
                    event['score_diff'], secs, event['period'], game_id, home_id, away_id
                )
                total_feats = {**live_feats, **pre_game_feats}
                
                dist_params = model.predict_distribution_params(total_feats)
                means = dist_params['mean']
                stds = dist_params['std']
                
                aleatoric = np.mean(stds**2)
                epistemic = np.var(means)
                total_std = np.sqrt(aleatoric + epistemic)
                
                print(f"Time: {secs:.0f}s | Diff: {event['score_diff']} | Mean: {np.mean(means):.2f} | Std: {total_std:.2f} | Moment: {live_feats['score_momentum']:.2f}")
                
            last_sampled_time = secs

if __name__ == "__main__":
    diagnose()
