import time
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.acquisition import LiveClient
from spread_src.features.engineering import TeamStatsEngine, RosterEngine, FeatureEngine
from src.models.prediction import PredictionEngine

class LiveGameOrchestrator:
    def __init__(self, client=None, model_type='lr', skip_model=False):
        """
        Initialize live game orchestrator.
        
        Args:
            client: Optional LiveClient instance
            model_type: 'lr' or 'xgboost' for model selection (default: 'lr')
            skip_model: If True, skip loading prediction model (for spread trading)
        """
        self.live_client = client if client else LiveClient()
        self.team_engine = TeamStatsEngine()
        self.roster_engine = RosterEngine()
        if not skip_model:
            self.prediction_engine = PredictionEngine(model_type=model_type)
        else:
            # Create a minimal prediction engine stand-in for context storage
            class MinimalPredictionEngine:
                def __init__(self):
                    self.current_game_context = {}
                    self.current_game_id = None
            self.prediction_engine = MinimalPredictionEngine()
        self.feature_engine = FeatureEngine()

        
    def get_todays_games(self):
        """Fetches today's games and returns a list of game info."""
        games = self.live_client.get_todays_games()
        # Filter for games that are scheduled (1), in progress (2), or finished (3)
        # We are interested in 1 and 2.
        active_games = [g for g in games if g['gameStatus'] in [1, 2]]
        return active_games

    def setup_game_context(self, game_id, home_team_id, away_team_id):
        """
        Loads pre-game context (Team Stats & Roster Strength).
        """
        print(f"Setting up context for Game {game_id}...")
        
        # 1. Team Stats (Season/Recent)
        # Use get_latest_features for LIVE inference (unshifted)
        home_team_feats = self.team_engine.get_latest_features(home_team_id)
        away_team_feats = self.team_engine.get_latest_features(away_team_id)
        
        # Add context keys (is_home)
        home_team_feats['is_home'] = 1
        away_team_feats['is_home'] = 0
        
        # Rename keys to match model expectation (home_..., away_...)
        team_feats = {}
        for k, v in home_team_feats.items():
            team_feats[f'home_{k}'] = v
        for k, v in away_team_feats.items():
            team_feats[f'away_{k}'] = v
            
        #print("TEAM FEATURES: ", team_feats)
        
        # 2. Roster Stats
        # Try to get active roster from live boxscore first
        home_players = []
        away_players = []
        
        try:
            live_data = self.live_client.get_live_game_data(game_id)
            if 'homeTeam' in live_data and 'players' in live_data['homeTeam']:
                home_players = [p['personId'] for p in live_data['homeTeam']['players']]
                
            if 'awayTeam' in live_data and 'players' in live_data['awayTeam']:
                away_players = [p['personId'] for p in live_data['awayTeam']['players']]
        except Exception as e:
            print(f"Could not fetch live roster (Game likely hasn't started): {e}")
            print("Using projected roster based on recent minutes.")
        #print("HOME PLAYERS: ", home_players)
        #print("AWAY PLAYERS: ", away_players)
        # Get Roster Features (Projected if list is empty)
        home_roster_feats = self.roster_engine.get_projected_roster_features(home_team_id, home_players if home_players else None)
        away_roster_feats = self.roster_engine.get_projected_roster_features(away_team_id, away_players if away_players else None)
        
        # Rename keys to match model expectation (home_roster_..., away_roster_...)
        final_roster_feats = {}
        for k, v in home_roster_feats.items():
            final_roster_feats[f'home_{k}'] = v
        for k, v in away_roster_feats.items():
            final_roster_feats[f'away_{k}'] = v
            
        # Combine
        context = {**team_feats, **final_roster_feats}
        
        # Load into Prediction Engine
        self.prediction_engine.current_game_context = context
        self.prediction_engine.current_game_id = game_id
        
        # Fill NaNs
        for k, v in self.prediction_engine.current_game_context.items():
            if pd.isna(v): self.prediction_engine.current_game_context[k] = 0.0
            
        #print("Context loaded successfully.")
        return context

    def run_live_loop(self, game_id, home_team_id, away_team_id, refresh_rate=5):
        """
        Polls live data and generates predictions.
        """
        #print(f"Starting Live Inference for Game {game_id}...")
        self.setup_game_context(game_id, home_team_id, away_team_id)
        self.feature_engine.reset()
        
        while True:
            try:
                # 1. Get Live Data
                data = self.live_client.get_live_game_data(game_id)
                
                if not data:
                    print("No data received. Waiting...")
                    time.sleep(refresh_rate)
                    continue
                    
                # 2. Parse Live Features
                # We need to construct a 'row' similar to PBP event row for FeatureEngine
                # OR we can manually set the state in FeatureEngine if we trust the boxscore totals.
                # FeatureEngine tracks state incrementally.
                # BUT live boxscore gives us TOTALS.
                # So we can just overwrite the FeatureEngine state with the totals from boxscore.
                
                home_team = data['homeTeam']
                away_team = data['awayTeam']
                
                # Update FeatureEngine State directly
                self.feature_engine.home_stats.update({
                    'fgm': home_team['statistics']['fieldGoalsMade'],
                    'fga': home_team['statistics']['fieldGoalsAttempted'],
                    'fg3m': home_team['statistics']['threePointersMade'],
                    'to': home_team['statistics']['turnovers'],
                    'reb': home_team['statistics']['reboundsTotal'],
                    'pts': home_team['score'],
                    'fta': home_team['statistics']['freeThrowsAttempted'],
                    'oreb': home_team['statistics']['reboundsOffensive']
                })
                
                self.feature_engine.away_stats.update({
                    'fgm': away_team['statistics']['fieldGoalsMade'],
                    'fga': away_team['statistics']['fieldGoalsAttempted'],
                    'fg3m': away_team['statistics']['threePointersMade'],
                    'to': away_team['statistics']['turnovers'],
                    'reb': away_team['statistics']['reboundsTotal'],
                    'pts': away_team['score'],
                    'fta': away_team['statistics']['freeThrowsAttempted'],
                    'oreb': away_team['statistics']['reboundsOffensive']
                })
                
                # Construct "Event Row" for the remaining features (Score, Time)
                # Parse Clock
                remaining_time = 0
                period = data['period']
                
                if 'gameClock' in data:
                    t_str = data['gameClock']
                    t_str = t_str.replace('PT', '').replace('M', ':').replace('S', '')
                    if ':' in t_str:
                        m, s = t_str.split(':')
                        remaining_time = int(m) * 60 + float(s)
                
                total_seconds = remaining_time
                if period <= 4:
                    total_seconds += (4 - period) * 720
                
                # Update history for momentum and calculate current features
                score_diff = home_team['score'] - away_team['score']
                self.feature_engine.history.append((total_seconds, score_diff))
                # Prune history to last 10 mins
                if len(self.feature_engine.history) > 1000:
                    self.feature_engine.history = [h for h in self.feature_engine.history if h[0] < total_seconds + 600]

                live_features = self.feature_engine.calculate_current_features(
                    score_diff, 
                    total_seconds, 
                    game_id, 
                    home_team_id, 
                    away_team_id
                )
                
                # 3. Predict
                # Merge with context
                full_feats = {**live_features, **self.prediction_engine.current_game_context}
                
                result = self.prediction_engine.predict_with_confidence(full_feats)
                
                # 4. Display
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Score: {home_team['score']}-{away_team['score']} | "
                      f"Win Prob: {result['probability']:.1%} "
                      f"(95% CI: {result['ci_95_lower']:.1%}-{result['ci_95_upper']:.1%})", end="")
                
                if data['gameStatus'] == 3: # Final
                    print("\nGame Finished.")
                    break
                    
                time.sleep(refresh_rate)
                
            except KeyboardInterrupt:
                print("\nStopping Live Inference.")
                break
            except Exception as e:
                print(f"\nError in loop: {e}")
                time.sleep(refresh_rate)
