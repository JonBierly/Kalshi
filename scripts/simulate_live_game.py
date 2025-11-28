
import time
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.acquisition import HistoricalDataClient, LiveDataClient, DataSchema
from src.inference.orchestrator import LiveGameOrchestrator
from src.models.prediction import PredictionEngine

# Mock Live Client that replays historical data
class MockLiveClient:
    def __init__(self, game_id):
        self.game_id = game_id
        self.historical_client = HistoricalDataClient()
        self.game_id = game_id
        self.historical_client = HistoricalDataClient()
        self.pbp_df = self.historical_client.get_game_pbp(game_id).sort_values('event_num')
        self.current_index = 0
        self.current_index = 0
        
        # Get active players from historical boxscore (simplified)
        # In a real mock, we'd query the boxscore endpoint, but here we just assume we know them
        # or let the orchestrator fallback to projected.
        
    def get_todays_games(self):
        return [{'gameId': self.game_id, 'gameCode': 'MOCK/GAME', 'gameStatus': 2, 'homeTeam': {'teamId': 0}, 'awayTeam': {'teamId': 0}}]
        
    def get_live_game_data(self, game_id=None):
        if self.current_index >= len(self.pbp_df):
            return {'gameStatus': 3, 'homeTeam': {'score': 0}, 'awayTeam': {'score': 0}} # Finished
            
        # Simulate "Live" state by aggregating up to current index
        current_slice = self.pbp_df.iloc[:self.current_index+1]
        last_row = current_slice.iloc[-1]
        
        # Construct a payload similar to live boxscore
        # We need to calculate totals from the slice
        home_stats = {
            'fieldGoalsMade': len(current_slice[(current_slice['event_type'] == 1) & (current_slice['player_team_id'] == last_row['home_team_id'])]),
            'fieldGoalsAttempted': len(current_slice[(current_slice['event_type'].isin([1, 2])) & (current_slice['player_team_id'] == last_row['home_team_id'])]),
            'threePointersMade': len(current_slice[(current_slice['event_type'] == 1) & (current_slice['description'].str.contains('3pt')) & (current_slice['player_team_id'] == last_row['home_team_id'])]),
            'turnovers': len(current_slice[(current_slice['event_type'] == 5) & (current_slice['player_team_id'] == last_row['home_team_id'])]),
            'reboundsTotal': len(current_slice[(current_slice['event_type'] == 4) & (current_slice['player_team_id'] == last_row['home_team_id'])])
        }
        
        away_stats = {
            'fieldGoalsMade': len(current_slice[(current_slice['event_type'] == 1) & (current_slice['player_team_id'] == last_row['away_team_id'])]),
            'fieldGoalsAttempted': len(current_slice[(current_slice['event_type'].isin([1, 2])) & (current_slice['player_team_id'] == last_row['away_team_id'])]),
            'threePointersMade': len(current_slice[(current_slice['event_type'] == 1) & (current_slice['description'].str.contains('3pt')) & (current_slice['player_team_id'] == last_row['away_team_id'])]),
            'turnovers': len(current_slice[(current_slice['event_type'] == 5) & (current_slice['player_team_id'] == last_row['away_team_id'])]),
            'reboundsTotal': len(current_slice[(current_slice['event_type'] == 4) & (current_slice['player_team_id'] == last_row['away_team_id'])])
        }
        
        # Clock
        # PBP has remaining_time in seconds.
        rem = last_row['remaining_time']
        mins = int(rem // 60)
        secs = int(rem % 60)
        clock_str = f"PT{mins}M{secs}.00S"
        
        payload = {
            'gameStatus': 2,
            'gameStatusText': clock_str, # For parsing logic compatibility
            'period': last_row['period'],
            'gameClock': clock_str,
            'homeTeam': {
                'teamId': int(last_row['home_team_id']),
                'score': int(last_row['home_score']),
                'statistics': home_stats,
                'players': [] # Empty to force projected roster usage
            },
            'awayTeam': {
                'teamId': int(last_row['away_team_id']),
                'score': int(last_row['away_score']),
                'statistics': away_stats,
                'players': []
            }
        }
        
        self.current_index += 1 # Process 1 event per tick
        return payload



if __name__ == "__main__":
    # Use a real game ID from the database for simulation
    # Example: 0022300001 (DEN vs LAL)
    GAME_ID = '0022300001' 
    HOME_ID = 1610612743 # DEN
    AWAY_ID = 1610612747 # LAL
    
    mock_client = MockLiveClient(GAME_ID)
    orch = LiveGameOrchestrator(client=mock_client)
    orch.run_live_loop(GAME_ID, HOME_ID, AWAY_ID)
