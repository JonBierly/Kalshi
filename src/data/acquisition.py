import pandas as pd
import time
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator
try:
    from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2, playbyplay, playbyplayv3, boxscoreadvancedv3, teamgamelog
    from nba_api.stats.static import teams
    from nba_api.live.nba.endpoints import scoreboard, boxscore
except ImportError:
    print("Warning: nba_api not installed. Historical data fetching will fail.")

class DataSchema:
    """Defines the standard columns for the PBP DataFrame."""
    COLUMNS = [
        'game_id', 'timestamp', 'period', 'remaining_time', 
        'home_score', 'away_score', 'score_diff',
        'event_type', 'player_id', 'description', 
        'home_team_id', 'away_team_id'
    ]

class HistoricalDataClient:
    """
    Client to fetch historical NBA play-by-play data using nba_api.
    """
    def __init__(self):
        pass

    def get_season_games(self, season: str = '2023-24') -> List[str]:
        """
        Fetches all game IDs for a given season.
        Format for season is 'YYYY-YY' (e.g., '2023-24').
        """
        print(f"Fetching games for season {season}...")
        # league_id_nullable='00' is NBA
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable='00', timeout=30)
        games = gamefinder.get_data_frames()[0]
        # Filter for Regular Season (002) and Playoffs (004) only
        # Preseason is 001
        valid_games = games[games['GAME_ID'].str.startswith(('002', '004'))]
        return valid_games['GAME_ID'].unique().tolist()

    def get_game_pbp(self, game_id: str) -> pd.DataFrame:
        """
        Fetches raw PBP data for a specific game and standardizes the schema.
        Uses PlayByPlayV3 as per user update.
        """
        print(f"Fetching PBP for game {game_id}...")
        time.sleep(0.5) # Rate limit protection

        try:
            pbp_endpoint = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
            raw_df = pbp_endpoint.get_data_frames()[0]
        except Exception as e:
            print(f"V3 failed ({e})")
            return pd.DataFrame(columns=DataSchema.COLUMNS)
            
        return self._process_raw_pbp(raw_df, game_id)

    def get_advanced_boxscore(self, game_id: str) -> pd.DataFrame:
        """
        Fetches advanced box score stats for a game.
        """
        print(f"Fetching Advanced Stats for game {game_id}...")
        time.sleep(0.6) # Rate limit protection
        
        try:
            endpoint = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=30)
            raw_df = endpoint.get_data_frames()[0]
            return raw_df
        except Exception as e:
            print(f"Advanced Stats failed for {game_id}: {e}")
            return pd.DataFrame()

    def get_team_game_log(self, team_id: int, season: str = '2023-24', season_type: str = 'Regular Season') -> pd.DataFrame:
        """
        Fetches game logs for a specific team and season.
        season_type: 'Regular Season', 'Playoffs', 'Pre Season', 'All Star'
        """
        print(f"Fetching Game Log for Team {team_id} ({season} - {season_type})...")
        time.sleep(0.6)
        
        try:
            endpoint = teamgamelog.TeamGameLog(team_id=team_id, season=season, season_type_all_star=season_type, timeout=30)
            raw_df = endpoint.get_data_frames()[0]
            return raw_df
        except Exception as e:
            print(f"Team Game Log failed for {team_id}: {e}")
            return pd.DataFrame()

    def _process_raw_pbp(self, df: pd.DataFrame, game_id: str) -> pd.DataFrame:
        """Standardizes raw NBA API data to our project schema (V2/V3 compatible)."""
        df = df.copy()
        
        # V3 Format Handling
        if 'gameId' in df.columns:
            def parse_v3_time(t_str):
                if not isinstance(t_str, str): return 0
                t_str = t_str.replace('PT', '').replace('M', ':').replace('S', '')
                try:
                    if ':' in t_str:
                        m, s = t_str.split(':')
                        return int(m) * 60 + float(s)
                    else:
                        return float(t_str)
                except:
                    return 0

            df['remaining_time_period'] = df['clock'].apply(parse_v3_time)
            
            processed = pd.DataFrame()
            processed['game_id'] = df['gameId']
            processed['timestamp'] = pd.to_datetime('now')
            processed['period'] = df['period']
            processed['remaining_time'] = df['remaining_time_period']
            processed['remaining_time'] = df['remaining_time_period']
            
            # Fix: Forward fill scores because V3 often has empty strings for non-scoring events
            s_home = df['scoreHome'].replace('', np.nan).ffill().fillna(0)
            s_away = df['scoreAway'].replace('', np.nan).ffill().fillna(0)
            
            processed['home_score'] = pd.to_numeric(s_home, errors='coerce').astype(int)
            processed['away_score'] = pd.to_numeric(s_away, errors='coerce').astype(int)
            processed['score_diff'] = processed['home_score'] - processed['away_score']
            processed['event_type'] = df['actionType']
            processed['player_id'] = df['personId']
            processed['description'] = df['description']
            processed['home_team_id'] = 0 
            processed['away_team_id'] = 0
            processed['player_team_id'] = df['teamId']
            # V3 usually has actionNumber or orderNumber. Let's try actionNumber first, then orderNumber, then index.
            if 'actionNumber' in df.columns:
                processed['event_num'] = df['actionNumber']
            elif 'orderNumber' in df.columns:
                processed['event_num'] = df['orderNumber']
            else:
                processed['event_num'] = df.index
            
            cols = DataSchema.COLUMNS + ['event_num']
            if 'player_team_id' in processed.columns:
                cols.append('player_team_id')
            
            return processed[cols]
            
        # V2/Legacy Format Handling
        if 'PCTIMESTRING' not in df.columns:
            return pd.DataFrame(columns=DataSchema.COLUMNS)

        def time_to_seconds(t_str):
            if not t_str: return 0
            try:
                m, s = map(int, t_str.split(':'))
                return m * 60 + s
            except:
                return 0
                
        df['remaining_time_period'] = df['PCTIMESTRING'].apply(time_to_seconds)
        
        if 'SCORE' in df.columns:
            df['SCORE'] = df['SCORE'].ffill().fillna('0 - 0')
            try:
                df[['away_score', 'home_score']] = df['SCORE'].str.split(' - ', expand=True).astype(int)
            except:
                df['home_score'] = 0
                df['away_score'] = 0
        else:
             df['home_score'] = 0
             df['away_score'] = 0
        
        processed = pd.DataFrame()
        processed['game_id'] = game_id
        processed['timestamp'] = pd.to_datetime('now') 
        processed['period'] = df['PERIOD']
        processed['remaining_time'] = df['remaining_time_period']
        processed['home_score'] = df['home_score']
        processed['away_score'] = df['away_score']
        processed['score_diff'] = processed['home_score'] - processed['away_score']
        processed['event_type'] = df['EVENTMSGTYPE'] 
        processed['player_id'] = df['PLAYER1_ID']
        processed['description'] = df['HOMEDESCRIPTION'].fillna('') + ' ' + df['VISITORDESCRIPTION'].fillna('')
        processed['home_team_id'] = df['PLAYER1_TEAM_ID'] 
        processed['away_team_id'] = df['PLAYER2_TEAM_ID'] 
        
        processed['event_num'] = df['EVENTNUM']
        
        # Return columns + event_num for sorting
        cols = DataSchema.COLUMNS + ['event_num']
        if 'player_team_id' in processed.columns:
            cols.append('player_team_id')
            
        return processed[cols]

class MockHistoricalDataClient(HistoricalDataClient):
    """
    Generates synthetic PBP data for testing when API is unavailable.
    """
    def get_season_games(self, season: str = '2023-24') -> List[str]:
        return ['0022300001', '0022300002']

    def get_game_pbp(self, game_id: str) -> pd.DataFrame:
        print(f"Generating MOCK PBP for game {game_id}...")
        # Create a synthetic game
        rows = []
        home_score = 0
        away_score = 0
        for period in range(1, 5):
            for minute in range(12, 0, -1):
                # 1 event per minute for simplicity
                # Determine who scored/acted
                is_home_action = (random.random() > 0.5)
                if is_home_action:
                    home_score += random.choice([2, 3])
                    player_team_id = 1610612737
                else:
                    away_score += random.choice([2, 3])
                    player_team_id = 1610612738
                
                rows.append({
                    'game_id': game_id,
                    'timestamp': pd.to_datetime('now'),
                    'period': period,
                    'remaining_time': minute * 60,
                    'home_score': home_score,
                    'away_score': away_score,
                    'score_diff': home_score - away_score,
                    'event_type': 1, # Make
                    'player_id': 12345,
                    'description': 'Shot Made',
                    'home_team_id': 1610612737, # Hawks
                    'away_team_id': 1610612738,  # Celtics
                    'player_team_id': player_team_id
                })
        return pd.DataFrame(rows, columns=DataSchema.COLUMNS)

class LiveDataClient(ABC):
    """Abstract base class for live data sources."""
    @abstractmethod
    def stream_game(self, game_id: str) -> Generator[pd.DataFrame, None, None]:
        """Yields new PBP events as they happen."""
        pass

class ReplayDataClient(LiveDataClient):
    """
    Simulates a live game by replaying historical data event-by-event.
    Useful for development and backtesting.
    """
    def __init__(self, historical_client: HistoricalDataClient, speed_factor: float = 1.0):
        self.hist_client = historical_client
        self.speed_factor = speed_factor

    def stream_game(self, game_id: str) -> Generator[pd.DataFrame, None, None]:
        """
        Fetches the full game data and yields it row by row (or chunk by chunk)
        to simulate a live stream.
        """
        full_game_df = self.hist_client.get_game_pbp(game_id)
        
        print(f"Starting replay for game {game_id}...")
        for i in range(len(full_game_df)):
            # Simulate latency/timing if needed
            # time.sleep(0.1 / self.speed_factor) 
            
            # Yield the current state of the game up to this event
            # In a real live feed, we might get the latest event or a batch of recent events.
            # Here we yield the single new event.
            yield full_game_df.iloc[[i]]

class LiveClient:
    """
    Client for fetching real-time data using nba_api.live endpoints.
    """
    def get_todays_games(self) -> List[Dict]:
        """
        Fetches the list of games for the current day.
        Returns a list of dictionaries with game info.
        """
        try:
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()
            return games
        except Exception as e:
            print(f"Error fetching scoreboard: {e}")
            return []

    def get_live_game_data(self, game_id: str) -> Dict:
        """
        Fetches the live boxscore data for a specific game.
        Contains scores, clock, and active player stats.
        """
        try:
            box = boxscore.BoxScore(game_id=game_id)
            return box.game.get_dict()
        except Exception as e:
            # Silently return empty dict for games that haven't started yet
            return {}
