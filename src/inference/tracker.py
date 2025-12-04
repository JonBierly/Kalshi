import time
import pandas as pd
from datetime import datetime
from src.inference.orchestrator import LiveGameOrchestrator
from src.data.kalshi import KalshiClient
from src.data.database import DatabaseManager, OddsHistory
from sqlalchemy.orm import sessionmaker

class OddsTracker:
    """
    Tracks live NBA games and compares model predictions to Kalshi market odds.
    Saves prediction history to database for analysis.
    """
    
    def __init__(self, kalshi_key_id, kalshi_key_path='key.key', model_type='lr'):
        """
        Initialize the odds tracker.
        
        Args:
            kalshi_key_id: Kalshi API key ID
            kalshi_key_path: Path to Kalshi API private key file
            model_type: Model to use ('lr' or 'xgboost')
        """
        self.db = DatabaseManager()
        self.Session = sessionmaker(bind=self.db.engine)
        
        self.orch = LiveGameOrchestrator(model_type=model_type)
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        
        # Stores matched (NBA game, Kalshi market) pairs
        self.active_matches = []

        
    def setup(self):
        """Authenticate with Kalshi and match today's NBA games to Kalshi markets."""
        # Authenticate with Kalshi API
        if not self.kalshi.login():
            raise Exception("Failed to login to Kalshi")
            
        # Get today's NBA games from NBA API
        nba_games = self.orch.get_todays_games()
        print(f"Found {len(nba_games)} NBA games today.")
        
        # Get active Kalshi NBA markets
        kalshi_events = self.kalshi.get_nba_markets()
        print(f"Found {len(kalshi_events)} active Kalshi NBA events.")
        
        # Match NBA games to Kalshi markets
        self.active_matches = self._match_markets(nba_games, kalshi_events)
        print(f"Matched {len(self.active_matches)} games to markets.")
        
    def _match_markets(self, nba_games, kalshi_events):
        """
        Match NBA games to Kalshi betting markets by team names and date.
        
        Strategy:
        1. Generate today's date code (e.g., "25NOV30" for Nov 30, 2025)
        2. Filter Kalshi events to only today's games
        3. Normalize team city names (remove spaces, handle abbreviations)
        4. Find Kalshi event containing both team cities in title
        5. Extract home team tricode from event ticker (last 3 chars)
        6. Find market ending with "-{HOME_TRICODE}" (this is the "home team wins" market)
        
        Args:
            nba_games: List of NBA game dicts from NBA API
            kalshi_events: List of Kalshi event dicts from Kalshi API
            
        Returns:
            List of dicts with keys: 'nba_game', 'kalshi_ticker', 'kalshi_market'
        """
        matches = []
        
        # Generate today's date code in Kalshi format (e.g., "25NOV30")
        # For late-night games (before 1:30 AM), use yesterday's date
        from datetime import timedelta
        now = datetime.now()
        
        # If before 1:30 AM, use yesterday's date
        if now.hour < 1 or (now.hour == 1 and now.minute < 30):
            game_date = now - timedelta(days=1)
        else:
            game_date = now
        
        year_code = str(game_date.year)[-2:]  # Last 2 digits of year (e.g., "25")
        month_code = game_date.strftime('%b').upper()  # 3-letter month (e.g., "NOV")
        day_code = game_date.strftime('%d')  # 2-digit day (e.g., "30")
        today_date_code = f"{year_code}{month_code}{day_code}"
        
        print(f"Filtering markets for today's date: {today_date_code}")
        
        # Filter events to only today's games
        todays_events = [
            event for event in kalshi_events 
            if today_date_code in event.get('event_ticker', '')
        ]
        print(f"Found {len(todays_events)} events for today (filtered from {len(kalshi_events)} total)")
        
        def normalize(name):
            """Normalize team names for matching."""
            n = name.lower().replace(' ', '').replace('.', '').replace('76ers', 'sixers')
            # Handle city abbreviations
            mapping = {
                'la': 'losangeles',
                'ny': 'newyork',
                'gs': 'goldenstate',
                'no': 'neworleans',
                'sa': 'sanantonio'
            }
            return mapping.get(n, n)
            
        for game in nba_games:
            home_team = game['homeTeam']['teamCity']
            away_team = game['awayTeam']['teamCity']
            
            # Find Kalshi event with both teams in title (from today's events only)
            matched_event = None
            for event in todays_events:
                title = normalize(event.get('title', ''))
                
                if normalize(home_team) in title and normalize(away_team) in title:
                    matched_event = event
                    break
            
            if not matched_event:
                continue
                
            # Get markets for this event (fetch if not included)
            markets = matched_event.get('markets', [])
            if not markets:
                markets = self.kalshi.get_event_markets(matched_event['event_ticker'])
            
            # Extract home team tricode from event ticker
            # Event ticker format: KXNBAGAME-{DATE}{AWAY}{HOME}
            # Example: KXNBAGAME-25NOV30HOUUTA -> home tricode = UTA
            event_ticker = matched_event['event_ticker']
            home_tricode = event_ticker[-3:]
            
            # Find market for "home team wins" (ends with -{HOME_TRICODE})
            target_market = None
            for m in markets:
                if m['ticker'].endswith(f"-{home_tricode}"):
                    target_market = m
                    break
            
            if target_market:
                matches.append({
                    'nba_game': game,
                    'kalshi_ticker': target_market['ticker'],
                    'kalshi_market': target_market
                })
                
        return matches

    def run_loop(self, interval=60):
        """
        Main tracking loop - continuously updates predictions and logs to database.
        
        For each matched game:
        1. Get live game data from NBA API
        2. Calculate model prediction
        3. Get current Kalshi market odds
        4. Log both to database
        5. Display comparison
        
        Args:
            interval: Seconds between updates (default: 60)
        """
        print("Starting tracking loop...")
        session = self.Session()
        
        try:
            while True:
                print(f"\n--- Update at {datetime.now().strftime('%H:%M:%S')} ---")
                
                for match in self.active_matches:
                    game = match['nba_game']
                    ticker = match['kalshi_ticker']
                    
                    # Setup game context for model if needed
                    if self.orch.prediction_engine.current_game_id != game['gameId']:
                        self.orch.setup_game_context(
                            game['gameId'], 
                            game['homeTeam']['teamId'], 
                            game['awayTeam']['teamId']
                        )
                        self.orch.feature_engine.reset()
                        
                    # Get live game data
                    live_data = self.orch.live_client.get_live_game_data(game['gameId'])
                    if not live_data:
                        print(f"Skipping {game['gameCode']} (No live data)")
                        continue
                        
                    # Update feature engine with current game stats
                    home_stats = live_data['homeTeam']['statistics']
                    away_stats = live_data['awayTeam']['statistics']
                    
                    self.orch.feature_engine.home_stats = {
                        'fgm': home_stats['fieldGoalsMade'], 
                        'fga': home_stats['fieldGoalsAttempted'],
                        'fg3m': home_stats['threePointersMade'], 
                        'to': home_stats['turnovers'],
                        'reb': home_stats['reboundsTotal']
                    }
                    self.orch.feature_engine.away_stats = {
                        'fgm': away_stats['fieldGoalsMade'], 
                        'fga': away_stats['fieldGoalsAttempted'],
                        'fg3m': away_stats['threePointersMade'], 
                        'to': away_stats['turnovers'],
                        'reb': away_stats['reboundsTotal']
                    }
                    
                    # Parse game clock
                    remaining_time = 0
                    period = live_data['period']
                    if 'gameClock' in live_data:
                        t_str = live_data['gameClock'].replace('PT', '').replace('M', ':').replace('S', '')
                        if ':' in t_str:
                            m, s = t_str.split(':')
                            remaining_time = int(m) * 60 + float(s)
                    
                    # Calculate total seconds remaining (including future quarters)
                    total_seconds = remaining_time
                    if period <= 4:
                        total_seconds += (4 - period) * 720
                        
                    # Build live features for model
                    live_features = {
                        'score_diff': live_data['homeTeam']['score'] - live_data['awayTeam']['score'],
                        'seconds_remaining': total_seconds,
                        'home_efg': self.orch.feature_engine._calc_efg(self.orch.feature_engine.home_stats),
                        'away_efg': self.orch.feature_engine._calc_efg(self.orch.feature_engine.away_stats),
                        'turnover_diff': self.orch.feature_engine.home_stats['to'] - self.orch.feature_engine.away_stats['to'],
                        'home_rebound_rate': self.orch.feature_engine._calc_reb_rate(
                            self.orch.feature_engine.home_stats['reb'], 
                            self.orch.feature_engine.away_stats['reb']
                        ),
                        'game_id': game['gameId'],
                        'home_team_id': game['homeTeam']['teamId'],
                        'away_team_id': game['awayTeam']['teamId']
                    }
                    
                    # Combine with pre-game context (team stats, roster, etc.)
                    full_feats = {**live_features, **self.orch.prediction_engine.current_game_context}
                    
                    # Get model prediction with confidence intervals
                    model_result = self.orch.prediction_engine.predict_with_confidence(full_feats)
                    model_prob = model_result['probability']
                    
                    # Get Kalshi market prices
                    kalshi_prob = 0.0
                    yes_price = 0
                    no_price = 0
                    
                    details = self.kalshi.get_market_details(ticker)
                    if details:
                        yes_price = details.get('yes_ask', 0)  # Price to buy "home wins"
                        no_price = details.get('no_ask', 0)    # Price to buy "home loses"
                        kalshi_prob = yes_price / 100.0        # Convert cents to probability
                    
                    # Display comparison
                    clock_display = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
                    score_display = f"{live_data['homeTeam']['score']}-{live_data['awayTeam']['score']}"
                    
                    print(f"{game['gameCode']}: {score_display} (Q{period} {clock_display}) | "
                          f"Model {model_prob:.1%} | Kalshi {kalshi_prob:.1%} (Ticker: {ticker})")
                    print(f"   CI 50%: {model_result['ci_50_lower']:.1%}-{model_result['ci_50_upper']:.1%}")
                    print(f"   CI 60%: {model_result['ci_60_lower']:.1%}-{model_result['ci_60_upper']:.1%}")
                    print(f"   CI 95%: {model_result['ci_95_lower']:.1%}-{model_result['ci_95_upper']:.1%}")
                    
                    # Log to database
                    record = OddsHistory(
                        game_id=game['gameId'],
                        timestamp=datetime.utcnow(),
                        model_home_win_prob=float(model_prob),
                        kalshi_home_win_prob=float(kalshi_prob),
                        kalshi_market_ticker=ticker,
                        kalshi_yes_price=int(yes_price),
                        kalshi_no_price=int(no_price),
                        home_team_id=game['homeTeam']['teamId'],
                        away_team_id=game['awayTeam']['teamId']
                    )
                    session.add(record)
                    
                session.commit()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Stopping tracker...")
        finally:
            session.close()
