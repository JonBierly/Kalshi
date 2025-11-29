import time
import pandas as pd
from datetime import datetime
from src.inference.orchestrator import LiveGameOrchestrator
from src.data.kalshi import KalshiClient
from src.data.database import DatabaseManager, OddsHistory
from sqlalchemy.orm import sessionmaker

class OddsTracker:
    def __init__(self, kalshi_key_id, kalshi_key_path='key.key', model_type='lr'):
        """
        Initialize odds tracker.
        
        Args:
            kalshi_key_id: Kalshi API key ID
            kalshi_key_path: Path to Kalshi API key file
            model_type: 'lr' or 'xgboost' for model selection (default: 'lr')
        """
        self.db = DatabaseManager()
        self.Session = sessionmaker(bind=self.db.engine)
        
        self.orch = LiveGameOrchestrator(model_type=model_type)
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        
        self.active_matches = [] # List of (game_info, kalshi_market_ticker)

        
    def setup(self):
        """Authenticates and matches games."""
        # 1. Login to Kalshi
        if not self.kalshi.login():
            raise Exception("Failed to login to Kalshi")
            
        # 2. Get Today's NBA Games
        nba_games = self.orch.get_todays_games()
        print(f"Found {len(nba_games)} NBA games today.")
        
        # 3. Get Kalshi Markets
        kalshi_events = self.kalshi.get_nba_markets()
        print(f"Found {len(kalshi_events)} active Kalshi NBA events.")
        
        # 4. Match
        self.active_matches = self._match_markets(nba_games, kalshi_events)
        print(f"Matched {len(self.active_matches)} games to markets.")
        
    def _match_markets(self, nba_games, kalshi_events):
        matches = []
        
        # Helper to normalize names
        def normalize(name):
            n = name.lower().replace(' ', '').replace('.', '').replace('76ers', 'sixers')
            # Handle common abbreviations that might come from NBA API
            mapping = {
                'la': 'losangeles',
                'ny': 'newyork',
                'gs': 'goldenstate',
                'no': 'neworleans',
                'sa': 'sanantonio'
            }
            return mapping.get(n, n)
            
        for game in nba_games:
            # Kalshi uses City names (e.g. Detroit vs Indiana)
            home_team = game['homeTeam']['teamCity']
            away_team = game['awayTeam']['teamCity']
            
            # Kalshi events usually have titles like "Detroit Pistons vs Indiana Pacers"
            # We look for an event that contains BOTH team names
            
            matched_event = None
            for event in kalshi_events:
                title = normalize(event.get('title', ''))
                # Debug print for first game to see titles
                if game == nba_games[0]:
                    print(f"DEBUG: Comparing '{normalize(home_team)}' & '{normalize(away_team)}' vs '{title}'")
                    
                if normalize(home_team) in title and normalize(away_team) in title:
                    matched_event = event
                    break
            
            if matched_event:
                # Found the event, now find the specific market for "Home Team Wins"
                # Usually tickers are like 'NBA-DATE-AWAY-HOME' or similar.
                # But we can just look at the markets inside the event if available, 
                # or we might need to fetch markets for this event.
                # The get_nba_markets call might return events without full market list.
                # Let's assume we need to fetch markets for the event or infer ticker.
                
                # Actually, let's try to find the market ticker from the event markets if present
                # Or use the event ticker to find child markets.
                # For simplicity, let's assume the event has a 'markets' field or we query it.
                # Kalshi API structure: Event -> Markets.
                # Let's try to fetch markets for this event ticker if we can't find them.
                
                # For now, let's print what we found and try to guess the ticker or fetch it.
                # A common ticker format for "Winner" is the event ticker itself sometimes?
                # No, markets have their own tickers.
                # Let's assume we can get the market list.
                
                # If the event object has 'markets', use it.
                markets = matched_event.get('markets', [])
                
                # If no markets in event, fetch them
                if not markets:
                    markets = self.kalshi.get_event_markets(matched_event['event_ticker'])
                
                # Debug: Print markets for the first matched event
                # if markets:
                #     print(f"DEBUG: Markets for {home_team} vs {away_team}: {[m.get('ticker') for m in markets]}")
                
                target_market = None
                
                # Strategy: Use the Event Ticker to determine the Home Team Suffix
                # Event Ticker: KXNBAGAME-{DATE}{AWAY}{HOME}
                # The last 3 characters should be the Home Team Tricode.
                # We want the market that ends with this Tricode.
                
                event_ticker = matched_event['event_ticker']
                home_tricode = event_ticker[-3:]
                
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
            #         print(f"Matched: {home_team} vs {away_team} -> {target_market['ticker']}")
            #     else:
            #         print(f"Event found for {home_team} vs {away_team} but no market ending in -{home_tricode} found.")
            # else:
            #     print(f"No Kalshi event found for {home_team} vs {away_team}")
                
        return matches

    def run_loop(self, interval=60):
        """Main tracking loop."""
        print("Starting tracking loop...")
        session = self.Session()
        
        try:
            while True:
                print(f"\n--- Update at {datetime.now().strftime('%H:%M:%S')} ---")
                
                for match in self.active_matches:
                    game = match['nba_game']
                    ticker = match['kalshi_ticker']
                    
                    # 1. Get Model Odds
                    # We need to run the orchestrator's prediction logic for a single step
                    # The orchestrator is designed for a loop, but we can extract the single step logic
                    # or just instantiate the engine and update it.
                    
                    # Setup context if not already
                    if self.orch.prediction_engine.current_game_id != game['gameId']:
                        self.orch.setup_game_context(game['gameId'], game['homeTeam']['teamId'], game['awayTeam']['teamId'])
                        self.orch.feature_engine.reset()
                        
                    # Get Live Data
                    live_data = self.orch.live_client.get_live_game_data(game['gameId'])
                    if not live_data:
                        print(f"Skipping {game['gameCode']} (No live data)")
                        continue
                        
                    # Update Feature Engine
                    home_stats = live_data['homeTeam']['statistics']
                    away_stats = live_data['awayTeam']['statistics']
                    
                    self.orch.feature_engine.home_stats = {
                        'fgm': home_stats['fieldGoalsMade'], 'fga': home_stats['fieldGoalsAttempted'],
                        'fg3m': home_stats['threePointersMade'], 'to': home_stats['turnovers'],
                        'reb': home_stats['reboundsTotal']
                    }
                    self.orch.feature_engine.away_stats = {
                        'fgm': away_stats['fieldGoalsMade'], 'fga': away_stats['fieldGoalsAttempted'],
                        'fg3m': away_stats['threePointersMade'], 'to': away_stats['turnovers'],
                        'reb': away_stats['reboundsTotal']
                    }
                    
                    # Calc Features
                    # (Simplified time parsing for now, assuming orchestrator logic is robust enough or we reuse it)
                    # We'll just use the raw update logic from orchestrator if we can, but we can't call 'run_live_loop'.
                    # We manually construct the features.
                    
                    # Time parsing
                    remaining_time = 0
                    period = live_data['period']
                    if 'gameClock' in live_data:
                        t_str = live_data['gameClock'].replace('PT', '').replace('M', ':').replace('S', '')
                        if ':' in t_str:
                            m, s = t_str.split(':')
                            remaining_time = int(m) * 60 + float(s)
                    
                    total_seconds = remaining_time
                    if period <= 4:
                        total_seconds += (4 - period) * 720
                        
                    live_features = {
                        'score_diff': live_data['homeTeam']['score'] - live_data['awayTeam']['score'],
                        'seconds_remaining': total_seconds,
                        'home_efg': self.orch.feature_engine._calc_efg(self.orch.feature_engine.home_stats),
                        'away_efg': self.orch.feature_engine._calc_efg(self.orch.feature_engine.away_stats),
                        'turnover_diff': self.orch.feature_engine.home_stats['to'] - self.orch.feature_engine.away_stats['to'],
                        'home_rebound_rate': self.orch.feature_engine._calc_reb_rate(self.orch.feature_engine.home_stats['reb'], self.orch.feature_engine.away_stats['reb']),
                        'game_id': game['gameId'],
                        'home_team_id': game['homeTeam']['teamId'],
                        'away_team_id': game['awayTeam']['teamId']
                    }
                    
                    full_feats = {**live_features, **self.orch.prediction_engine.current_game_context}
                    
                    model_result = self.orch.prediction_engine.predict_with_confidence(full_feats)
                    model_prob = model_result['probability']
                    
                    # 2. Get Kalshi Odds
                    market_book = self.kalshi.get_orderbook(ticker)
                    kalshi_prob = 0.0
                    yes_price = 0
                    no_price = 0
                    
                    if market_book:
                        # Use the best ask for "Yes" as the buy price (cost to bet on Home Win)
                        # Or use the midpoint.
                        # Kalshi prices are in cents (1-99).
                        # Let's use the last traded price or the best bid/ask average.
                        # For simplicity, let's check the 'yes' bids/asks.
                        # Structure: {'yes': [[price, count], ...], 'no': ...}
                        
                        yes_bids = market_book.get('yes', [])
                        yes_asks = market_book.get('yes', []) # Wait, usually separate
                        
                        # Actually, let's just use the 'get_market_details' for last price or current probability
                        # But orderbook is better for live.
                        # Let's try to get the best 'yes' ask (lowest price to buy 'yes')
                        # If empty, no liquidity.
                        
                        best_ask = 0
                        if yes_asks:
                            # Sort by price ascending? usually API returns sorted?
                            # Let's assume list of [price, qty].
                            # We want lowest price.
                            # Actually, let's just fetch market details for 'last_price' or 'yes_bid' / 'yes_ask'
                            pass
                            
                        # Fallback to market details for simplicity
                        details = self.kalshi.get_market_details(ticker)
                        if details:
                            yes_price = details.get('yes_ask', 0)
                            no_price = details.get('no_ask', 0)
                            # Implied prob is roughly price / 100
                            kalshi_prob = yes_price / 100.0
                    
                    # Format time for display
                    clock_display = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
                    
                    print(f"{game['gameCode']}: {live_data['homeTeam']['score']}-{live_data['awayTeam']['score']} (Q{period} {clock_display}) | Model {model_prob:.1%} | Kalshi {kalshi_prob:.1%} (Ticker: {ticker})")
                    print(f"   CI 50%: {model_result['ci_50_lower']:.1%}-{model_result['ci_50_upper']:.1%}")
                    print(f"   CI 60%: {model_result['ci_60_lower']:.1%}-{model_result['ci_60_upper']:.1%}")
                    print(f"   CI 95%: {model_result['ci_95_lower']:.1%}-{model_result['ci_95_upper']:.1%}")
                    
                    # 3. Log to DB
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
