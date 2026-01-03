"""
Spread Market Tracker - Track and compare spread market predictions.

Similar to OddsTracker but for spread markets instead of binary win/loss.
"""

import time
import pandas as pd
from datetime import datetime
from src.inference.orchestrator import LiveGameOrchestrator
from data.kalshi import KalshiClient
from spread_src.data.spread_markets import parse_spread_ticker, SpreadMarket, check_spread_arbitrage
from spread_src.models.spread_model import SpreadDistributionModel
import joblib
from scipy import stats
import numpy as np


class SpreadTracker:
    """
    Tracks live NBA games and compares spread predictions to Kalshi spread markets.
    """
    
    def __init__(self, kalshi_key_id, kalshi_key_path='key.key'):
        """
        Initialize the spread tracker.
        
        Args:
            kalshi_key_id: Kalshi API key ID
            kalshi_key_path: Path to Kalshi API private key file
        """
        # Reuse orchestrator for live game data
        self.orch = LiveGameOrchestrator(model_type='lr')
        
        # Kalshi client
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        
        # Load spread model
        print("Loading spread model...")
        self.spread_models = joblib.load('models/nba_spread_model.pkl')
        print(f"✓ Loaded {len(self.spread_models)} spread models")
        
        # Active matches: (game, spread_markets)
        self.active_matches = []
        
        # Stateful feature engines per game_id
        from spread_src.features.engineering import FeatureEngine
        self.feature_engines = {}
    
    def setup(self):
        """Match today's NBA games to Kalshi spread markets."""
        # Authenticate
        if not self.kalshi.login():
            raise Exception("Failed to login to Kalshi")
        
        # Get today's NBA games
        nba_games = self.orch.get_todays_games()
        print(f"Found {len(nba_games)} NBA games today.")
        
        # Build date code (e.g., "25DEC01")
        # For late-night games (before 1:30 AM), use yesterday's date
        # since the game started yesterday
        from datetime import datetime, timedelta
        now = datetime.now()
        
        # If before 1:30 AM, use yesterday's date
        if now.hour < 1 or (now.hour == 1 and now.minute < 30):
            game_date = now - timedelta(days=1)
            print(f"Using yesterday's date (late-night game)")
        else:
            game_date = now
        
        year_code = str(game_date.year)[-2:]
        month_code = game_date.strftime('%b').upper()
        day_code = game_date.strftime('%d')
        today_date_code = f"{year_code}{month_code}{day_code}"
        
        print(f"Date code: {today_date_code}")
        print("\nFetching spread markets for each game...")
        
        # For each game, build event ticker and fetch markets
        for game in nba_games:
            home_tri = game['homeTeam']['teamTricode']
            away_tri = game['awayTeam']['teamTricode']
            
            # Build event ticker: KXNBASPREAD-{DATE}{AWAY}{HOME}
            event_ticker = f"KXNBASPREAD-{today_date_code}{away_tri}{home_tri}"
            
            print(f"\n  {away_tri} @ {home_tri}")
            print(f"    Event ticker: {event_ticker}")
            
            # Fetch markets for this event
            try:
                markets = self.kalshi.get_event_markets(event_ticker)
                
                if not markets:
                    print(f"    ✗ No markets found")
                    continue
                
                print(f"    ✓ Found {len(markets)} spread markets")
                
                # Parse markets into SpreadMarket objects
                spread_markets = []
                for market in markets:
                    try:
                        ticker = market['ticker']
                        print(f"      DEBUG: Parsing ticker: {ticker}")
                        
                        parsed = parse_spread_ticker(ticker)
                        
                        spread_market = SpreadMarket(
                            ticker=market['ticker'],
                            team=parsed['spread_team'],
                            spread=parsed['spread_value'],
                            yes_bid=market.get('yes_bid', 0),
                            yes_ask=market.get('yes_ask', 0),
                            no_bid=market.get('no_bid', 0),
                            no_ask=market.get('no_ask', 0),
                            subtitle=market.get('title', '')
                        )
                        
                        spread_markets.append(spread_market)
                        print(f"      - {spread_market.team} >{spread_market.spread}: {spread_market.yes_ask}¢")
                        
                    except Exception as e:
                        print(f"      Error parsing market '{ticker}': {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Add to active matches
                if spread_markets:
                    self.active_matches.append({
                        'nba_game': game,
                        'spread_markets': spread_markets
                    })
                    
            except Exception as e:
                print(f"    ✗ Error fetching markets: {e}")
                continue
        
        print(f"\n✓ Matched {len(self.active_matches)} games with spread markets")
    
    def predict_spread_distribution(self, live_features):
        """
        Predict spread distribution using ensemble.
        
        Model predicts SCORE REMAINDER (how much margin will change).
        We add current_diff to get expected final diff.
        
        Variance model predicts variance (squared error), we take sqrt for std.
        
        Returns mean and std of final score differential.
        """
        # Prepare features
        from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST
        feature_order = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
        row_data = {col: live_features.get(col, 0.0) for col in feature_order}
        X_live = pd.DataFrame([row_data], columns=feature_order)
        
        # Get current score diff for reconstruction
        current_diff = live_features.get('score_diff', 0.0)
        
        # Get predictions from ensemble (these are REMAINDERS, not final diffs)
        remainder_preds = []
        std_preds = []
        
        for model_pair in self.spread_models:
            remainder_pred = model_pair['mean_model'].predict(X_live)[0]
            
            # Support both old (std_model) and new (variance_model) formats
            if 'variance_model' in model_pair:
                variance_pred = model_pair['variance_model'].predict(X_live)[0]
                std_pred = np.sqrt(max(variance_pred, 1.0))  # sqrt(variance) = std
            else:
                std_pred = model_pair['std_model'].predict(X_live)[0]
            
            remainder_preds.append(remainder_pred)
            std_preds.append(max(std_pred, 1.0))
        
        # Reconstruct expected final diff = current_diff + predicted_remainder
        mean_remainder = np.mean(remainder_preds)
        mean = current_diff + mean_remainder  # Expected final diff
        std = np.mean(std_preds)
        
        return mean, std
    
    def run_loop(self, interval=30):
        """
        Main tracking loop for spread markets.
        
        For each game:
        1. Get live data
        2. Predict spread distribution
        3. For each spread market, compute P(diff > threshold)
        4. Compare to market price
        5. Show opportunities
        """
        print("Starting spread tracking loop...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                print(f"\n{'='*80}")
                print(f"Update at {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*80}\n")
                
                for match in self.active_matches:
                    game = match['nba_game']
                    spread_markets = match['spread_markets']
                    
                    # Setup game context and use correct feature engine
                    game_id = game['gameId']
                    if game_id not in self.feature_engines:
                        from spread_src.features.engineering import FeatureEngine
                        self.feature_engines[game_id] = FeatureEngine()
                        
                    if self.orch.prediction_engine.current_game_id != game_id:
                        self.orch.setup_game_context(
                            game_id,
                            game['homeTeam']['teamId'],
                            game['awayTeam']['teamId']
                        )
                    
                    # Switch the orchestrator to use this game's specific engine
                    self.orch.feature_engine = self.feature_engines[game_id]
                    
                    # Get live data
                    live_data = self.orch.live_client.get_live_game_data(game['gameId'])
                    if not live_data:
                        print(f"Skipping {game['gameCode']} (no live data)")
                        continue
                    
                    # Build live features (same as binary tracker)
                    home_stats = live_data['homeTeam']['statistics']
                    away_stats = live_data['awayTeam']['statistics']
                    
                    self.orch.feature_engine.home_stats.update({
                        'fgm': home_stats['fieldGoalsMade'],
                        'fga': home_stats['fieldGoalsAttempted'],
                        'fg3m': home_stats['threePointersMade'],
                        'to': home_stats['turnovers'],
                        'reb': home_stats['reboundsTotal'],
                        'pts': live_data['homeTeam']['score'],
                        'fta': home_stats['freeThrowsAttempted'],
                        'oreb': home_stats['reboundsOffensive']
                    })
                    self.orch.feature_engine.away_stats.update({
                        'fgm': away_stats['fieldGoalsMade'],
                        'fga': away_stats['fieldGoalsAttempted'],
                        'fg3m': away_stats['threePointersMade'],
                        'to': away_stats['turnovers'],
                        'reb': away_stats['reboundsTotal'],
                        'pts': live_data['awayTeam']['score'],
                        'fta': away_stats['freeThrowsAttempted'],
                        'oreb': away_stats['reboundsOffensive']
                    })
                    
                    # Calculate seconds remaining
                    period = live_data['period']
                    remaining_time = 0
                    if 'gameClock' in live_data:
                        t_str = live_data['gameClock'].replace('PT', '').replace('M', ':').replace('S', '')
                        if ':' in t_str:
                            m, s = t_str.split(':')
                            remaining_time = int(m) * 60 + float(s)
                    
                    total_seconds = remaining_time
                    if period <= 4:
                        total_seconds += (4 - period) * 720
                    
                    # Update history and calculate features
                    score_diff = live_data['homeTeam']['score'] - live_data['awayTeam']['score']
                    self.orch.feature_engine.history.append((total_seconds, score_diff))
                    if len(self.orch.feature_engine.history) > 1000:
                        self.orch.feature_engine.history = [h for h in self.orch.feature_engine.history if h[0] < total_seconds + 600]

                    live_features = self.orch.feature_engine.calculate_current_features(
                        score_diff,
                        total_seconds,
                        game['gameId'],
                        game['homeTeam']['teamId'],
                        game['awayTeam']['teamId']
                    )
                    live_features['is_home'] = 1 # Keep for legacy compatibility
                    
                    # Combine with context
                    full_feats = {**live_features, **self.orch.prediction_engine.current_game_context}
                    
                    # Get spread distribution prediction
                    mean_diff, std_diff = self.predict_spread_distribution(full_feats)
                    
                    # Display game info
                    home_tri = game['homeTeam']['teamTricode']
                    away_tri = game['awayTeam']['teamTricode']
                    score = f"{away_tri} {live_data['awayTeam']['score']} @ {home_tri} {live_data['homeTeam']['score']}"
                    clock = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
                    
                    print(f"\n{game['gameCode']}: {score} | Q{period} {clock}")
                    print(f"Stats: Pace={live_features['live_pace']:.1f}, Momentum={live_features['score_momentum']:.1f}, 3P%={live_features['home_3p_reliance']:.1%}/{live_features['away_3p_reliance']:.1%}")
                    print(f"Model: μ={mean_diff:+.1f}, σ={std_diff:.1f}")
                    print(f"\nSpread Markets:")
                    
                    # Refresh market prices (they get stale!)
                    print(f"\nRefreshing market prices...")
                    for market in spread_markets:
                        try:
                            fresh_data = self.kalshi.get_market_details(market.ticker)
                            if fresh_data:
                                market.yes_bid = fresh_data.get('yes_bid', market.yes_bid)
                                market.yes_ask = fresh_data.get('yes_ask', market.yes_ask)
                                market.no_bid = fresh_data.get('no_bid', market.no_bid)
                                market.no_ask = fresh_data.get('no_ask', market.no_ask)
                        except Exception as e:
                            print(f"  Warning: Failed to refresh {market.ticker}: {e}")
                    
                    # Check for arbitrage first
                    arb = check_spread_arbitrage(spread_markets)
                    if arb:
                        print(f"\n⚡ ARBITRAGE DETECTED:")
                        for opp in arb:
                            print(f"  {opp['strategy']}")
                            print(f"  Profit: {opp['arbitrage_cents']:.1f}¢ risk-free!")
                    
                    # For each spread market, compute edge
                    print()
                    for market in spread_markets:
                        # Determine if home or away
                        home_team_tri = game['homeTeam']['teamTricode']
                        is_home_market = (market.team == home_team_tri)
                        
                        threshold = market.spread
                        
                        # Model probability
                        dist = stats.norm(loc=mean_diff, scale=std_diff)
                        
                        if is_home_market:
                            # Home wins by >threshold
                            model_prob = 1 - dist.cdf(threshold)
                        else:
                            # Away wins by >threshold
                            # = Home loses by >threshold
                            # = Home diff < -threshold
                            model_prob = dist.cdf(-threshold)
                        
                        # Market prices (bid/ask)
                        # To BUY YES: pay yes_ask
                        # To SELL YES: receive yes_bid
                        yes_bid = market.yes_bid / 100.0
                        yes_ask = market.yes_ask / 100.0
                        
                        # Edge for buying (most common)
                        buy_edge = model_prob - yes_ask
                        
                        # Display
                        edge_str = f"{buy_edge:+.1%}"
                        if abs(buy_edge) > 0.05:
                            edge_str = f"🔥 {edge_str}"
                        
                        print(f"  {market.team} >{threshold}: Model {model_prob:.1%} | Bid={market.yes_bid}¢ Ask={market.yes_ask}¢ | Buy Edge: {edge_str}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping tracker...")
