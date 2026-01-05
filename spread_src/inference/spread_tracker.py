"""
Spread Market Tracker - Track and compare spread market predictions.
"""

import time
import pandas as pd
from datetime import datetime
from src.inference.orchestrator import LiveGameOrchestrator
from data.kalshi import KalshiClient
from spread_src.data.spread_markets import parse_spread_ticker, SpreadMarket, check_spread_arbitrage
from spread_src.models.spread_model import SpreadDistributionModel
from scipy import stats
import numpy as np


class SpreadTracker:
    """
    Tracks live NBA games and compares spread predictions to Kalshi spread markets.
    """
    
    def __init__(self, kalshi_key_id, kalshi_key_path='key.key', model_path='models/nba_spread_ngboost.pkl'):
        # Reuse orchestrator for live game data
        self.orch = LiveGameOrchestrator(model_type='lr', skip_model=True)
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        self.spread_model = SpreadDistributionModel(model_path)
        self.active_matches = []
        from spread_src.features.engineering import FeatureEngine
        self.feature_engines = {}
    
    def setup(self):
        if not self.kalshi.login():
            raise Exception("Failed to login to Kalshi")
        
        nba_games = self.orch.get_todays_games()
        from datetime import datetime, timedelta
        now = datetime.now()
        if now.hour < 1 or (now.hour == 1 and now.minute < 30):
            game_date = now - timedelta(days=1)
        else:
            game_date = now
        
        today_date_code = f"{str(game_date.year)[-2:]}{game_date.strftime('%b').upper()}{game_date.strftime('%d')}"
        
        for game in nba_games:
            home_tri = game['homeTeam']['teamTricode']
            away_tri = game['awayTeam']['teamTricode']
            event_ticker = f"KXNBASPREAD-{today_date_code}{away_tri}{home_tri}"
            
            try:
                markets = self.kalshi.get_event_markets(event_ticker)
                if not markets: continue
                
                spread_markets = []
                for market in markets:
                    try:
                        parsed = parse_spread_ticker(market['ticker'])
                        spread_markets.append(SpreadMarket(
                            ticker=market['ticker'],
                            team=parsed['spread_team'],
                            spread=parsed['spread_value'],
                            yes_bid=market.get('yes_bid', 0),
                            yes_ask=market.get('yes_ask', 0),
                            no_bid=market.get('no_bid', 0),
                            no_ask=market.get('no_ask', 0),
                            subtitle=market.get('title', '')
                        ))
                    except: continue
                
                if spread_markets:
                    self.active_matches.append({'nba_game': game, 'spread_markets': spread_markets})
            except: continue
    
    def predict_spread_distribution(self, live_features):
        result = self.spread_model.predict_spread_probabilities(live_features, thresholds=[0.0])
        return result['mean_diff'], result['std_diff']
    
    def run_loop(self, interval=30):
        print("Starting spread tracking loop...\n")
        try:
            while True:
                print(f"\nUpdate at {datetime.now().strftime('%H:%M:%S')}")
                for match in self.active_matches:
                    game = match['nba_game']
                    spread_markets = match['spread_markets']
                    game_id = game['gameId']
                    
                    if game_id not in self.feature_engines:
                        from spread_src.features.engineering import FeatureEngine
                        self.feature_engines[game_id] = FeatureEngine()
                        
                    if self.orch.prediction_engine.current_game_id != game_id:
                        self.orch.setup_game_context(game_id, game['homeTeam']['teamId'], game['awayTeam']['teamId'])
                    
                    self.orch.feature_engine = self.feature_engines[game_id]
                    live_data = self.orch.live_client.get_live_game_data(game_id)
                    if not live_data: continue
                    
                    # Update stats
                    h_stats = live_data['homeTeam']['statistics']
                    a_stats = live_data['awayTeam']['statistics']
                    h_pts = live_data['homeTeam']['score']
                    a_pts = live_data['awayTeam']['score']
                    
                    self.orch.feature_engine.home_stats.update({
                        'fgm': h_stats.get('fieldGoalsMade', 0),
                        'fga': h_stats.get('fieldGoalsAttempted', 1),
                        'fg3m': h_stats.get('threePointersMade', 0),
                        'to': h_stats.get('turnovers', 0),
                        'reb': h_stats.get('reboundsTotal', 0),
                        'pts': h_pts,
                        'fta': h_stats.get('freeThrowsAttempted', 0),
                        'oreb': h_stats.get('reboundsOffensive', 0)
                    })
                    self.orch.feature_engine.away_stats.update({
                        'fgm': a_stats.get('fieldGoalsMade', 0),
                        'fga': a_stats.get('fieldGoalsAttempted', 1),
                        'fg3m': a_stats.get('threePointersMade', 0),
                        'to': a_stats.get('turnovers', 0),
                        'reb': a_stats.get('reboundsTotal', 0),
                        'pts': a_pts,
                        'fta': a_stats.get('freeThrowsAttempted', 0),
                        'oreb': a_stats.get('reboundsOffensive', 0)
                    })
                    
                    import re
                    match_clock = re.search(r'PT(\d+)M([\d.]+)S', live_data.get('gameClock', 'PT0M0.00S'))
                    remaining_time = (int(match_clock.group(1)) * 60 + float(match_clock.group(2))) if match_clock else 0
                    
                    period = live_data['period']
                    total_seconds = remaining_time + (max(0, 4 - period) * 720)
                    score_diff = h_pts - a_pts
                    
                    live_features = self.orch.feature_engine.calculate_current_features(
                        score_diff, total_seconds, period, game_id, 
                        game['homeTeam']['teamId'], game['awayTeam']['teamId']
                    )
                    
                    full_feats = {**live_features, **self.orch.prediction_engine.current_game_context}
                    mean_diff, std_diff = self.predict_spread_distribution(full_feats)
                    
                    print(f"\n{game['gameCode']}: {a_pts}@{h_pts} | Q{period} {int(remaining_time//60)}:{int(remaining_time%60):02d}")
                    print(f"Model: μ={mean_diff:+.1f}, σ={std_diff:.1f}")
                    
                    for market in spread_markets:
                        try:
                            fresh = self.kalshi.get_market_details(market.ticker)
                            market.yes_bid = fresh.get('yes_bid', 0)
                            market.yes_ask = fresh.get('yes_ask', 0)
                        except: pass
                        
                        prediction = self.spread_model.predict_for_market(full_feats, market.spread, market.team, game['homeTeam']['teamTricode'])
                        prob = prediction['probability']
                        edge = prob - (market.yes_ask / 100.0)
                        edge_str = f"{edge:+.1%}"
                        if edge > 0.04: edge_str = f"🔥 {edge_str}"
                        print(f"  {market.team} >{market.spread}: {prob:.1%} | Ask={market.yes_ask}¢ | Edge: {edge_str}")
                
                time.sleep(interval)
        except KeyboardInterrupt: pass
