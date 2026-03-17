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

        print("  Discovering today's games from Kalshi events...")

        # Build tricode -> team_id map from local static data (no network call needed)
        try:
            from nba_api.stats.static import teams as nba_teams_static
            tri_to_id = {t['abbreviation']: t['id'] for t in nba_teams_static.get_teams()}
        except Exception:
            tri_to_id = {}

        # Discover today's games via Kalshi events API (no stats.nba.com needed!)
        spread_events = self.kalshi.get_todays_spread_events()

        if not spread_events:
            print("  ⚠️ No open KXNBASPREAD events found on Kalshi for today")
            return

        for event in spread_events:
            event_ticker = event['event_ticker']
            home_tri     = event['home_tri']
            away_tri     = event['away_tri']

            # Resolve numeric team IDs from static map (needed for feature engine)
            home_id = tri_to_id.get(home_tri, 0)
            away_id = tri_to_id.get(away_tri, 0)

            try:
                markets = self.kalshi.get_event_markets(event_ticker)
                if not markets:
                    continue

                spread_markets = []
                for market in markets:
                    try:
                        parsed = parse_spread_ticker(market['ticker'])
                        spread_markets.append(SpreadMarket(
                            ticker=market['ticker'],
                            team=parsed['spread_team'],
                            spread=parsed['spread_value'],
                            yes_bid=market.get('yes_bid') if market.get('yes_bid') is not None else int(float(market.get('yes_bid_dollars') or 0) * 100),
                            yes_ask=market.get('yes_ask') if market.get('yes_ask') is not None else int(float(market.get('yes_ask_dollars') or 0) * 100),
                            no_bid=market.get('no_bid', 0),
                            no_ask=market.get('no_ask', 0),
                            subtitle=market.get('title', '')
                        ))
                    except:
                        continue

                if spread_markets:
                    game = {
                        'gameId':    event_ticker,  # placeholder — resolved below from live CDN
                        'gameStatus': 1,             # assume not started yet
                        'homeTeam': {'teamId': home_id, 'teamTricode': home_tri},
                        'awayTeam': {'teamId': away_id, 'teamTricode': away_tri},
                        'gameCode': f"{away_tri}@{home_tri}",
                    }
                    self.active_matches.append({'nba_game': game, 'spread_markets': spread_markets})
                    print(f"  ✓ {away_tri} @ {home_tri} → {len(spread_markets)} spread markets")
            except Exception as e:
                print(f"  ⚠️ Error loading {away_tri}@{home_tri}: {e}")
                continue

        # Immediately try to resolve real NBA game IDs from the live CDN scoreboard
        self.resolve_live_game_ids()


    def resolve_live_game_ids(self):
        """
        Query the NBA live CDN scoreboard (S3 — NOT stats.nba.com, not blocked on VPS)
        and map real numeric game IDs onto active_matches by matching team tricodes.
        Called at setup and at the start of each trading iteration to pick up
        games that have just tipped off.
        """
        from data.s3_client import S3DataClient
        s3 = S3DataClient()

        try:
            from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
            board = live_scoreboard.ScoreBoard()
            live_games = board.games.get_dict()
            
            # CHECK FOR STALENESS: Is the scoreboard still yesterday's?
            # scoreboard.gameDate is 'YYYY-MM-DD'
            from datetime import datetime
            from zoneinfo import ZoneInfo
            today_et = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
            board_date = getattr(board, 'game_date', getattr(board, 'gameDate', ''))
            
            if not board_date or board_date < today_et:
                print(f"  ⚠️ S3 Scoreboard is stale ({board_date} vs {today_et}). Triggering Smart Scan...")
                smart_ids = s3.get_todays_game_ids()
                if smart_ids:
                    # Fetch individual boxscores for these IDs
                    live_games = []
                    for gid in smart_ids:
                        info = s3.get_game_info(gid)
                        if info:
                            # Reconstruct minimal lg dict matching scoreboard format
                            live_games.append({
                                'gameId': gid,
                                'gameStatus': info['status'],
                                'homeTeam': {'teamTricode': info['home_tri']},
                                'awayTeam': {'teamTricode': info['away_tri']}
                            })
        except Exception as e:
            print(f"  ⚠️ Could not fetch live scoreboard for ID resolution: {e}")
            return

        # Build (away_tri, home_tri) -> (real_game_id, game_status) from live data
        live_id_map = {}
        for lg in live_games:
            a_tri = lg.get('awayTeam', {}).get('teamTricode', '')
            h_tri = lg.get('homeTeam', {}).get('teamTricode', '')
            gid   = lg.get('gameId', '')
            gstat = lg.get('gameStatus', 1)  # 1=not started, 2=live, 3=final
            if a_tri and h_tri and gid:
                live_id_map[(a_tri, h_tri)] = (gid, gstat)

        for match in self.active_matches:
            game = match['nba_game']
            key = (game['awayTeam']['teamTricode'], game['homeTeam']['teamTricode'])
            if key in live_id_map:
                real_id, gstat = live_id_map[key]
                if game['gameId'] != real_id:
                    print(f"  🔗 Resolved {game['gameCode']} → {real_id} (status {gstat})")
                game['gameId']     = real_id
                game['gameStatus'] = gstat



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
                        'oreb': h_stats.get('reboundsOffensive', 0),
                        'stl': h_stats.get('steals', 0),
                        'blk': h_stats.get('blocks', 0)
                    })
                    self.orch.feature_engine.away_stats.update({
                        'fgm': a_stats.get('fieldGoalsMade', 0),
                        'fga': a_stats.get('fieldGoalsAttempted', 1),
                        'fg3m': a_stats.get('threePointersMade', 0),
                        'to': a_stats.get('turnovers', 0),
                        'reb': a_stats.get('reboundsTotal', 0),
                        'pts': a_pts,
                        'fta': a_stats.get('freeThrowsAttempted', 0),
                        'oreb': a_stats.get('reboundsOffensive', 0),
                        'stl': a_stats.get('steals', 0),
                        'blk': a_stats.get('blocks', 0)
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
                            market.yes_bid = fresh.get('yes_bid') if fresh.get('yes_bid') is not None else int(float(fresh.get('yes_bid_dollars') or 0) * 100)
                            market.yes_ask = fresh.get('yes_ask') if fresh.get('yes_ask') is not None else int(float(fresh.get('yes_ask_dollars') or 0) * 100)
                        except: pass
                        
                        prediction = self.spread_model.predict_for_market(full_feats, market.spread, market.team, game['homeTeam']['teamTricode'])
                        prob = prediction['probability']
                        edge = prob - (market.yes_ask / 100.0)
                        edge_str = f"{edge:+.1%}"
                        if edge > 0.04: edge_str = f"🔥 {edge_str}"
                        print(f"  {market.team} >{market.spread}: {prob:.1%} | Ask={market.yes_ask}¢ | Edge: {edge_str}")
                
                time.sleep(interval)
        except KeyboardInterrupt: pass
