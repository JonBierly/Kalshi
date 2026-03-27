#!/usr/bin/env python
"""
Simple +EV Trader for Kalshi spread markets.

Clean, simplified trading strategy:
1. Model predicts P(spread > threshold)
2. Compare to market bid/ask
3. If edge >= 4%, place order at beat-by-1 price
4. Cancel orders when edge drops or price uncompetitive

Usage:
    python -m spread_src.scripts.simple_live_trader
"""

import os
import sys
import time
import re
import json
from datetime import datetime
from typing import Optional, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.trading.simple_trader import SimpleEdgeTrader, TradeOpportunity
from spread_src.execution.portfolio import Portfolio
from spread_src.execution.order_manager import OrderManager
from spread_src.execution.trade_logger import TradeLogger
from spread_src.execution.risk_manager import RiskManager
from spread_src.trading.position_sizer import PositionSizer
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.inference.spread_tracker import SpreadTracker
from spread_src.features.engineering import add_interaction_features
from data.kalshi import KalshiClient
from spread_src.patches import apply_patch

class SimpleLiveTrader:
    """
    Simple +EV trading engine.
    
    - Finds edges based on model vs market prices
    - Places orders at beat-by-1 prices
    - Manages exposure per game
    """
    
    def __init__(
        self,
        kalshi_key_id: str,
        kalshi_key_path: str = 'key.key',
        dry_run: bool = True,
        max_game_exposure: float = 15.0,
        max_ticker_exposure: float = 5.0,
        min_edge: float = 0.08,
        min_spread: int = 4,
    ):
        """
        Initialize trader.
        
        Args:
            kalshi_key_id: Kalshi API key
            kalshi_key_path: Path to private key
            dry_run: If True, log orders without executing
            max_game_exposure: Max $ at risk per game (default $10)
            max_ticker_exposure: Max $ at risk per ticker (default $3)
            min_edge: Minimum edge to trade (default 4%)
            min_spread: Minimum bid-ask spread in cents to trade (default 4)
        """
        self.dry_run = dry_run
        self.max_game_exposure = max_game_exposure
        self.max_ticker_exposure = max_ticker_exposure
        self.min_spread = min_spread
        
        print("=" * 80)
        print("SIMPLE +EV TRADER")
        print("=" * 80)
        print(f"Mode: {'DRY-RUN (simulation)' if dry_run else 'LIVE (real money!)'}")
        print(f"Max per game: ${max_game_exposure}")
        print(f"Max per ticker: ${max_ticker_exposure}")
        print(f"Min edge: {min_edge:.0%}")
        print(f"Min spread: {min_spread}¢")
        print("=" * 80)
        
        # Initialize components
        print("\n✓ Initializing...")
        
        # Apply API patches to bypass NBA CDN blocks BEFORE starting other services
        try:
            apply_patch()
        except Exception as e:
            print(f"⚠️  Failed to apply NBA API patches: {e}")
            
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        self.portfolio = Portfolio(max_exposure=max_game_exposure * 12)  # Allow 12 games
        self.portfolio.refresh_state(self.kalshi)
        
        self.order_mgr = OrderManager(self.kalshi, dry_run=dry_run)
        self.trader = SimpleEdgeTrader(min_edge=min_edge, cancel_threshold=0.02)
        self.risk_mgr = RiskManager(max_game_exposure * 12, max_game_exposure)
        self.position_sizer = PositionSizer(kelly_fraction=0.25)  # Quarter-Kelly
        
        # CRITICAL: Build event_tickers map from synced positions
        # Format: {event_prefix: [ticker1, ticker2, ...]}
        # This ensures exposure calculation includes ALL positions for each game
        self.event_tickers = {}
        for ticker, pos in self.portfolio.positions.items():
            if pos != 0:
                # Extract event prefix (e.g., KXNBASPREAD-25DEC10PHXOKC from KXNBASPREAD-25DEC10PHXOKC-OKC15)
                parts = ticker.rsplit('-', 1)
                event_prefix = parts[0] if len(parts) >= 2 else ticker
                if event_prefix not in self.event_tickers:
                    self.event_tickers[event_prefix] = []
                self.event_tickers[event_prefix].append(ticker)
        
        if self.event_tickers:
            print(f"\n📍 Tracking {len(self.event_tickers)} events with positions:")
            for event_prefix, tickers in self.event_tickers.items():
                event_short = event_prefix.split('-')[-1] if '-' in event_prefix else event_prefix
                print(f"  {event_short}: {len(tickers)} markets")
        
        # Initialize tracker (loads NGBoost model internally)
        model_path = 'models/nba_spread_ngboost_v3_final.pkl'
        self.tracker = SpreadTracker(kalshi_key_id, kalshi_key_path, model_path=model_path)
        
        # Reuse the model from tracker
        self.spread_model = self.tracker.spread_model
        
        # Database logging
        self.trade_logger = TradeLogger('data/nba_data.db')
        
        # Warm-up tracking
        self.game_tracking_start = {}  # {game_id: start_time}
        self.latest_market_data = {}   # {ticker: {bid, ask, model_prob}}
        print("✓ Ready")
    
    def run(self, interval: int = 10):
        """Main trading loop."""
        print("\nMatching games to spread markets...")
        self.tracker.setup()
        
        if not self.tracker.active_matches:
            print("⚠️ No games matched to spread markets")
            return
        
        print(f"✓ Tracking {len(self.tracker.active_matches)} game(s)\n")
        print("Starting trading loop... Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'=' * 80}")
                print(f"Iteration {iteration} @ {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'=' * 80}")
                
                # Refresh state
                self.portfolio.refresh_state(self.kalshi)
                self.latest_market_data = {} # Reset for this iteration

                # Refresh real NBA game IDs (picks up games that just tipped off)
                self.tracker.resolve_live_game_ids()


                # Check fills
                fills = self.order_mgr.check_for_fills()
                for fill in fills:
                    trade_id = self.trade_logger.get_trade_id_by_order_id(fill.order_id)
                    self.portfolio.update_fill(
                        ticker=fill.ticker,
                        side=fill.side,
                        price=fill.price,
                        size=fill.size,
                        trade_id=trade_id
                    )
                    print(f"  ✅ FILL: {fill.side} {fill.size} @ {fill.price}¢")
                
                # Evaluate and trade each game
                all_opportunities = []
                
                for match in self.tracker.active_matches:
                    game = match['nba_game']
                    spread_markets = match['spread_markets']
                    
                    opps = self._evaluate_game(game, spread_markets)
                    if opps:
                        all_opportunities.extend(opps)
                
                # Manage existing orders
                self._manage_orders(all_opportunities)
                
                # Place new orders
                self._place_orders(all_opportunities)
                
                # Export Dashboard State
                self._export_dashboard_state()
                
                # Print status
                self._print_status()
                
                print(f"\nWaiting {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping trader...")
            
            # Cancel all pending orders and log them
            open_orders = self.order_mgr.get_open_orders()
            for order in open_orders:
                self._cancel_order_with_logging(order.order_id)
            
            # Settle any finalized markets
            print("\n🔍 Checking for settled markets...")
            self.portfolio.settle_unsettled_trades(self.kalshi, self.trade_logger)
            
            self._print_status()
            print("✓ Shutdown complete")
    
    def _evaluate_game(self, game, spread_markets) -> list:
        """Evaluate a game and find +EV opportunities."""
        game_id = game['gameId']
        
        # Setup context and use the correct stateful engine for this game
        if game_id not in self.tracker.feature_engines:
            from spread_src.features.engineering import FeatureEngine
            self.tracker.feature_engines[game_id] = FeatureEngine()
            
        if self.tracker.orch.prediction_engine.current_game_id != game_id:
            self.tracker.orch.setup_game_context(
                game_id,
                game['homeTeam']['teamId'],
                game['awayTeam']['teamId']
            )
        
        # Switch the orchestrator to use this game's specific engine
        self.tracker.orch.feature_engine = self.tracker.feature_engines[game_id]
        
        # Get live data
        live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
        if not live_data:
            return []
        
        # Check if game has actually started (NBA games start ~10 min after listed time)
        period = live_data.get('period', 0)
        home_score = live_data.get('homeTeam', {}).get('score', 0)
        away_score = live_data.get('awayTeam', {}).get('score', 0)
        game_status = live_data.get('gameStatus', 1)  # 1=Not Started, 2=In Progress, 3=Finished
        
        if game_status == 1 or (period == 0 and home_score == 0 and away_score == 0):
            home_tri = game['homeTeam']['teamTricode']
            away_tri = game['awayTeam']['teamTricode']
            print(f"\n{away_tri} @ {home_tri} - Game not started yet, skipping")
            return []
        
        # CRITICAL: Update feature_engine with live boxscore stats
        if 'statistics' in live_data.get('homeTeam', {}):
            home_stats = live_data['homeTeam']['statistics']
            away_stats = live_data['awayTeam']['statistics']
            
            self.tracker.orch.feature_engine.home_stats.update({
                'fgm': home_stats.get('fieldGoalsMade', 0),
                'fga': home_stats.get('fieldGoalsAttempted', 1),
                'fg3m': home_stats.get('threePointersMade', 0),
                'to': home_stats.get('turnovers', 0),
                'reb': home_stats.get('reboundsTotal', 0),
                'pts': live_data['homeTeam']['score'],
                'fta': home_stats.get('freeThrowsAttempted', 0),
                'oreb': home_stats.get('reboundsOffensive', 0),
                'stl': home_stats.get('steals', 0),
                'blk': home_stats.get('blocks', 0)
            })
            self.tracker.orch.feature_engine.away_stats.update({
                'fgm': away_stats.get('fieldGoalsMade', 0),
                'fga': away_stats.get('fieldGoalsAttempted', 1),
                'fg3m': away_stats.get('threePointersMade', 0),
                'to': away_stats.get('turnovers', 0),
                'reb': away_stats.get('reboundsTotal', 0),
                'pts': live_data['awayTeam']['score'],
                'fta': away_stats.get('freeThrowsAttempted', 0),
                'oreb': away_stats.get('reboundsOffensive', 0),
                'stl': away_stats.get('steals', 0),
                'blk': away_stats.get('blocks', 0)
            })
        
        # Extract game info
        home_score = live_data['homeTeam']['score']
        away_score = live_data['awayTeam']['score']
        score_diff = home_score - away_score
        
        period = live_data.get('period', 4)
        game_clock = live_data.get('gameClock', 'PT0M00.00S')
        total_seconds = self._parse_time(period, game_clock)
        
        home_tri = game['homeTeam']['teamTricode']
        away_tri = game['awayTeam']['teamTricode']
        
        print(f"\n{away_tri} {away_score} @ {home_tri} {home_score} | {int(total_seconds//60)}:{int(total_seconds%60):02d} left")
        
        # Optimized: Batch refresh market prices for this game
        # Extract event prefix from the first market
        if spread_markets:
            try:
                parts = spread_markets[0].ticker.rsplit('-', 1)
                event_ticker = parts[0] if len(parts) >= 2 else spread_markets[0].ticker
                fresh_markets = self.kalshi.get_event_markets(event_ticker)
                if fresh_markets:
                    # Create lookup
                    price_lookup = {m['ticker']: m for m in fresh_markets}
                    for m in spread_markets:
                        if m.ticker in price_lookup:
                            item = price_lookup[m.ticker]
                            m.yes_bid = item.get('yes_bid') if item.get('yes_bid') is not None else int(float(item.get('yes_bid_dollars') or 0) * 100)
                            m.yes_ask = item.get('yes_ask') if item.get('yes_ask') is not None else int(float(item.get('yes_ask_dollars') or 0) * 100)
            except Exception as e:
                print(f"  ⚠️ Failed to refresh market prices: {e}")
        
        # Track when we first saw this game for warm-up
        if game_id not in self.game_tracking_start:
            self.game_tracking_start[game_id] = time.time()
        
        # Check warm-up status
        trader_elapsed = time.time() - self.game_tracking_start[game_id]
        
        # Game elapsed: In Q1, we check if 2 mins (120s) have passed. In later periods, it's definitely > 2 mins.
        if period == 1:
            game_elapsed = 2880 - total_seconds
        else:
            game_elapsed = 9999 # Already past Q1
            
        warmed_up = (trader_elapsed >= 120) and (game_elapsed >= 120)
        
        if not warmed_up:
            wait_reason = ""
            if game_elapsed < 60:
                wait_reason = f"game clock {int(game_elapsed)}s/60s"
            if trader_elapsed < 60:
                trader_reason = f"trader buffer {int(trader_elapsed)}s/60s"
                wait_reason = f"{wait_reason} and {trader_reason}" if wait_reason else trader_reason
            print(f"  ⏳ WARM-UP: Waiting for {wait_reason}")
        
        # Skip Q1 entirely (>36 min remaining). Model overconfident early, edge signals unreliable.
        if total_seconds > 2160:
            print(f"  ⏳ Skipping: Q1 ({total_seconds/60:.1f} min left, waiting for Q2)")
            return []

        # Skip late game
        if total_seconds < 120:
            print(f"  ⏰ Skipping: <2 min left")
            return []
        
        # Build features
        live_features = self._build_features(game, score_diff, total_seconds, period)
        enriched = add_interaction_features(live_features)
        
        # Debug: show categorized features being sent to model
        print(f"  === LIVE FEATURE CONTEXT ===")
        print(f"  [Game State]  pace={live_features.get('live_pace', 0):.1f}, swaps={live_features.get('lead_changes', 0)}, momentum_2m={live_features.get('momentum_2min', 0.0):+.1f}")
        print(f"  [Efficiency]  home_efg={live_features.get('home_efg', 0):.1%}, away_efg={live_features.get('away_efg', 0):.1%}, catchup={live_features.get('required_catchup_rate', 0):.3f}")
        print(f"  [Team Form]   home_SOS={live_features.get('home_team_recent_SOS', 0):.3f}, home_std={live_features.get('home_team_recent_scoring_consistency', 0):.1f}")
        print(f"                away_SOS={live_features.get('away_team_recent_SOS', 0):.3f}, away_std={live_features.get('away_team_recent_scoring_consistency', 0):.1f}")
        
        # PRIOR DECAY VISUALIZATION
        print(f"  [Prior Decay] home_season_margin: {enriched.get('home_season_margin_x_time', 0):+.2f} (from {live_features.get('home_team_season_win_margin', 0):+.1f})")
        print(f"                away_season_margin: {enriched.get('away_season_margin_x_time', 0):+.2f} (from {live_features.get('away_team_season_win_margin', 0):+.1f})")
        
        print(f"  [Variance]    time_x_margin={enriched.get('time_x_margin', 0):.1f}, log_time={enriched.get('log_time', 0):.2f}, volatility={enriched.get('score_volatility', 0):.2f}")
        
        # Get model prediction (NGBoost learns uncertainty directly, no manual multiplier needed)
        params = self.spread_model.predict_distribution_params(live_features)
        mean_diff = float(np.mean(params['mean']))
        std_diff = float(np.mean(params['std']))
        
        # Store for dashboard export
        if not hasattr(self, '_dashboard_cache'):
            self._dashboard_cache = {}
        
        # Get distribution components for dashboard
        X_live = pd.DataFrame([{col: enriched.get(col, 0.0) for col in self.spread_model.feature_order}], 
                              columns=self.spread_model.feature_order)
        locs, scales, dfs = [], [], []
        for model in self.spread_model.ensemble:
            dist = model.pred_dist(X_live.values)
            locs.append(dist.loc[0])
            scales.append(dist.scale[0])
            dfs.append(dist.df[0] if hasattr(dist, 'df') else 30.0)
            
        self._dashboard_cache[game_id] = {
            "score_diff": score_diff,
            "total_seconds": total_seconds,
            "period": period,
            "live_features": live_features,
            "enriched": enriched,
            "distribution": {
                "loc": np.mean(locs) + score_diff,
                "scale": np.mean(scales),
                "df": np.mean(dfs)
            }
        }
        
        print(f"  Model: {mean_diff:+.1f} ± {std_diff:.1f}")
        
        # Evaluate each market
        opportunities = []
        
        print(f"\n  {'Market':<12} {'Bid-Ask':<12} {'Model':<20} {'Edge':<12}")
        print(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*12}")
        
        for market in spread_markets:
            # Skip markets with tight bid-ask spread (data shows these lose money)
            bid = market.yes_bid or 0
            ask = market.yes_ask or 0
            market_spread = ask - bid if (bid > 0 and ask > 0) else 0
            if market_spread < self.min_spread:
                market_name = market.ticker.split('-')[-1] if '-' in market.ticker else market.ticker[-10:]
                print(f"  {market_name:<12} {'SKIP':>12} spread {market_spread}¢ < {self.min_spread}¢ min")
                continue
            
            is_home = (market.team == home_tri)
            threshold = market.spread
            
            # Get probabilities from model
            result = self.spread_model.predict_spread_probabilities(
                live_features, 
                [threshold if is_home else -threshold]
            )
            
            if is_home:
                model_prob = result['probabilities'][0]
                ci_lower = result['ci_90_lower'][0]
                ci_upper = result['ci_90_upper'][0]
            else:
                model_prob = 1 - result['probabilities'][0]
                ci_lower = 1 - result['ci_90_upper'][0]
                ci_upper = 1 - result['ci_90_lower'][0]
            
            # Store for order management and dashboard
            self.latest_market_data[market.ticker] = {
                'bid': market.yes_bid,
                'ask': market.yes_ask,
                'model_prob': model_prob,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            # Log prediction for every market evaluated (regardless of trade)
            self.trade_logger.log_prediction(
                game_id=game_id,
                ticker=market.ticker,
                seconds_remaining=int(total_seconds),
                score_diff=score_diff,
                predicted_prob=model_prob,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                bid_price=market.yes_bid,
                ask_price=market.yes_ask,
                features=live_features
            )
            
            # Calculate edge using beat-by-1 prices (same as trading logic)
            # BUY at bid+1, edge = ci_lower - buy_price
            # SELL at ask-1, edge = sell_price - ci_upper
            buy_price = (market.yes_bid + 1) if market.yes_bid else 0
            sell_price = (market.yes_ask - 1) if market.yes_ask else 0
            buy_edge = (ci_lower - buy_price / 100) if buy_price else 0
            sell_edge = (sell_price / 100 - ci_upper) if sell_price else 0
            best_edge = max(buy_edge, sell_edge)
            
            # Market short name
            market_name = market.ticker.split('-')[-1] if '-' in market.ticker else market.ticker[-10:]
            
            # Format strings
            bid_ask_str = f"{market.yes_bid or 0}-{market.yes_ask or 0}¢"
            
            if sell_edge > buy_edge and sell_edge >= 0:
                action_str = "SELL YES"
                # For clarity when selling, show the EV+ side (which is NO)
                no_prob = 1 - model_prob
                no_ci_lower = 1 - ci_upper
                no_ci_upper = 1 - ci_lower
                model_str = f"NO {no_prob*100:.0f}¢ ({no_ci_lower*100:.0f}-{no_ci_upper*100:.0f})"
            else:
                action_str = "BUY  YES"
                model_str = f"YES {model_prob*100:.0f}¢ ({ci_lower*100:.0f}-{ci_upper*100:.0f})"
            
            # Store model fair value for position EV display
            if not hasattr(self, '_model_fair_values'):
                self._model_fair_values = {}
            self._model_fair_values[market.ticker] = model_prob * 100
            
            # Edge indicator
            if best_edge >= self.trader.min_edge:
                edge_str = f"✅ {action_str} {best_edge:.1%}"
            elif best_edge >= 0.02:
                edge_str = f"🟡 {action_str} {best_edge:.1%}"
            else:
                edge_str = f"❌ {action_str} {best_edge:.1%}"
            
            print(f"  {market_name:<12} {bid_ask_str:<12} {model_str:<20} {edge_str}")
            
            # Dynamic edge threshold: Q2/Q3 requires more edge than Q4
            # Q4 = ≤720s (12 min), Q2/Q3 = 720-2160s
            effective_min_edge = self.trader.min_edge if total_seconds <= 720 else 0.12
            self.trader.min_edge = effective_min_edge

            # Get edge-based opportunities from trader (with conservative CI-based edge)
            opps = self.trader.evaluate_market(
                ticker=market.ticker,
                model_prob=model_prob,
                bid=market.yes_bid,
                ask=market.yes_ask,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )

            # Restore base min_edge for next market
            self.trader.min_edge = self.min_edge
            
            for opp in opps:
                # Add game context
                opp.game_id = game_id
                opp.seconds_remaining = total_seconds
                opp.warmed_up = warmed_up
                opportunities.append(opp)
        
        if opportunities:
            print(f"\n  📊 {len(opportunities)} opportunities found")
        
        return opportunities
    
    def _manage_orders(self, opportunities: list):
        """Cancel stale orders."""
        open_orders = self.order_mgr.get_open_orders()
        if not open_orders:
            return
        
        # Build lookup for current opportunities
        opp_lookup = {(o.ticker, o.action): o for o in opportunities}
        
        for order in open_orders:
            key = (order.ticker, order.side)
            current_opp = opp_lookup.get(key)
            
            # Get latest data for this ticker
            md = self.latest_market_data.get(order.ticker)
            if not md:
                # No data for this market this iteration (game skipped or finished) - cancel
                print(f"  ❌ Cancel {order.order_id}: No current market data")
                self._cancel_order_with_logging(order.order_id)
                continue
                
            # Use the trader's logic to check if we should cancel
            should_cancel, reason = self.trader.should_cancel_order(
                order_side='yes', # Spread markets are always 'yes' side
                order_action=order.side, # 'buy' or 'sell'
                order_price=int(order.price),
                model_prob=md['model_prob'],
                current_bid=md['bid'],
                current_ask=md['ask']
            )
            
            if should_cancel:
                print(f"  ❌ Cancel {order.order_id}: {reason}")
                self._cancel_order_with_logging(order.order_id)
    
    def _cancel_order_with_logging(self, order_id: str):
        """Cancel order and log cancellation to database."""
        # Cancel via order manager
        self.order_mgr.cancel_order(order_id)
        
        # Look up trade_id from database and log cancellation
        trade_id = self.trade_logger.get_trade_id_by_order_id(order_id)
        if trade_id:
            self.trade_logger.log_order_canceled(trade_id)
    
    def _place_orders(self, opportunities: list):
        """Place new orders for opportunities."""
        if not opportunities:
            return
        
        # Filter for warmed up opportunities
        ready_opps = [o for o in opportunities if getattr(o, 'warmed_up', False)]
        if not ready_opps and opportunities:
            # Only print if we actually have potential opps but they are all warming up
            print("  (All opportunities suppressed during warm-up)")
            return
        
        open_orders = self.order_mgr.get_open_orders()
        existing = {(o.ticker, o.side) for o in open_orders}
        
        # Group by event ticker prefix for exposure management
        # Ticker format: KXNBASPREAD-25DEC10PHXOKC-OKC15
        # Event prefix: KXNBASPREAD-25DEC10PHXOKC
        by_event = {}
        for opp in opportunities:
            # Extract event prefix (everything before the last dash + market identifier)
            parts = opp.ticker.rsplit('-', 1)
            event_prefix = parts[0] if len(parts) >= 2 else opp.ticker
            if event_prefix not in by_event:
                by_event[event_prefix] = []
            by_event[event_prefix].append(opp)
        
        for event_prefix, event_opps in by_event.items():
            # Calculate current game exposure INCLUDING pending orders
            game_exposure = 0.0
            
            # Add exposure from ALL filled positions for this event
            # Match positions by event prefix
            for ticker, pos in self.portfolio.positions.items():
                if ticker.startswith(event_prefix) and pos != 0:
                    cost_basis = self.portfolio.cost_basis.get(ticker, 50.0)
                    if pos > 0:
                        # Long position: exposure = cost
                        game_exposure += (cost_basis / 100.0) * pos
                    else:
                        # Short position: exposure = (100 - cost)
                        game_exposure += ((100 - cost_basis) / 100.0) * abs(pos)
            
            # Add exposure from PENDING orders in this event
            for order in open_orders:
                if order.ticker.startswith(event_prefix):
                    order_pos = self.portfolio.positions.get(order.ticker, 0)
                    game_exposure += self.risk_mgr._calculate_order_exposure(
                        order.side, order.price, order.size, order_pos
                    )
            
            # Extract short event name for display
            event_short = event_prefix.split('-')[-1] if '-' in event_prefix else event_prefix
            print(f"\n  Event {event_short}: Current exposure ${game_exposure:.2f} / ${self.max_game_exposure:.2f}")
            
            # Sort by Edge (percentage) to prioritize best deals
            event_opps.sort(key=lambda x: x.edge, reverse=True)
            
            for opp in event_opps:
                market_name = opp.ticker[-10:]
                
                # Skip if already have order for this ticker+side
                if (opp.ticker, opp.action) in existing:
                    print(f"    {market_name} {opp.action.upper()}: Already have order, skipping")
                    continue
                
                # Calculate current exposure for THIS TICKER specifically
                ticker_exposure = 0.0
                pos = self.portfolio.positions.get(opp.ticker, 0)
                if pos != 0:
                    cost = self.portfolio.cost_basis.get(opp.ticker, 50.0)
                    if pos > 0:
                        ticker_exposure = (cost / 100.0) * pos
                    else:
                        ticker_exposure = ((100 - cost) / 100.0) * abs(pos)
                
                # Add PENDING orders for this ticker
                for order in open_orders:
                    if order.ticker == opp.ticker:
                        ticker_exposure += self.risk_mgr._calculate_order_exposure(
                            order.side, order.price, order.size, pos
                        )
                
                # Calculate potential exposure change from this order
                current_pos = self.portfolio.positions.get(opp.ticker, 0)
                order_exposure = self.risk_mgr._calculate_order_exposure(
                    opp.action, opp.price, 1, current_pos  # Calculate for 1 contract first
                )
                
                # Check if this is position-reducing (negative exposure = good!)
                is_reducing = (
                    (opp.action == 'buy' and current_pos < 0) or
                    (opp.action == 'sell' and current_pos > 0)
                )
                
                if is_reducing:
                    # Position-reducing orders are always allowed - close full position
                    size = min(20, abs(current_pos))
                    print(f"    {market_name} {opp.action.upper()}: CLOSING pos={current_pos}, size={size} ✅")
                else:
                    # Position-increasing: check exposure limit first
                    if game_exposure >= self.max_game_exposure:
                        print(f"    {market_name} {opp.action.upper()}: BLOCKED - game exposure ${game_exposure:.2f} >= ${self.max_game_exposure:.2f}")
                        continue
                    
                    if ticker_exposure >= self.max_ticker_exposure:
                        print(f"    {market_name} {opp.action.upper()}: BLOCKED - ticker exposure ${ticker_exposure:.2f} >= ${self.max_ticker_exposure:.2f}")
                        continue
                    
                    # Use Kelly criterion for sizing with conservative CI bounds
                    # Combined budget: whichever is smaller
                    remaining_game = self.max_game_exposure - game_exposure
                    remaining_ticker = self.max_ticker_exposure - ticker_exposure
                    bankroll = min(remaining_game, remaining_ticker)

                    fair_value = opp.model_prob * 100

                    # Dynamic Kelly fraction by game phase — model is less reliable early
                    # Q2 (24-36 min): 1/8 Kelly  Q3 (12-24 min): 1/6 Kelly  Q4 (<12 min): 1/4 Kelly
                    secs = getattr(opp, 'seconds_remaining', total_seconds)
                    if secs > 1440:
                        phase_kelly = 0.125       # Q2: eighth-Kelly
                    elif secs > 720:
                        phase_kelly = 1 / 6       # Q3: sixth-Kelly
                    else:
                        phase_kelly = 0.25        # Q4: quarter-Kelly
                    self.position_sizer.kelly_fraction = phase_kelly

                    size = self.position_sizer.calculate_size(
                        fair_value=fair_value,
                        price=opp.price,
                        bankroll=remaining_game, # Use full game budget for Kelly math
                        action=opp.action,
                        ci_lower=opp.ci_lower,
                        ci_upper=opp.ci_upper,
                        position=current_pos,
                        min_size=1,
                        max_size=20
                    )
                    
                    # Restore default Kelly fraction
                    self.position_sizer.kelly_fraction = 0.25

                    # Clamp to individual ticker limit
                    cost_per = opp.price / 100.0
                    ticker_cap_contracts = int(remaining_ticker / cost_per)
                    if size > ticker_cap_contracts:
                        size = ticker_cap_contracts
                
                if size < 1:
                    if opp.edge >= self.trader.min_edge:
                        print(f"    {market_name} {opp.action.upper()}: SKIPPED (size=0) - Edge {opp.edge:.1%} check Kelly vs bankroll ${remaining_game:.2f}")
                    continue
                
                # Calculate actual exposure for this size
                actual_exposure = self.risk_mgr._calculate_order_exposure(
                    opp.action, opp.price, size, current_pos
                )
                
                # Place order
                marker = "[CLOSING]" if is_reducing else "[OPENING]"
                print(f"  🎯 {marker} {opp.ticker[-10:]} {opp.action.upper()} {size} @ {opp.price}¢ (edge: {opp.edge:.1%}, exp: {'$' + f'{actual_exposure:.2f}' if actual_exposure >= 0 else '-$' + f'{-actual_exposure:.2f}'})")
                
                order_id = self.order_mgr.place_limit_order(
                    ticker=opp.ticker,
                    side=opp.action,
                    price=opp.price,
                    size=size
                )
                
                if order_id:
                    # Log to database
                    game_id_for_log = None
                    if '-' in opp.ticker:
                        parts = opp.ticker.split('-')
                        if len(parts) >= 2:
                            game_id_for_log = parts[1][:10]
                    
                    self.trade_logger.log_order_placed(
                        ticker=opp.ticker,
                        side=opp.action,
                        price=opp.price,
                        size=size,
                        game_id=game_id_for_log,
                        model_fair=opp.model_prob * 100,
                        ci_lower=opp.ci_lower * 100 if opp.ci_lower else None,
                        ci_upper=opp.ci_upper * 100 if opp.ci_upper else None,
                        market_spread=opp.market_spread,
                        position_before=current_pos,
                        seconds_remaining=getattr(opp, 'seconds_remaining', None),
                        kalshi_order_id=order_id,
                        strategy_id='simple_ev'
                    )
                    
                    # Update exposure tracking for next iteration
                    game_exposure += actual_exposure
                    existing.add((opp.ticker, opp.action))
    
    def _get_game_exposure(self, game_id: str) -> float:
        """Get current exposure for a game."""
        exposure = 0.0
        for ticker, pos in self.portfolio.positions.items():
            if game_id in ticker and pos != 0:
                cost = self.portfolio.cost_basis.get(ticker, 50.0)
                exposure += abs(pos) * cost / 100
        return exposure
    
    def _export_dashboard_state(self):
        """Export current state for dashboard visualization."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": {
                "cash": self.portfolio.cash,
                "realized_pnl": self.portfolio.realized_pnl,
                "exposure": self.portfolio.get_exposure(),
                "positions": self.portfolio.positions,
                "cost_basis": self.portfolio.cost_basis
            },
            "open_orders": [
                {
                    "ticker": o.ticker,
                    "side": o.side,
                    "price": o.price,
                    "size": o.size
                } for o in self.order_mgr.get_open_orders()
            ],
            "games": []
        }
        
        for match in self.tracker.active_matches:
            game = match['nba_game']
            game_id = game['gameId']
            
            # Use cached prediction data from _evaluate_game
            cache = getattr(self, '_dashboard_cache', {}).get(game_id)
            if not cache:
                continue
                
            live_features = cache["live_features"]
            enriched = cache["enriched"]
            
            driving_features = {
                "live_state": {
                    "Pace": f"{live_features.get('live_pace', 0):.1f}",
                    "Momentum": f"{live_features.get('score_momentum', 0):+.1f}",
                    "Home eFG%": f"{live_features.get('home_efg', 0):.1%}",
                    "Away eFG%": f"{live_features.get('away_efg', 0):.1%}",
                    "Catchup Rate": f"{live_features.get('required_catchup_rate', 0):.4f}",
                    "Steals": f"H:{self.tracker.orch.feature_engine.home_stats.get('stl',0)} A:{self.tracker.orch.feature_engine.away_stats.get('stl',0)}",
                    "Blocks": f"H:{self.tracker.orch.feature_engine.home_stats.get('blk',0)} A:{self.tracker.orch.feature_engine.away_stats.get('blk',0)}"
                },
                "team_context": {
                    "Home Margin": f"{live_features.get('home_team_recent_win_margin', 0):+.1f}",
                    "Home SOS": f"{live_features.get('home_team_recent_SOS', 0):.3f}",
                    "Away Margin": f"{live_features.get('away_team_recent_win_margin', 0):+.1f}",
                    "Away SOS": f"{live_features.get('away_team_recent_SOS', 0):.3f}",
                },
                "variance_drivers": {
                    "Lead Changes": live_features.get('lead_changes', 0),
                    "Volatility": f"{live_features.get('score_volatility', 0):.2f}",
                    "Time x Margin": f"{enriched.get('time_x_margin', 0):.0f}",
                }
            }
            
            game_state = {
                "game_id": game_id,
                "home_team": game['homeTeam']['teamTricode'],
                "away_team": game['awayTeam']['teamTricode'],
                "home_score": cache["score_diff"] + (cache["total_seconds"] * 0), # Placeholder for home_score extraction if needed elsewhere
                "away_score": 0, # Placeholder
                "seconds_remaining": cache["total_seconds"],
                "period": cache["period"],
                "distribution": cache["distribution"],
                "driving_features": driving_features,
                "markets": []
            }
            
            # Correct scores from live data if possible
            live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
            if live_data:
                game_state["home_score"] = live_data['homeTeam']['score']
                game_state["away_score"] = live_data['awayTeam']['score']
            
            for market in match['spread_markets']:
                market_orders = [o for o in self.order_mgr.get_open_orders() if o.ticker == market.ticker]
                curr_p = self.portfolio.positions.get(market.ticker, 0)
                
                # Calculate exposures
                # Note: SimpleLiveTrader risk_mgr._calculate_order_exposure(side, price, size, current_pos)
                pending_buy_exp = sum(self.risk_mgr._calculate_order_exposure('buy', o.price, o.size, curr_p) for o in market_orders if o.side == 'buy')
                pending_sell_exp = sum(self.risk_mgr._calculate_order_exposure('sell', o.price, o.size, curr_p) for o in market_orders if o.side == 'sell')
                
                pos_exp = 0.0
                if curr_p != 0:
                    cost = self.portfolio.cost_basis.get(market.ticker, 50.0)
                    pos_exp = self.risk_mgr._calculate_order_exposure('buy' if curr_p > 0 else 'sell', cost, abs(curr_p), 0)

                game_state["markets"].append({
                    "ticker": market.ticker,
                    "spread": market.spread,
                    "team": market.team,
                    "bid": market.yes_bid,
                    "ask": market.yes_ask,
                    "fair_value": getattr(self, '_model_fair_values', {}).get(market.ticker),
                    "position": curr_p,
                    "pending_buy": sum(o.size for o in market_orders if o.side == 'buy'),
                    "pending_buy_exp": pending_buy_exp,
                    "pending_sell": sum(o.size for o in market_orders if o.side == 'sell'),
                    "pending_sell_exp": pending_sell_exp,
                    "position_exp": pos_exp
                })
            
            state["games"].append(game_state)
            
        with open('data/dashboard_state.json', 'w') as f:
            json.dump(state, f)
    
    def _build_features(self, game, score_diff, total_seconds, period) -> dict:
        """Build feature dict for model."""
        # Update history for momentum calculation
        self.tracker.orch.feature_engine.history.append((total_seconds, score_diff))
        if len(self.tracker.orch.feature_engine.history) > 1000:
             self.tracker.orch.feature_engine.history = [h for h in self.tracker.orch.feature_engine.history if h[0] < total_seconds + 600]

        # Use the stateful FeatureEngine to calculate all current features
        live_features = self.tracker.orch.feature_engine.calculate_current_features(
            score_diff,
            total_seconds,
            period,
            game['gameId'],
            game['homeTeam']['teamId'],
            game['awayTeam']['teamId']
        )
        
        # Combine with context (historical stats + roster stats)
        full_feats = {**live_features, **self.tracker.orch.prediction_engine.current_game_context}
        full_feats['is_home'] = 1 # Keep for legacy compatibility
        
        return full_feats
    
    def _parse_time(self, period: int, game_clock: str) -> float:
        """Parse game clock to total seconds remaining."""
        try:
            match = re.search(r'PT(\d+)M([\d.]+)S', game_clock)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                period_seconds = minutes * 60 + seconds
            else:
                period_seconds = 0
            
            if period < 4:
                return period_seconds + (4 - period) * 12 * 60
            else:
                return period_seconds
        except:
            return 0
    
    def _print_status(self):
        """Print current status."""
        print(f"\n=== STATUS ===")
        print(f"Cash: ${self.portfolio.cash:.2f}")
        
        # Calculate filled and pending exposure
        filled_exposure = self.portfolio.get_exposure()
        
        # Calculate pending order exposure
        pending_exposure = 0.0
        open_orders = self.order_mgr.get_open_orders()
        for order in open_orders:
            order_pos = self.portfolio.positions.get(order.ticker, 0)
            pending_exposure += self.risk_mgr._calculate_order_exposure(
                order.side, order.price, order.size, order_pos
            )
        
        total_exposure = filled_exposure + pending_exposure
        
        if pending_exposure > 0:
            print(f"Exposure: ${filled_exposure:.2f} (filled) + ${pending_exposure:.2f} (pending) = ${total_exposure:.2f}")
        else:
            print(f"Exposure: ${filled_exposure:.2f}")
        
        print(f"P&L: ${self.portfolio.realized_pnl:+.2f}")
        
        if self.portfolio.positions:
            total_ev = 0.0
            print("Positions:")
            print(f"  {'Ticker':<12} {'Pos':>5} {'Cost':>8} {'Model':>8} {'EV':>10}")
            print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*10}")
            
            for ticker, pos in self.portfolio.positions.items():
                if pos != 0:
                    cost = self.portfolio.cost_basis.get(ticker, 50.0)
                    
                    # Get model fair value for this position
                    # (stored during evaluation, fallback to cost if not available)
                    model_fair = getattr(self, '_model_fair_values', {}).get(ticker, cost)
                    
                    # Calculate EV
                    # For SHORT position (-): You sold YES at cost, you win if NO happens
                    # EV = (1-model_prob) * cost - model_prob * (100-cost)
                    # Simplified: EV = cost - model_fair
                    if pos < 0:
                        # Short: you win (cost) if NO, lose (100-cost) if YES
                        # EV per contract = (1-P)*cost - P*(100-cost) = cost - model_fair
                        ev_per_contract = (cost - model_fair) / 100
                        position_ev = ev_per_contract * abs(pos)
                    else:
                        # Long: you win (100-cost) if YES, lose (cost) if NO
                        # EV per contract = P*(100-cost) - (1-P)*cost = model_fair - cost
                        ev_per_contract = (model_fair - cost) / 100
                        position_ev = ev_per_contract * pos
                    
                    total_ev += position_ev
                    
                    market_name = ticker[-10:] if len(ticker) > 10 else ticker
                    ev_str = f"${position_ev:+.2f}" if position_ev >= 0 else f"-${-position_ev:.2f}"
                    print(f"  {market_name:<12} {pos:>+5} {cost:>7.1f}¢ {model_fair:>7.1f}¢ {ev_str:>10}")
            
            print(f"  {'-'*45}")
            total_str = f"${total_ev:+.2f}" if total_ev >= 0 else f"-${-total_ev:.2f}"
            print(f"  {'TOTAL EV':<28} {total_str:>16}")
        
        print(self.order_mgr.get_order_summary())


def main():
    """Run simple trader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple +EV Trader')
    parser.add_argument('--live', action='store_true', help='Run in live mode (real money)')
    parser.add_argument('--min-edge', type=float, default=0.08, help='Min edge to trade')
    parser.add_argument('--min-spread', type=int, default=4, help='Min bid-ask spread in cents (default 4)')
    parser.add_argument('--interval', type=int, default=15, help='Seconds between iterations')
    
    args = parser.parse_args()
    
    # API credentials (from environment or defaults for local dev)
    kalshi_key_id = os.environ.get("KALSHI_KEY_ID", "a40ff1c6-12ac-4a6c-9669-ffe12f3de235")
    kalshi_key_path = os.environ.get("KALSHI_KEY_PATH", "key.key")
    
    # Dynamic Balance and Risk settings
    kalshi_client = KalshiClient(kalshi_key_id, kalshi_key_path)
    try:
        current_balance = kalshi_client.get_balance().get('balance', 500)
        # 80% of balance, rounded down to nearest hundred
        bal = int((current_balance * 0.8) // 100) * 100
        # Ensure at least $100 bare minimum
        bal = max(bal, 100)
        print(f"Kalshi Balance: ${current_balance:,.2f} | Assigned Bankroll (80% rounded down): ${bal:,.2f}")
    except Exception as e:
        print(f"⚠️ Failed to fetch balance: {e}. Defaulting to $500 bankroll.")
        bal = 500
        
    risk_rate = 0.05
    MAX_GAME_EXPOSURE = bal * risk_rate
    MAX_TICKER_EXPOSURE = MAX_GAME_EXPOSURE / 3.0  # Max 1/3 of game exposure per ticker
    
    trader = SimpleLiveTrader(
        kalshi_key_id=kalshi_key_id,
        kalshi_key_path=kalshi_key_path,
        dry_run=not args.live,
        max_game_exposure=MAX_GAME_EXPOSURE,
        max_ticker_exposure=MAX_TICKER_EXPOSURE,
        min_edge=args.min_edge,
        min_spread=args.min_spread
    )
    
    trader.run(interval=args.interval)


if __name__ == "__main__":
    main()
