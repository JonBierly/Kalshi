#!/usr/bin/env python
"""
Dynamic Rebalancer - NBA trading bot for Kalshi spread markets.

Features:
1. Every 15-second iteration optimizes the portfolio.
2. Uses Maker-style "Beat-by-1" pricing for entries and exits.
3. Prioritizes best "ladder" rungs by Edge % within the $10 game budget.
4. Immediate Toxic Flow exits if fair value drops below cost.
5. Hysteresis buffer (2 contracts) to prevent churn.

Usage:
    python -m spread_src.scripts.rebalancing_live_trader --live
"""

import os
import sys
import time
import re
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.trading.rebalancing_logic import MultiAssetRebalancer, RebalancingAction
from spread_src.execution.portfolio import Portfolio
from spread_src.execution.order_manager import OrderManager
from spread_src.execution.trade_logger import TradeLogger
from spread_src.execution.risk_manager import RiskManager
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.inference.spread_tracker import SpreadTracker
from spread_src.features.engineering import add_interaction_features
from data.kalshi import KalshiClient


class RebalancingLiveTrader:
    """
    Dynamic Rebalancing trading engine.
    """
    
    def __init__(
        self,
        kalshi_key_id: str,
        kalshi_key_path: str = 'key.key',
        dry_run: bool = True,
        max_game_exposure: float = 10.0,
        max_ticker_exposure: float = 3.0,
        min_edge: float = 0.04,
    ):
        self.dry_run = dry_run
        self.max_game_exposure = max_game_exposure
        self.max_ticker_exposure = max_ticker_exposure
        self.min_edge = min_edge
        
        print("=" * 80)
        print("DYNAMIC REBALANCER")
        print("=" * 80)
        print(f"Mode: {'DRY-RUN (simulation)' if dry_run else 'LIVE (real money!)'}")
        print(f"Max per game: ${max_game_exposure}")
        print(f"Max per ticker: ${max_ticker_exposure}")
        print(f"Min edge: {min_edge:.0%}")
        print("=" * 80)
        
        # Initialize components
        print("\n✓ Initializing...")
        
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        self.portfolio = Portfolio(max_exposure=max_game_exposure * 10)
        self.portfolio.refresh_state(self.kalshi)
        
        self.order_mgr = OrderManager(self.kalshi, dry_run=dry_run)
        self.rebalancer = MultiAssetRebalancer(
            bankroll=max_game_exposure,
            kelly_fraction=0.20, 
            max_ticker_exposure=max_ticker_exposure,
            scale_up_band=0.10,
            derisk_band=0.20,
            min_trade_spread=8
        )
        self.weight_history = {} # {game_id: {ticker: [w1, w2, w3]}}
        self.risk_mgr = RiskManager(max_game_exposure * 10, max_game_exposure)
        
        # Build event_tickers map
        self.event_tickers = {}
        self._sync_event_tickers()
        
        # Initialize tracker
        model_path = 'models/nba_spread_ngboost.pkl'
        self.tracker = SpreadTracker(kalshi_key_id, kalshi_key_path, model_path=model_path)
        self.spread_model = self.tracker.spread_model
        
        # Database logging
        self.trade_logger = TradeLogger('data/nba_data.db')
        
        # Warm-up tracking
        self.game_tracking_start = {}
        print("✓ Ready")
    
    def _sync_event_tickers(self):
        """Build event_tickers map from synced positions."""
        self.event_tickers = {}
        for ticker, pos in self.portfolio.positions.items():
            if pos != 0:
                parts = ticker.rsplit('-', 1)
                event_prefix = parts[0] if len(parts) >= 2 else ticker
                if event_prefix not in self.event_tickers:
                    self.event_tickers[event_prefix] = []
                if ticker not in self.event_tickers[event_prefix]:
                    self.event_tickers[event_prefix].append(ticker)

    def run(self, interval: int = 10):
        """Main trading loop."""
        print("\nMatching games to spread markets...")
        self.tracker.setup()
        
        if not self.tracker.active_matches:
            print("⚠️ No games matched to spread markets")
            return
        
        print(f"✓ Tracking {len(self.tracker.active_matches)} game(s)\n")
        print("Starting rebalancing loop... Press Ctrl+C to stop\n")
        
        iteration = 0
        try:
            while True:
                iteration += 1
                curr_time = datetime.now().strftime('%H:%M:%S')
                print(f"\n{'=' * 80}")
                print(f"Iteration {iteration} @ {curr_time}")
                print(f"{'=' * 80}")
                
                # 1. Refresh State
                self.portfolio.refresh_state(self.kalshi)
                self._sync_event_tickers()
                
                # 2. Check Fills
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
                
                # 3. Evaluate each game
                all_actions_by_game = {} # {game_id: [actions]}
                
                for match in self.tracker.active_matches:
                    game = match['nba_game']
                    spread_markets = match['spread_markets']
                    game_id = game['gameId']
                    
                    actions = self._evaluate_game_rebalancing(game, spread_markets)
                    if actions:
                        all_actions_by_game[game_id] = actions
                
                # 4. Global Management and Execution
                self._execute_rebalancing(all_actions_by_game)
                
                # 4b. Emergency Brake
                for match in self.tracker.active_matches:
                    event_prefix = self._get_event_prefix(match['spread_markets'][0].ticker)
                    game_exp = self._get_game_exposure(event_prefix)
                    if game_exp > 20.0:  # Hard limit 2x max
                        print(f"\n🛑 EMERGENCY BRAKE: Game {event_prefix} exposure ${game_exp:.2f} exceeds safety limit!")
                        self.order_mgr.cancel_all_orders()
                        sys.exit(1)
                
                # 5. Export Dashboard State
                self._export_dashboard_state()
                
                # 6. Periodic Management
                if iteration % 5 == 0:
                    print("\n🔍 Periodically checking for settled markets...")
                    self.portfolio.settle_unsettled_trades(self.kalshi, self.trade_logger)
                
                # 7. Print status
                self._print_status()
                
                print(f"\nWaiting {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping trader...")
            open_orders = self.order_mgr.get_open_orders()
            for order in open_orders:
                self._cancel_order_with_logging(order.order_id)
            print("\n🔍 Checking for settled markets...")
            self.portfolio.settle_unsettled_trades(self.kalshi, self.trade_logger)
            self._print_status()
            print("✓ Shutdown complete")

    def _evaluate_game_rebalancing(self, game, spread_markets) -> List[RebalancingAction]:
        """Evaluate a game and find rebalancing actions."""
        game_id = game['gameId']
        
        # Setup context
        if game_id not in self.tracker.feature_engines:
            from spread_src.features.engineering import FeatureEngine
            self.tracker.feature_engines[game_id] = FeatureEngine()
            
        if self.tracker.orch.prediction_engine.current_game_id != game_id:
            self.tracker.orch.setup_game_context(
                game_id,
                game['homeTeam']['teamId'],
                game['awayTeam']['teamId']
            )
        
        self.tracker.orch.feature_engine = self.tracker.feature_engines[game_id]
        
        # Get live data
        live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
        if not live_data:
            return []
        
        period = live_data.get('period', 0)
        game_status = live_data.get('gameStatus', 1)
        
        if game_status == 1 or (period == 0 and live_data['homeTeam']['score'] == 0 and live_data['awayTeam']['score'] == 0):
            return []
        
        # Update features
        home_score = live_data['homeTeam']['score']
        away_score = live_data['awayTeam']['score']
        score_diff = home_score - away_score
        game_clock = live_data.get('gameClock', 'PT0M00.00S')
        total_seconds = self._parse_time(period, game_clock)
        
        if total_seconds < 180:
            print(f"  ⏭️  Skipping {game_id[-10:]}: Less than 3 minutes remaining")
            return []
            
        live_features = self._build_features(game, score_diff, total_seconds, period)
        
        # Track warm-up
        if game_id not in self.game_tracking_start:
            self.game_tracking_start[game_id] = time.time()
        trader_elapsed = time.time() - self.game_tracking_start[game_id]
        warmed_up = (trader_elapsed >= 60)

        # Get prediction
        params = self.spread_model.predict_distribution_params(live_features)
        mean_diff = np.mean(params['mean'])
        std_diff = np.mean(params['std'])
        
        home_tri = game['homeTeam']['teamTricode']
        away_tri = game['awayTeam']['teamTricode']
        
        status_tag = " [TRADING]" if warmed_up else f" [WARMING UP {int(60 - trader_elapsed)}s]"
        print(f"\n{away_tri} {away_score} @ {home_tri} {home_score} | {int(total_seconds//60)}:{int(total_seconds%60):02d} left | Model: {mean_diff:+.1f} ± {std_diff:.1f}{status_tag}")

        tickers = []
        probs = []
        prices = []
        market_map = {} # {ticker: market}
        ci_lower_map = {} # {ticker: ci_lower}
        ci_upper_map = {} # {ticker: ci_upper}
        prob_map = {} # {ticker: model_prob}
        
        for market in spread_markets:
            try:
                fresh = self.kalshi.get_market_details(market.ticker)
                if fresh:
                    market.yes_bid = fresh.get('yes_bid', market.yes_bid)
                    market.yes_ask = fresh.get('yes_ask', market.yes_ask)
            except:
                pass
            
            is_home = (market.team == home_tri)
            threshold = market.spread
            result = self.spread_model.predict_spread_probabilities(live_features, [threshold if is_home else -threshold])
            
            if is_home:
                model_prob, ci_lower, ci_upper = result['probabilities'][0], result['ci_90_lower'][0], result['ci_90_upper'][0]
            else:
                model_prob, ci_lower, ci_upper = 1 - result['probabilities'][0], 1 - result['ci_90_upper'][0], 1 - result['ci_90_lower'][0]
            
            self.trade_logger.log_prediction(
                game_id=game_id, ticker=market.ticker, seconds_remaining=int(total_seconds),
                score_diff=score_diff, predicted_prob=model_prob, ci_lower=ci_lower, ci_upper=ci_upper,
                bid_price=market.yes_bid, ask_price=market.yes_ask, features=live_features
            )
            
            ci_lower_map[market.ticker] = ci_lower
            ci_upper_map[market.ticker] = ci_upper
            prob_map[market.ticker] = model_prob
            
            # Store fair value for display
            if not hasattr(self, '_model_fair_values'): self._model_fair_values = {}
            self._model_fair_values[market.ticker] = model_prob * 100
            
            tickers.append(market.ticker)
            probs.append(model_prob)
            mid_price = (market.yes_bid + market.yes_ask) / 2.0 if (market.yes_bid and market.yes_ask) else 50.0
            prices.append(mid_price / 100.0)
            market_map[market.ticker] = market

        # 1. Calculate Optimal Weights
        # We'll use the Student-T from the first ensemble member for covariance
        X_live = pd.DataFrame([{col: live_features.get(col, 0.0) for col in self.spread_model.feature_order}], 
                              columns=self.spread_model.feature_order)
        dist = self.spread_model.ensemble[0].pred_dist(X_live.values)
        
        # Calculate individual edges for logic (Scale Up check) and display
        # Use mid-price for the "portfolio" solve, but individual_edges will be used for display
        individual_edges = {}
        for i, t in enumerate(tickers):
            individual_edges[t] = probs[i] - prices[i]

        raw_optimal_weights = self.rebalancer.calculate_optimal_weights(
            tickers=tickers,
            probs=probs,
            prices=prices,
            distribution=dist
        )
        
        # 2. Smooth Weights (60s game-time moving average)
        if game_id not in self.weight_history:
            self.weight_history[game_id] = {t: [] for t in tickers}
            
        smoothed_weights = {}
        for ticker in tickers:
            hist = self.weight_history[game_id].get(ticker, [])
            hist.append((total_seconds, raw_optimal_weights.get(ticker, 0.0)))
            
            # Prune entries older than 60 game-seconds
            # total_seconds is "seconds remaining", so it decreases.
            # Keep only entries where h[0] <= total_seconds + 60
            hist = [h for h in hist if h[0] <= total_seconds + 60]
            
            self.weight_history[game_id][ticker] = hist
            smoothed_weights[ticker] = np.mean([h[1] for h in hist])

        # 3. Evaluate Rebalancing
        current_weights = {}
        bids = {}
        asks = {}
        pending_pos = {}
        
        open_orders = self.order_mgr.get_open_orders()
        
        for t in tickers:
            pos = self.portfolio.positions.get(t, 0)
            cost = self.portfolio.cost_basis.get(t, 50.0)
            # Weight = Exposure / Bankroll
            if pos >= 0:
                current_weights[t] = (pos * (cost / 100.0)) / self.max_game_exposure
            else:
                current_weights[t] = -(abs(pos) * ((100 - cost) / 100.0)) / self.max_game_exposure
                
            bids[t] = market_map[t].yes_bid
            asks[t] = market_map[t].yes_ask
            
            p_pos = 0
            for order in open_orders:
                if order.ticker == t:
                    p_pos += (order.size if order.side == 'buy' else -order.size)
            pending_pos[t] = p_pos

        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights=current_weights,
            optimal_weights=smoothed_weights,
            bids=bids,
            asks=asks,
            individual_edges=individual_edges,
            current_positions=self.portfolio.positions
        )
        
        # Add metadata and print summary table
        print(f"  {'Ticker':<12} | {'Pos':>4} | {'Bid/Ask':>9} | {'BuyE':>5} | {'SellE':>5} | {'SmoothW':>7} | {'Status'}")
        print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*10}")
        
        for t in tickers:
            pos = self.portfolio.positions.get(t, 0)
            curr_w = current_weights.get(t, 0.0)
            
            idx = tickers.index(t)
            m_prob = probs[idx]
            bid_p = bids.get(t, 0)
            ask_p = asks.get(t, 100)
            spread = ask_p - bid_p
            
            # Use execution prices [5, 95]
            buy_price = max(5, min(95, bid_p + 1))
            sell_price = max(5, min(95, ask_p - 1))
            
            buy_edge = m_prob - (buy_price / 100.0)
            sell_edge = (sell_price / 100.0) - m_prob
            
            raw_w = raw_optimal_weights.get(t, 0.0)
            smooth_w = smoothed_weights.get(t, 0.0)
            
            # Determine Status
            diff = smooth_w - curr_w
            status = "---"
            
            # Action Mapping (must match rebalancer logic)
            if any(a.ticker == t for a in actions):
                action = next(a for a in actions if a.ticker == t)
                status = f"✅ {action.reason.split()[-1]}"
            else:
                # Why was it skipped?
                if spread < self.rebalancer.min_trade_spread:
                    if diff > self.rebalancer.scale_up_band:
                        status = "⏳ PATIENT (S)" # Wait for spread
                    elif abs(pos) > 0 and diff < -self.rebalancer.derisk_band:
                        status = "🤝 HOLDING (S)" # Strong hand in tight market
                    else:
                        status = "🚫 TIGHT"
                elif diff > 0 and diff <= self.rebalancer.scale_up_band:
                    status = "💤 BAND (B)"
                elif diff < 0 and diff >= -self.rebalancer.derisk_band:
                    status = "💤 BAND (D)"

            bid_ask_str = f"{bid_p:02d}-{ask_p:02d}"
            print(f"  {t[-10:]: <12} | {pos: >4} | {bid_ask_str: >9} | {buy_edge: >5.1%} | {sell_edge: >5.1%} | {smooth_w: >7.1%} | {status}")

        for act in actions:
            act.warmed_up = warmed_up
            act.game_id = game_id
            act.model_prob = prob_map.get(act.ticker, 0.0)
            act.market_spread = (market_map[act.ticker].yes_ask - market_map[act.ticker].yes_bid)
            act.seconds_remaining = total_seconds
            act.ci_lower = ci_lower_map.get(act.ticker)
            act.ci_upper = ci_upper_map.get(act.ticker)
            
        return actions

    def _execute_rebalancing(self, all_actions_by_game: Dict[str, List[RebalancingAction]]):
        """
        Global execution logic: sort by edge and allocate game budget.
        """
        open_orders = self.order_mgr.get_open_orders()
        
        # 1. Cancel stale orders
        # If an order's ticker/action isn't in any current rebalancing action, cancel it.
        # This covers cases where edge vanished and optimal became current.
        all_action_keys = {} # {(ticker, action): action}
        for actions in all_actions_by_game.values():
            for a in actions:
                all_action_keys[(a.ticker, a.action)] = a
                
        for order in open_orders:
            act = all_action_keys.get((order.ticker, order.side))
            if not act:
                print(f"  ❌ Cancel {order.ticker[-10:]} {order.side}: No longer optimal")
                self._cancel_order_with_logging(order.order_id)
            elif abs(order.price - act.price) > 1:
                print(f"  ❌ Cancel {order.ticker[-10:]} {order.side}: Price sync ({order.price}¢ -> {act.price}¢)")
                self._cancel_order_with_logging(order.order_id)

        # 2. Process Games
        for game_id, actions in all_actions_by_game.items():
            # Sort by edge to prioritize "ladder" rungs
            actions.sort(key=lambda x: x.edge, reverse=True)
            
            # Calculate current game exposure
            # Prefix for markets in this game
            event_prefix = self._get_event_prefix(actions[0].ticker)
            
            # Filled exposure
            game_filled_exp = 0.0
            for t, p in self.portfolio.positions.items():
                if t.startswith(event_prefix) and p != 0:
                    cost = self.portfolio.cost_basis.get(t, 50.0)
                    game_filled_exp += self.risk_mgr.get_exposure_delta('buy' if p > 0 else 'sell', cost, abs(p), 0)
            
            # Pending exposure
            game_pending_exp = 0.0
            for order in self.order_mgr.get_open_orders():
                if order.ticker.startswith(event_prefix):
                    curr_p = self.portfolio.positions.get(order.ticker, 0)
                    game_pending_exp += self.risk_mgr.get_exposure_delta(order.side, order.price, order.size, curr_p)
            
            curr_game_exp = game_filled_exp + game_pending_exp
            print(f"  Game {event_prefix.split('-')[-1]}: Exp ${curr_game_exp:.2f} / ${self.max_game_exposure:.2f}")

            for act in actions:
                if not getattr(act, 'warmed_up', False):
                    continue
                
                # Already have order?
                if any(o.ticker == act.ticker and o.side == act.action for o in self.order_mgr.get_open_orders()):
                    continue
                    
                # Calculate ticker exposure
                curr_ticker_pos = self.portfolio.positions.get(act.ticker, 0)
                ticker_filled_exp = self._get_ticker_exposure(act.ticker)
                # Ticker pending
                ticker_pending_exp = 0.0
                for order in self.order_mgr.get_open_orders():
                    if order.ticker == act.ticker:
                        ticker_pending_exp += self.risk_mgr.get_exposure_delta(order.side, order.price, order.size, curr_ticker_pos)
                
                curr_ticker_exp = ticker_filled_exp + ticker_pending_exp
                
                # Potential delta
                delta = self.risk_mgr.get_exposure_delta(act.action, act.price, act.size, curr_ticker_pos)
                
                # Exposure Checks
                can_trade = True
                if not act.is_toxic_exit:
                    potential_game_exp = curr_game_exp + delta
                    potential_ticker_exp = curr_ticker_exp + delta
                    
                    if potential_game_exp > self.max_game_exposure and delta > 0:
                        print(f"    ⚠️ {act.ticker[-10:]}: Game exposure limit hit (${curr_game_exp:.2f} + ${delta:.2f} > ${self.max_game_exposure:.2f})")
                        can_trade = False
                    if potential_ticker_exp > self.max_ticker_exposure and delta > 0:
                        print(f"    ⚠️ {act.ticker[-10:]}: Ticker exposure limit hit (${curr_ticker_exp:.2f} + ${delta:.2f} > ${self.max_ticker_exposure:.2f})")
                        can_trade = False
                
                if can_trade and act.size > 0:
                    print(f"  🎯 {act.reason.upper()} {act.ticker[-10:]} {act.action.upper()} {act.size} @ {act.price}¢ (edge: {act.edge:.1%})")
                    order_id = self.order_mgr.place_limit_order(act.ticker, act.action, act.price, act.size)
                    if order_id:
                        self.trade_logger.log_order_placed(
                            ticker=act.ticker, side=act.action, price=act.price, size=act.size,
                            game_id=act.game_id[:10], model_fair=act.model_prob * 100,
                            ci_lower=act.ci_lower * 100, ci_upper=act.ci_upper * 100,
                            market_spread=act.market_spread, seconds_remaining=act.seconds_remaining,
                            position_before=self.portfolio.positions.get(act.ticker, 0),
                            kalshi_order_id=order_id
                        )
                        curr_game_exp += delta

    def _get_game_exposure(self, event_prefix: str) -> float:
        """Calculate total (filled + pending) exposure for a game."""
        filled_exp = 0.0
        for t, p in self.portfolio.positions.items():
            if t.startswith(event_prefix) and p != 0:
                cost = self.portfolio.cost_basis.get(t, 50.0)
                filled_exp += self.risk_mgr.get_exposure_delta('buy' if p > 0 else 'sell', cost, abs(p), 0)
        
        pending_exp = 0.0
        for order in self.order_mgr.get_open_orders():
            if order.ticker.startswith(event_prefix):
                curr_p = self.portfolio.positions.get(order.ticker, 0)
                pending_exp += self.risk_mgr.get_exposure_delta(order.side, order.price, order.size, curr_p)
                
        return filled_exp + pending_exp

    def _get_event_prefix(self, ticker: str) -> str:
        parts = ticker.rsplit('-', 1)
        return parts[0] if len(parts) >= 2 else ticker

    def _get_ticker_exposure(self, ticker: str) -> float:
        pos = self.portfolio.positions.get(ticker, 0)
        if pos == 0: return 0.0
        cost = self.portfolio.cost_basis.get(ticker, 50.0)
        if pos > 0: return (cost / 100.0) * pos
        else: return ((100 - cost) / 100.0) * abs(pos)

    def _cancel_order_with_logging(self, order_id: str):
        self.order_mgr.cancel_order(order_id)
        trade_id = self.trade_logger.get_trade_id_by_order_id(order_id)
        if trade_id: self.trade_logger.log_order_canceled(trade_id)

    def _build_features(self, game, score_diff, total_seconds, period) -> dict:
        self.tracker.orch.feature_engine.history.append((total_seconds, score_diff))
        if len(self.tracker.orch.feature_engine.history) > 1000:
             self.tracker.orch.feature_engine.history = [h for h in self.tracker.orch.feature_engine.history if h[0] < total_seconds + 600]
        live_features = self.tracker.orch.feature_engine.calculate_current_features(
            score_diff, total_seconds, period, game['gameId'], game['homeTeam']['teamId'], game['awayTeam']['teamId']
        )
        full_feats = {**live_features, **self.tracker.orch.prediction_engine.current_game_context}
        full_feats['is_home'] = 1
        return full_feats

    def _parse_time(self, period: int, game_clock: str) -> float:
        try:
            match = re.search(r'PT(\d+)M([\d.]+)S', game_clock)
            if match:
                minutes, seconds = int(match.group(1)), float(match.group(2))
                period_seconds = minutes * 60 + seconds
            else: period_seconds = 0
            return period_seconds + (4 - period) * 12 * 60 if period < 4 else period_seconds
        except: return 0

    def _print_status(self):
        print(f"\n=== STATUS ===")
        print(f"Cash: ${self.portfolio.cash:.2f}")
        filled_exp = self.portfolio.get_exposure()
        pending_exp = 0.0
        for order in self.order_mgr.get_open_orders():
            curr_p = self.portfolio.positions.get(order.ticker, 0)
            pending_exp += self.risk_mgr.get_exposure_delta(order.side, order.price, order.size, curr_p)
        print(f"Exposure: ${filled_exp:.2f} (filled) + ${pending_exp:.2f} (pending) = ${filled_exp+pending_exp:.2f}")
        
        if self.portfolio.positions:
            print(f"\nPositions (EV vs Model):")
            print(f"  {'Ticker':<12} {'Pos':>5} {'Cost':>8} {'Model':>8} {'EV':>10}")
            total_ev = 0.0
            for ticker, pos in self.portfolio.positions.items():
                if pos != 0:
                    cost = self.portfolio.cost_basis.get(ticker, 50.0)
                    model_fair = getattr(self, '_model_fair_values', {}).get(ticker, cost)
                    ev_per = (model_fair - cost) / 100 if pos > 0 else (cost - model_fair) / 100
                    pos_ev = ev_per * abs(pos)
                    total_ev += pos_ev
                    print(f"  {ticker[-10:]:<12} {pos:>+5} {cost:>7.1f}¢ {model_fair:>7.1f}¢ ${pos_ev:+.2f}")
            print(f"  {'TOTAL EV':<35} ${total_ev:+.2f}")
        print(self.order_mgr.get_order_summary())

    def _export_dashboard_state(self):
        """Export current state for dashboard visualization."""
        import json
        
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
            
            # Use cached prediction data if available
            live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
            if not live_data:
                continue
                
            home_score = live_data['homeTeam']['score']
            away_score = live_data['awayTeam']['score']
            
            # Reconstruct distribution params for this game
            # We need the most recent features
            period = live_data.get('period', 0)
            game_clock = live_data.get('gameClock', 'PT0M00.00S')
            total_seconds = self._parse_time(period, game_clock)
            
            # Get latest features (already computed in evaluate_game_rebalancing but not stored globally)
            # We'll just rebuild them here for the export
            live_features = self._build_features(game, home_score - away_score, total_seconds, period)
            params = self.spread_model.predict_distribution_params(live_features)
            
            # For Student-T, we need loc, scale, and df
            # NGBoost pred_dist returns the distribution object
            X_live = pd.DataFrame([{col: live_features.get(col, 0.0) for col in self.spread_model.feature_order}], 
                                  columns=self.spread_model.feature_order)
            
            # We'll just use the average of the ensemble for the dashboard
            locs = []
            scales = []
            dfs = []
            
            for model in self.spread_model.ensemble:
                dist = model.pred_dist(X_live.values)
                locs.append(dist.loc[0])
                scales.append(dist.scale[0])
                dfs.append(dist.df[0] if hasattr(dist, 'df') else 30.0)
            
            # Categorize features for display
            driving_features = {
                "live": {
                    "Pace": f"{live_features.get('live_pace', 0):.1f}",
                    "Momentum": f"{live_features.get('score_momentum', 0):+.1f}",
                    "Home eFG%": f"{live_features.get('home_efg', 0):.1%}",
                    "Away eFG%": f"{live_features.get('away_efg', 0):.1%}",
                    "TO Diff": f"{live_features.get('turnover_diff', 0):+d}",
                },
                "team_recent": {
                    "Home OffRtg": f"{live_features.get('home_team_recent_off_rtg', 0):.1f}",
                    "Home DefRtg": f"{live_features.get('home_team_recent_def_rtg', 0):.1f}",
                    "Away OffRtg": f"{live_features.get('away_team_recent_off_rtg', 0):.1f}",
                    "Away DefRtg": f"{live_features.get('away_team_recent_def_rtg', 0):.1f}",
                },
                "volatility": {
                    "Lead Changes": live_features.get('lead_changes', 0),
                    "Volatility": f"{live_features.get('score_volatility', 0):.2f}",
                }
            }
            
            game_state = {
                "game_id": game_id,
                "home_team": game['homeTeam']['teamTricode'],
                "away_team": game['awayTeam']['teamTricode'],
                "home_score": home_score,
                "away_score": away_score,
                "seconds_remaining": total_seconds,
                "period": period,
                "distribution": {
                    "loc": np.mean(locs) + (home_score - away_score),
                    "scale": np.mean(scales),
                    "df": np.mean(dfs)
                },
                "driving_features": driving_features,
                "markets": []
            }
            
            for market in match['spread_markets']:
                # Find matching open orders
                market_orders = [o for o in self.order_mgr.get_open_orders() if o.ticker == market.ticker]
                pending_buy = sum(o.size for o in market_orders if o.side == 'buy')
                pending_sell = sum(o.size for o in market_orders if o.side == 'sell')
                
                # Calculate pending exposure
                # exposure_delta expects side, price, size, current_pos
                curr_p = self.portfolio.positions.get(market.ticker, 0)
                pending_buy_exp = sum(self.risk_mgr.get_exposure_delta('buy', o.price, o.size, curr_p) for o in market_orders if o.side == 'buy')
                pending_sell_exp = sum(self.risk_mgr.get_exposure_delta('sell', o.price, o.size, curr_p) for o in market_orders if o.side == 'sell')
                
                # Calculate position exposure
                pos_exp = 0.0
                if curr_p != 0:
                    cost = self.portfolio.cost_basis.get(market.ticker, 50.0)
                    pos_exp = self.risk_mgr.get_exposure_delta('buy' if curr_p > 0 else 'sell', cost, abs(curr_p), 0)

                game_state["markets"].append({
                    "ticker": market.ticker,
                    "spread": market.spread,
                    "team": market.team,
                    "bid": market.yes_bid,
                    "ask": market.yes_ask,
                    "fair_value": getattr(self, '_model_fair_values', {}).get(market.ticker),
                    "position": self.portfolio.positions.get(market.ticker, 0),
                    "pending_buy": pending_buy,
                    "pending_buy_exp": pending_buy_exp,
                    "pending_sell": pending_sell,
                    "pending_sell_exp": pending_sell_exp,
                    "position_exp": pos_exp
                })
            
            state["games"].append(game_state)
            
        os.makedirs('data', exist_ok=True)
        with open('data/dashboard_state.json', 'w') as f:
            json.dump(state, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic Rebalancer - NBA Trading Bot')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    parser.add_argument('--min-edge', type=float, default=0.05, help='Min edge to trade')
    parser.add_argument('--interval', type=int, default=10, help='Seconds between iterations')
    args = parser.parse_args()
    
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
    kalshi_key_path = "key.key"
    
    trader = RebalancingLiveTrader(
        kalshi_key_id=kalshi_key_id, kalshi_key_path=kalshi_key_path,
        dry_run=not args.live, max_game_exposure=15.0, max_ticker_exposure=5.0, min_edge=args.min_edge
    )
    trader.run(interval=args.interval)

if __name__ == "__main__":
    main()
