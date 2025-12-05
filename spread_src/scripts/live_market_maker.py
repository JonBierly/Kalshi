#!/usr/bin/env python
"""
Live market maker for Kalshi spread markets.

Provides liquidity by posting bid/ask quotes based on model predictions.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.execution.risk_manager import RiskManager
from spread_src.execution.portfolio import Portfolio
from spread_src.execution.order_manager import OrderManager
from spread_src.trading.market_maker import SpreadMarketMaker
from spread_src.trading.quote_generator import QuoteGenerator
from spread_src.trading.position_sizer import PositionSizer
from spread_src.trading.market_evaluator import MarketEvaluator
from spread_src.execution.trade_logger import TradeLogger
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.inference.spread_tracker import SpreadTracker
from src.data.kalshi import KalshiClient
import joblib


class LiveMarketMaker:
    """
    Live market making engine for spread markets.
    
    Loop:
    1. Fetch live game data + orderbooks
    2. Calculate model fair values
    3. Cancel existing orders
    4. Generate new quotes (one per iteration)
    5. Submit best opportunity
    6. Track fills and update positions
    """
    
    def __init__(
        self,
        kalshi_key_id,
        kalshi_key_path='key.key',
        dry_run=True,
        max_exposure=20.0,
        max_game_exposure=5.0
    ):
        """
        Initialize market maker.
        
        Args:
            kalshi_key_id: Kalshi API key
            kalshi_key_path: Path to private key
            dry_run: If True, log orders without executing
            max_exposure: Max $ at risk total
            max_game_exposure: Max $ at risk per game
        """
        self.dry_run = dry_run
        
        print("=" * 80)
        print("LIVE SPREAD MARKET MAKER")
        print("=" * 80)
        print(f"Mode: {'DRY-RUN (simulation)' if dry_run else 'LIVE (real money!)'}")
        print(f"Max exposure: ${max_exposure}")
        print(f"Max per game: ${max_game_exposure}")
        print("=" * 80)
        
        # Initialize components
        print("\n Initializing...")
        
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        self.tracker = SpreadTracker(kalshi_key_id, kalshi_key_path)
        
        self.risk_mgr = RiskManager(max_exposure, max_game_exposure)
        self.portfolio = Portfolio(initial_capital=max_exposure)
        
        # Sync with actual Kalshi positions
        self.portfolio.sync_positions(self.kalshi)
        
        # CRITICAL: Build game_tickers map from synced positions
        # Format: {game_id: [ticker1, ticker2, ...]}
        self.game_tickers = {}
        for ticker, pos in self.portfolio.positions.items():
            if pos != 0:
                # Extract game ID from ticker
                parts = ticker.split('-')
                if len(parts) >= 2:
                    game_id = parts[1][:10] if len(parts[1]) >= 10 else parts[1]
                    if game_id not in self.game_tickers:
                        self.game_tickers[game_id] = []
                    self.game_tickers[game_id].append(ticker)
        
        if self.game_tickers:
            print(f"\n📍 Tracking {len(self.game_tickers)} games with positions:")
            for game_id, tickers in self.game_tickers.items():
                print(f"  {game_id}: {len(tickers)} markets")
        
        self.order_mgr = OrderManager(self.kalshi, dry_run=dry_run)
        self.mm_strategy = SpreadMarketMaker()
        
        # Load spread models (wrapped in SpreadDistributionModel)
        print("Loading spread model...")
        from spread_src.models.spread_model import SpreadDistributionModel
        self.spread_model = SpreadDistributionModel('models/nba_spread_model.pkl')
        
        # Initialize modular components
        self.quote_generator = QuoteGenerator(max_position=5)
        self.position_sizer = PositionSizer(kelly_fraction=0.25)
        self.market_evaluator = MarketEvaluator(
            spread_model=self.spread_model,
            quote_generator=self.quote_generator,
            position_sizer=self.position_sizer
        )
        
        # Initialize database logger
        self.trade_logger = TradeLogger('data/nba_data.db')
        print("✓ Database logging enabled")
    
    def run(self, interval=30):
        """
        Main trading loop.
        
        Args:
            interval: Seconds between iterations
        """
        # Setup
        print("Matching games to spread markets...")
        self.tracker.setup()
        
        if not self.tracker.active_matches:
            print(" No games matched to spread markets")
            return
        
        print(f"✓ Tracking {len(self.tracker.active_matches)} game(s)\n")
        
        # Build game ticker mapping
        for match in self.tracker.active_matches:
            game = match['nba_game']
            markets = match['spread_markets']
            game_id = game['gameId']
            self.game_tickers[game_id] = [m.ticker for m in markets]
        
        print("Starting market making loop...")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'=' * 80}")
                print(f"Iteration {iteration} @ {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'=' * 80}\n")
                
                # Step 1: Check for fills
                fills = self.order_mgr.check_for_fills()
                for fill in fills:
                    # Get trade_id if we have it
                    trade_id = self.portfolio.trade_ids.get(fill.ticker)
                    
                    # Update portfolio (which auto-logs to DB if trade_id provided)
                    self.portfolio.update_fill(
                        ticker=fill.ticker,
                        side=fill.side,
                        price=fill.price,
                        size=fill.size,
                        trade_id=trade_id
                    )

                                
                # Step 2: Evaluate all markets and collect ALL opportunities
                all_opportunities = []
                
                for match in self.tracker.active_matches:
                    game = match['nba_game']
                    spread_markets = match['spread_markets']
                    
                    # Get live game data and evaluate
                    opps = self._evaluate_game(game, spread_markets)
                    if opps:
                        all_opportunities.extend(opps)
                
                # Step 3: Manage existing orders
                open_orders = self.order_mgr.get_open_orders()
                
                if open_orders:
                    print(f"\nManaging {len(open_orders)} pending orders...")
                    
                    # Build game time lookup for late-game cancellation
                    game_times = {}
                    for match in self.tracker.active_matches:
                        game = match['nba_game']
                        game_id = game['gameId']
                        live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
                        if live_data:
                            period = live_data.get('period', 4)
                            remaining_time = live_data.get('remaining_time', '0:00')
                            try:
                                if ':' in remaining_time:
                                    mins, secs = remaining_time.split(':')
                                    total_secs = int(mins) * 60 + int(secs)
                                else:
                                    total_secs = 0
                            except:
                                total_secs = 0
                            if period < 4:
                                total_secs += (4 - period) * 12 * 60
                            game_times[game_id] = total_secs
                    
                    for order in open_orders:
                        # Extract game ID from ticker (e.g., KXNBASPREAD-25DEC03LACATL-LAC8)
                        ticker_parts = order.ticker.split('-')
                        if len(ticker_parts) >= 2:
                            game_key = ticker_parts[1][:13]  # e.g., "25DEC03LACATL"
                            
                            # Find matching game time
                            order_game_time = None
                            for game_id, time_remaining in game_times.items():
                                if game_key in game_id:
                                    order_game_time = time_remaining
                                    break
                            
                            # Cancel if game has < 2 minutes
                            if order_game_time is not None and order_game_time < 120:
                                print(f"  {order.order_id}: Late game ({order_game_time}s left), canceling")
                                self.order_mgr.cancel_order(order.order_id)
                                continue
                        
                        # Find current opportunity for this ticker+side
                        current_opp = next(
                            (o for o in all_opportunities 
                             if o['ticker'] == order.ticker and o['side'] == order.side),
                            None
                        )
                        
                        if current_opp is None:
                            # No longer want to quote this ticker+side
                            print(f"  {order.order_id}: No longer quoting, canceling")
                            self.order_mgr.cancel_order(order.order_id)
                            continue
                        
                        # Check if our order is still competitive
                        price_diff = abs(order.price - current_opp['price'])
                        
                        # Keep if within 2¢ (prevents canceling own order that became best)
                        if price_diff <= 2.0:
                            print(f"  {order.order_id}: Still good, keeping")
                        else:
                            # Price moved significantly - cancel and replace
                            print(f"  {order.order_id}: Price changed by {price_diff:.1f}¢, canceling")
                            self.order_mgr.cancel_order(order.order_id)
                
                # Step 4: Place best new orders (if we have room)
                open_orders = self.order_mgr.get_open_orders()
                MAX_ORDERS = 30  # Allow up to 15 open orders
                MAX_POSITIONS = 10  # Max number of unique tickers with positions
                ORDERS_PER_ITERATION = 10  # Place up to 3 orders per iteration
                
                # Count current unique positions
                num_positions = sum(1 for pos in self.portfolio.positions.values() if pos != 0)
                
                # Calculate current exposure (including pending orders)
                current_exposure = self.portfolio.get_exposure()
                pending_exposure = sum(
                    self.risk_mgr._calculate_order_exposure(
                        o.side, o.price, o.size, self.portfolio.positions.get(o.ticker, 0)
                    ) for o in open_orders
                )
                total_exposure = current_exposure + pending_exposure
                at_max_exposure = total_exposure >= self.risk_mgr.ABSOLUTE_MAX_EXPOSURE * 0.95
                
                if len(open_orders) < MAX_ORDERS and all_opportunities:
                    # Find opportunities not already ordered
                    existing_keys = {(o.ticker, o.side) for o in open_orders}
                    new_opps = [o for o in all_opportunities if (o['ticker'], o['side']) not in existing_keys]
                    
                    # Filter out tickers we already have positions in if at position limit
                    if num_positions >= MAX_POSITIONS:
                        new_opps = [o for o in new_opps if self.portfolio.positions.get(o['ticker'], 0) != 0]
                    
                    # CRITICAL: If at max exposure, only consider position-reducing orders
                    if at_max_exposure:
                        reducing_opps = []
                        for opp in new_opps:
                            pos = self.portfolio.positions.get(opp['ticker'], 0)
                            is_reducing = (opp['side'] == 'buy' and pos < 0) or (opp['side'] == 'sell' and pos > 0)
                            if is_reducing:
                                reducing_opps.append(opp)
                        
                        if reducing_opps:
                            new_opps = reducing_opps
                            print(f"⚠️  At max exposure (${total_exposure:.2f}) - only placing position-reducing orders")
                        else:
                            new_opps = []
                            print(f"⚠️  At max exposure (${total_exposure:.2f}) - no position-reducing opportunities available")
                    
                    if new_opps:
                        # Sort by EV and take top N
                        new_opps.sort(key=lambda x: x['ev'], reverse=True)
                        orders_to_place = min(ORDERS_PER_ITERATION, MAX_ORDERS - len(open_orders), len(new_opps))
                        
                        for i in range(orders_to_place):
                            self._place_order(new_opps[i])
                    else:
                        if num_positions >= MAX_POSITIONS:
                            print(f"At position limit ({num_positions}/{MAX_POSITIONS})")
                        else:
                            print("All good opportunities already have orders")
                else:
                    if len(open_orders) >= MAX_ORDERS:
                        print(f"At order limit ({len(open_orders)}/{MAX_ORDERS})")
                    else:
                        print("No new opportunities")
                
                # Step 5: Check for settled positions
                self._check_and_settle_positions()
                
                # Step 6: Display status
                print(self.portfolio.get_position_summary())
                print(self.order_mgr.get_order_summary())
                
                # Wait
                print(f"\nWaiting {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping market maker...")
            self.order_mgr.cancel_all_orders()
            print(self.portfolio.get_position_summary())
            print("\n✓ Shutdown complete")
    
    def _evaluate_game(self, game, spread_markets):
        """
        Evaluate all markets for a game and return ALL profitable opportunities.
        
        Returns:
            List of opportunity dicts:
            [{
                'ticker': str,
                'side': str,
                'price': float,
                'size': int,
                'ev': float,
                'reason': str
            }, ...]
        """
        game_id = game['gameId']
        
        # Setup game context (same as tracker)
        if self.tracker.orch.prediction_engine.current_game_id != game_id:
            self.tracker.orch.setup_game_context(
                game_id,
                game['homeTeam']['teamId'],
                game['awayTeam']['teamId']
            )
            self.tracker.orch.feature_engine.reset()
        
        # Get live data
        live_data = self.tracker.orch.live_client.get_live_game_data(game_id)
        if not live_data:
            return []
        
        # Extract features (same as spread_tracker)
        home_score = live_data['homeTeam']['score']
        away_score = live_data['awayTeam']['score']
        score_diff = home_score - away_score
        
        # Calculate time remaining
        period = live_data.get('period', 4)
        game_clock = live_data.get('gameClock', 'PT0M00.00S')
        
        # Parse game clock (format: PT12M34.56S)
        import re
        total_seconds = 0
        
        try:
            match = re.search(r'PT(\d+)M([\d.]+)S', game_clock)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                period_seconds = minutes * 60 + seconds
            else:
                period_seconds = 0
            
            # Add remaining periods
            if period < 4:
                total_seconds = period_seconds + (4 - period) * 12 * 60
            else:
                total_seconds = period_seconds
                
        except Exception as e:
            print(f"  ⚠️  Error parsing time: {e}")
            print(f"      Period: {period}, Clock: {game_clock}")
            total_seconds = 0
        
        print(f"  Score: {game['awayTeam']['teamTricode']} {away_score} @ {game['homeTeam']['teamTricode']} {home_score}")
        print(f"  Time: Period {period}, {int(total_seconds//60)}:{int(total_seconds%60):02d} remaining")
        
        # FILTER: Skip games with <2 minutes left (too volatile)
        MIN_TIME_REMAINING = 120  # 2 minutes in seconds
        
        if total_seconds < MIN_TIME_REMAINING:
            print(f"  ⏰ Skipping: Only {total_seconds:.0f}s left (min: {MIN_TIME_REMAINING}s)")
            return []
        
        # Get scoresd catchup rate
        if total_seconds > 0 and score_diff != 0:
            time_elapsed = 2880 - total_seconds
            if time_elapsed > 0:
                points_per_second = score_diff / time_elapsed
                required_catchup_rate = points_per_second
            else:
                required_catchup_rate = 0.0
        else:
            required_catchup_rate = 0.0
        
        live_features = {
            'score_diff': score_diff,
            'seconds_remaining': total_seconds,
            'required_catchup_rate': required_catchup_rate,
            'is_home': 1,
            'home_efg': self.tracker.orch.feature_engine._calc_efg(self.tracker.orch.feature_engine.home_stats),
            'away_efg': self.tracker.orch.feature_engine._calc_efg(self.tracker.orch.feature_engine.away_stats),
            'turnover_diff': self.tracker.orch.feature_engine.home_stats['to'] - self.tracker.orch.feature_engine.away_stats['to'],
            'home_rebound_rate': self.tracker.orch.feature_engine._calc_reb_rate(
                self.tracker.orch.feature_engine.home_stats['reb'],
                self.tracker.orch.feature_engine.away_stats['reb']
            ),
            'game_id': game['gameId'],
            'home_team_id': game['homeTeam']['teamId'],
            'away_team_id': game['awayTeam']['teamId']
        }
        
        # Combine with context
        full_feats = {**live_features, **self.tracker.orch.prediction_engine.current_game_context}
        
        # Get spread distribution prediction
        mean_diff, std_diff = self.tracker.predict_spread_distribution(full_feats)
        
        home_tri = game['homeTeam']['teamTricode']
        away_tri = game['awayTeam']['teamTricode']
        
        print(f"\n{away_tri} @ {home_tri}:")
        print(f"  Score: {away_tri} {away_score} @ {home_tri} {home_score}")
        print(f"  Model: μ={mean_diff:+.1f}, σ={std_diff:.1f}")
        
        # Refresh orderbook prices
        print("  Refreshing prices...")
        for market in spread_markets:
            try:
                fresh = self.kalshi.get_market_details(market.ticker)
                if fresh:
                    market.yes_bid = fresh.get('yes_bid', market.yes_bid)
                    market.yes_ask = fresh.get('yes_ask', market.yes_ask)
            except:
                pass
        
        # Import for probability calculations
        from scipy import stats as scipy_stats
        
        # Evaluate each market
        opportunities = []
        
        print(f"\n  {'Market':<25} {'Bid-Ask':<12} {'Model (CI)':<20} {'Position':<10} {'Best EV'}")
        print(f"  {'-'*25} {'-'*12} {'-'*20} {'-'*10} {'-'*10}")
        
        # Get all thresholds for batch prediction
        thresholds = [m.spread for m in spread_markets]
        
        # Predict probabilities with confidence intervals from ensemble
        spread_result = self.spread_model.predict_spread_probabilities(full_feats, thresholds)
        
        for i, market in enumerate(spread_markets):
            # Determine if home or away market
            is_home_market = (market.team == home_tri)
            
            threshold = market.spread
            
            # Get probability and confidence interval from ensemble
            if is_home_market:
                # Home wins by >threshold
                model_prob = spread_result['probabilities'][i]
                ci_lower = spread_result['ci_90_lower'][i]
                ci_upper = spread_result['ci_90_upper'][i]
            else:
                # Away wins by >threshold = Home loses by >threshold
                # Need to recompute for negative threshold
                neg_result = self.spread_model.predict_spread_probabilities(full_feats, [-threshold])
                model_prob = 1 - neg_result['probabilities'][0]
                ci_lower = 1 - neg_result['ci_90_upper'][0]
                ci_upper = 1 - neg_result['ci_90_lower'][0]

                self.trade_logger.log_prediction(
                    game_id=game['gameId'],
                    ticker=market.ticker,
                    seconds_remaining=int(total_seconds),
                    score_diff=score_diff,
                    predicted_prob=model_prob,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper
                )
            
            # Convert to cents
            fair_value = model_prob * 100
            ci_lower_cents = ci_lower * 100
            ci_upper_cents = ci_upper * 100
            
            # Check current position
            position = self.portfolio.positions.get(market.ticker, 0)

            
            
            # Market info
            bid_ask = f"{market.yes_bid}-{market.yes_ask}¢"
            model_str = f"{fair_value:.0f}¢ ({ci_lower_cents:.0f}-{ci_upper_cents:.0f})"
            pos_str = f"{position:+d}" if position != 0 else "-"
            
            
            # Generate two-sided quotes using QuoteGenerator
            quotes = self.quote_generator.generate_quotes(
                fair_value=fair_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                market_bid=market.yes_bid,
                market_ask=market.yes_ask,
                position=position,
                seconds_remaining=total_seconds
            )
            
            # Create opportunities using MarketEvaluator
            market_spread = market.yes_ask - market.yes_bid
            ci_width = ci_upper - ci_lower
            bankroll = self.portfolio.get_available_capital()
            
            market_opps = self.market_evaluator.create_opportunities_from_quotes(
                ticker=market.ticker,
                fair_value=fair_value,
                quotes=quotes,
                market_spread=market_spread,
                position=position,
                bankroll=bankroll,
                ci_width=ci_width,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                seconds_remaining=int(total_seconds)
            )
            
            opportunities.extend(market_opps)
            
            # Only display markets with good opportunities (EV > 0)
            if market_opps:
                best_opp = max(market_opps, key=lambda x: x['ev'])
                
                # Extract market shortname
                ticker_parts = market.ticker.split('-')
                market_name = ticker_parts[-1] if ticker_parts else market.ticker
                
                # EV indicator
                if best_opp['ev'] > 30:
                    indicator = "🔥"
                elif best_opp['ev'] > 20:
                   indicator = "✨"
                else:
                    indicator = "💡"
                
                # Position indicator
                pos_str = f"(Pos: {position:+d})" if position != 0 else ""
                
                print(f"  {indicator} {market_name:<15} "
                      f"{best_opp['side'].upper():<5} "
                      f"{best_opp['size']:>2} @ {best_opp['price']:>5.1f}¢ "
                      f"→ EV: {best_opp['ev']:>4.1f}¢ "
                      f"{pos_str}")
        
        # Print summary
        if opportunities:
            print(f"\n📊 Found {len(opportunities)} opportunities (showing best by EV)")
        
        return opportunities
    
    
    def _place_order(self, opp):
        """
        Place order after risk checks.
        
        Args:
            opp: Opportunity dict from evaluation
        """
        print(f"\nBest opportunity: {opp['reason']}")
        print(f"  {opp['side'].upper()} {opp['size']} {opp['ticker']} @ {opp['price']:.1f}¢")
        print(f"  Expected value: {opp['ev']:.1f}¢")
        
        # Get pending orders for risk check
        pending_orders = self.order_mgr.get_open_orders()
        
        # Risk check
        approved, reason = self.risk_mgr.check_new_order(
            ticker=opp['ticker'],
            side=opp['side'],
            price=opp['price'],
            size=opp['size'],
            current_positions=self.portfolio.positions,
            current_exposure=self.portfolio.get_exposure(),
            game_tickers=self.game_tickers,
            portfolio=self.portfolio,  # For accurate game exposure calculation
            pending_orders=pending_orders  # Include pending orders in risk calc
        )
        
        if not approved:
            print(f"  ✗ REJECTED: {reason}")
            return
        
        print(f"  ✓ Risk check passed")
        
        # Place order
        order_id = self.order_mgr.place_limit_order(
            ticker=opp['ticker'],
            side=opp['side'],
            price=opp['price'],
            size=opp['size']
        )
        # Log order to database
        if order_id:
            # Extract game_id from ticker (e.g., NBAHOU-0012345678-B11.5 -> 0012345678)
            game_id = None
            if '-' in opp['ticker']:
                parts = opp['ticker'].split('-')
                if len(parts) >= 2:
                    game_id = parts[1][:10]  # First 10 chars after first dash
            
            trade_id = self.trade_logger.log_order_placed(
                ticker=opp['ticker'],
                side=opp['side'],
                price=opp['price'],
                size=opp['size'],
                game_id=game_id,
                model_fair=opp.get('model_fair'),
                ci_lower=opp.get('ci_lower'),
                ci_upper=opp.get('ci_upper'),
                market_spread=opp.get('market_spread'),
                seconds_remaining=opp.get('seconds_remaining'),
                position_before=self.portfolio.positions.get(opp['ticker'], 0)
            )
            # Store trade_id for later fill logging
            self.portfolio.trade_ids[opp['ticker']] = trade_id
            print(f"  ✓ Order placed: {order_id}")

        else:
            print(f"  ✗ Order failed")
    
    def _check_and_settle_positions(self):
        """
        Check Kalshi for settled markets and close positions.
        
        Strategy:
        1. Check each open position
        2. Query Kalshi API for market status
        3. If settled, calculate P&L and close position
        4. Log to database
        """
        for ticker, position in list(self.portfolio.positions.items()):
            if position == 0:
                continue
            
            try:
                # Get market details from Kalshi
                market = self.kalshi.get_market_details(ticker)
                
                if not market:
                    continue
                
                # Check if market is settled
                status = market.get('status')
                
                if status == 'settled':
                    # Get settlement result
                    result = market.get('result')  # 'yes' or 'no'
                    
                    if result:
                        outcome = (result == 'yes')
                        
                        # Get trade_id for logging
                        trade_id = self.portfolio.trade_ids.get(ticker)
                        
                        # Calculate P&L before settlement
                        cost = self.portfolio.cost_basis.get(ticker, 0.0) * abs(position)
                        if outcome:
                            payout = position * 1.0 if position > 0 else 0.0
                        else:
                            payout = 0.0
                        
                        if position > 0:
                            realized_pnl = payout - cost
                        else:
                            realized_pnl = cost - payout
                        
                        # Settle in portfolio
                        self.portfolio.settle_market(ticker, outcome)
                        
                        # Log to database
                        if trade_id:
                            self.trade_logger.log_position_closed(
                                trade_id=trade_id,
                                realized_pnl=realized_pnl
                            )
                        
                        print(f"\n{'='*50}")
                        print(f"🏁 SETTLED: {ticker}")
                        print(f"   Result: {result.upper()}")
                        print(f"   Position: {position}")
                        print(f"   P&L: ${realized_pnl:+.2f}")
                        print(f"{'='*50}\n")
                
            except Exception as e:
                # Don't crash the main loop on settlement errors
                print(f"⚠️  Error checking settlement for {ticker}: {e}")
                continue


def main():
    """Run live market maker."""
    # Get credentials
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"  # Your hardcoded key
    kalshi_key_path = "key.key"
    
    # Configuration
    DRY_RUN = False  # SET TO FALSE FOR REAL TRADING!
    MAX_EXPOSURE = 30.0
    MAX_GAME_EXPOSURE = 5.0
    UPDATE_INTERVAL = 15
    
    # Initialize and run
    mm = LiveMarketMaker(
        kalshi_key_id,
        kalshi_key_path,
        dry_run=DRY_RUN,
        max_exposure=MAX_EXPOSURE,
        max_game_exposure=MAX_GAME_EXPOSURE
    )
    
    mm.run(interval=UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
