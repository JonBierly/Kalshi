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
from data.kalshi import KalshiClient
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
        
        # Track model fair values for position display
        self.model_fair_values = {}  # ticker -> fair_value in cents
        
        # Initialize components
        print("\n✓ Initializing...")
        
        self.kalshi = KalshiClient(kalshi_key_id, kalshi_key_path)
        
        self.risk_mgr = RiskManager(max_exposure, max_game_exposure)
        self.portfolio = Portfolio(max_exposure=max_exposure)
        
        # Sync with actual Kalshi account - fetch fresh state
        self.portfolio.refresh_state(self.kalshi)
        
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
        
        # Initialize spread tracker
        self.tracker = SpreadTracker(kalshi_key_id, kalshi_key_path)
        
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
                
                # Step 1: Refresh state from Kalshi (ensures fresh data)
                self.portfolio.refresh_state(self.kalshi)
                
                # Step 2: Check for fills
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
                    game_times = self.order_mgr.build_game_times_map(
                        self.tracker.active_matches,
                        self.tracker.orch.live_client
                    )
                    
                    for order in open_orders:
                        # Extract game key from ticker
                        from spread_src.execution.risk_manager import RiskManager
                        game_key = RiskManager.extract_game_key(order.ticker)
                        
                        if game_key:
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
                
                
                # Step 4: Per-game order management and placement
                open_orders = self.order_mgr.get_open_orders()
                
                # Group opportunities by game
                games_dict = {}
                for opp in all_opportunities:
                    # Extract game ID from ticker
                    parts = opp['ticker'].split('-')
                    if len(parts) >= 2:
                        game_id = parts[1][:10] if len(parts[1]) >= 10 else parts[1]
                        if game_id not in games_dict:
                            games_dict[game_id] = []
                        games_dict[game_id].append(opp)
                
                # Process each game independently
                for game_id, game_opps in games_dict.items():
                    self._process_game(game_id, game_opps, open_orders)
                

                
                # (State already refreshed at iteration start)
                
                # Step 6: Display status
                self._print_enhanced_position_summary()
                print(self.order_mgr.get_order_summary())
                
                # Wait
                print(f"\nWaiting {interval}s...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping market maker...")
            self.order_mgr.cancel_all_orders()
            self._print_enhanced_position_summary()
            print("\n✓ Shutdown complete")
    
    def _print_enhanced_position_summary(self):
        """Print portfolio summary with model fair values."""
        print("\n=== PORTFOLIO ===")
        print(f"Cash: ${self.portfolio.cash:.2f}")
        print(f"Exposure: ${self.portfolio.get_exposure():.2f}")
        print(f"Realized P&L: ${self.portfolio.realized_pnl:+.2f}")
        print(f"\nPositions:")
        
        if not self.portfolio.positions or all(p == 0 for p in self.portfolio.positions.values()):
            print("  (none)")
        else:
            for ticker, pos in self.portfolio.positions.items():
                if pos != 0:
                    cost = self.portfolio.cost_basis.get(ticker, 0.0)
                    market_name = ticker.split('-')[-1] if '-' in ticker else ticker[-10:]
                    
                    # Get model fair value if available
                    model_fair = self.model_fair_values.get(ticker)
                    if model_fair is not None:
                        print(f"  {market_name}: {pos:+d} @ {cost:.1f}¢ (Model: {model_fair:.1f}¢)")
                    else:
                        print(f"  {market_name}: {pos:+d} @ {cost:.1f}¢")
    
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
        
        # Calculate actual spread
        actual_spread = home_score - away_score
        
        print(f"\n{away_tri} {away_score} @ {home_tri} {home_score} (Spread: {actual_spread:+.0f} | Model: {mean_diff:+.1f}±{std_diff:.1f})")
        
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
        
        print(f"\n  {'Market':<15} {'Position':<10} {'Order':<20} {'Model':<10} {'EV':<10}")
        print(f"  {'-'*15} {'-'*10} {'-'*20} {'-'*10} {'-'*10}")
        
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
            
            # Store model fair value for position display
            self.model_fair_values[market.ticker] = fair_value
            
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
                seconds_remaining=total_seconds,
                cost_basis=self.portfolio.cost_basis.get(market.ticker, None)
            )
            
            # Create opportunities using MarketEvaluator
            market_spread = market.yes_ask - market.yes_bid
            ci_width = ci_upper - ci_lower
            # Use max game exposure as bankroll for Kelly sizing (each game gets independent allocation)
            bankroll = self.risk_mgr.ABSOLUTE_MAX_GAME
            
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
                
                # Position display
                pos_display = f"{position:+d}" if position != 0 else "-"
                
                # Order display: SIDE SIZE @ PRICE
                order_str = f"{best_opp['side'].upper()} {best_opp['size']} @ {best_opp['price']:.1f}¢"
                
                # Model value
                model_display = f"{fair_value:.1f}¢"
                
                # EV display
                ev_display = f"{best_opp['ev']:+.1f}¢"
                
                print(f"  {indicator} {market_name:<12} {pos_display:<10} {order_str:<20} {model_display:<10} {ev_display}")
        
        # Print summary
        if opportunities:
            print(f"\n📊 Found {len(opportunities)} opportunities (showing best by EV)")
        
        return opportunities
    
    def _process_game(self, game_id, game_opps, open_orders):
        """
        Process all order placement for a single game.
        
        Strategy:
        1. Calculate current game exposure (positions + pending orders)
        2. Filter opportunities by EV threshold (5¢)
        3. Prioritize position-reducing orders if positions are losing
        4. Place orders on both bid and ask if both +EV
        5. Respect game exposure limit
        
        Args:
            game_id: Game identifier
            game_opps: List of opportunities for this game
            open_orders: All currently open orders
        """
        EV_THRESHOLD_OPENING = 5.0  # Minimum 5¢ EV for new positions
        EV_THRESHOLD_CLOSING = 0.0  # Minimum 2¢ EV for closing positions (lower to reduce risk)
        
        # Filter opportunities by EV threshold (checking if closing or opening)
        good_opps = []
        for opp in game_opps:
            pos = self.portfolio.positions.get(opp['ticker'], 0)
            is_closing = (opp['side'] == 'buy' and pos < 0) or (opp['side'] == 'sell' and pos > 0)
            
            threshold = EV_THRESHOLD_CLOSING if is_closing else EV_THRESHOLD_OPENING
            if opp['ev'] >= threshold:
                good_opps.append(opp)
        
        if not good_opps:
            return  # No profitable opportunities
        
        # Get tickers for this game
        game_tickers = self.game_tickers.get(game_id, [])
        
        # Calculate current game exposure (positions + pending orders)
        game_exposure = 0.0
        
        # Add exposure from filled positions
        for ticker in game_tickers:
            pos = self.portfolio.positions.get(ticker, 0)
            if pos != 0:
                cost_basis = self.portfolio.cost_basis.get(ticker, 50.0)
                if pos > 0:
                    game_exposure += (cost_basis / 100.0) * pos
                else:
                    game_exposure += ((100 - cost_basis) / 100.0) * abs(pos)
        
        # Add exposure from pending orders in this game
        for order in open_orders:
            if order.ticker in good_opps[0]['ticker']:  # Same game
                order_pos = self.portfolio.positions.get(order.ticker, 0)
                game_exposure += self.risk_mgr._calculate_order_exposure(
                    order.side, order.price, order.size, order_pos
                )
        
        # Separate closing vs opening opportunities
        closing_opps = []
        opening_opps = []
        
        for opp in good_opps:
            pos = self.portfolio.positions.get(opp['ticker'], 0)
            is_closing = (opp['side'] == 'buy' and pos < 0) or (opp['side'] == 'sell' and pos > 0)
            
            if is_closing:
                # Check if position is losing
                if pos != 0:
                    cost_basis = self.portfolio.cost_basis.get(opp['ticker'], 50.0)
                    fair_value = opp.get('model_fair', 50.0)
                    
                    if pos > 0:
                        pnl = (fair_value - cost_basis) * pos / 100
                    else:
                        pnl = (cost_basis - fair_value) * abs(pos) / 100
                    
                    # Mark high priority if losing
                    opp['priority'] = 'HIGH' if pnl < -0.25 else 'NORMAL'
                    opp['pnl'] = pnl
                
                closing_opps.append(opp)
            else:
                opp['priority'] = 'NORMAL'
                opening_opps.append(opp)
        
        # Sort closing by priority (HIGH first), then by EV
        closing_opps.sort(key=lambda x: (x.get('priority') != 'HIGH', -x['ev']))
        opening_opps.sort(key=lambda x: -x['ev'])
        
        # Determine which orders to place
        orders_to_place = []
        
        # Always try to place closing orders (even if over limit)
        if closing_opps:
            orders_to_place.extend(closing_opps)
        
        # Only add opening orders if under game exposure limit
        if game_exposure < self.risk_mgr.ABSOLUTE_MAX_GAME:
            for opp in opening_opps:
                # Calculate potential new exposure
                pos = self.portfolio.positions.get(opp['ticker'], 0)
                new_exposure = self.risk_mgr._calculate_order_exposure(
                    opp['side'], opp['price'], opp['size'], pos
                )
                
                if game_exposure + new_exposure <= self.risk_mgr.ABSOLUTE_MAX_GAME:
                    orders_to_place.append(opp)
                    game_exposure += new_exposure
                else:
                    break  # At limit
        
        # Filter out opportunities that already have orders
        existing_keys = {(o.ticker, o.side) for o in open_orders}
        new_orders = [o for o in orders_to_place if (o['ticker'], o['side']) not in existing_keys]
        
        # Place orders
        if new_orders:
            print(f"\n🎯 Game {game_id}: Placing {len(new_orders)} orders (Exposure: ${game_exposure:.2f}/${self.risk_mgr.ABSOLUTE_MAX_GAME:.2f})")
            for opp in new_orders:
                priority_marker = "🔴" if opp.get('priority') == 'HIGH' else ""
                closing_marker = "[CLOSING]" if opp in closing_opps else "[OPENING]"
                print(f"  {priority_marker} {closing_marker} {opp['ticker'][-6:]} {opp['side'].upper()} {opp['size']} @ {opp['price']:.1f}¢ (EV: {opp['ev']:.1f}¢)")
                self._place_order(opp)
    
    
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

        else:
            print(f"  ✗ Order failed")
    
    
    def _check_and_settle_positions(self):
        """
        Check Kalshi for settled markets and close positions.
        
        Delegates to Portfolio.settle_unsettled_trades() which handles all settlement logic.
        Then re-syncs positions from Kalshi to ensure exposure is accurate.
        """
        # Try to settle trades from database
        self.portfolio.settle_unsettled_trades(self.kalshi, self.trade_logger)
        
        # Re-sync positions from Kalshi to catch any settlements we missed
        # This ensures our exposure calculation matches Kalshi's actual state
        self.portfolio.sync_positions(self.kalshi)
    


def main():
    """Run live market maker."""
    # Get credentials
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"  # Your hardcoded key
    kalshi_key_path = "key.key"
    
    # Configuration
    DRY_RUN = False  # SET TO FALSE FOR REAL TRADING!
    MAX_EXPOSURE = 40.0
    MAX_GAME_EXPOSURE = 7.0
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
