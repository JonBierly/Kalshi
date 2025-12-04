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
        self.order_mgr = OrderManager(self.kalshi, dry_run=dry_run)
        self.mm_strategy = SpreadMarketMaker()
        
        # Load spread models (wrapped in SpreadDistributionModel)
        print("Loading spread model...")
        from spread_src.models.spread_model import SpreadDistributionModel
        self.spread_model = SpreadDistributionModel('models/nba_spread_model.pkl')
        
        # Track game tickers for risk management
        self.game_tickers = {}  # game_id -> [tickers]
    
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
                    self.portfolio.update_fill(
                        fill.ticker,
                        fill.side,
                        fill.price,
                        fill.size
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
                
                # Step 4: Place best new order (if we have room for one more)
                open_orders = self.order_mgr.get_open_orders()
                MAX_ORDERS = 10  # Allow 10 orders (5 pairs)
                
                if len(open_orders) < MAX_ORDERS and all_opportunities:
                    # Find best opportunity not already ordered
                    existing_keys = {(o.ticker, o.side) for o in open_orders}
                    new_opps = [o for o in all_opportunities if (o['ticker'], o['side']) not in existing_keys]
                    
                    if new_opps:
                        best_opp = max(new_opps, key=lambda x: x['ev'])
                        self._place_order(best_opp)
                    else:
                        print("All good opportunities already have orders")
                else:
                    if len(open_orders) >= MAX_ORDERS:
                        print(f"At order limit ({len(open_orders)}/{MAX_ORDERS})")
                    else:
                        print("No new opportunities")
                
                # Step 5: Display status
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
            
            # Generate two-sided quotes
            quotes = self._generate_two_sided_quotes(
                ticker=market.ticker,
                fair_value=fair_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                market_bid=market.yes_bid,
                market_ask=market.yes_ask,
                position=position,
                seconds_remaining=total_seconds
            )
            
            # Calculate liquidity factor (1.0 -> 0.5 based on spread)
            market_spread = market.yes_ask - market.yes_bid
            # Smooth penalty: 1.0 at 0¢ spread, 0.5 at 100¢+ spread
            liquidity_factor = max(0.5, 1.0 - (market_spread / 200.0))
            
            quote_str = ""
            
            if quotes['should_quote']:
                # Create BID opportunity (if not None)
                if quotes['bid_price'] is not None and quotes['bid_size'] > 0:
                    base_ev = fair_value - quotes['bid_price']
                    adjusted_ev = base_ev * liquidity_factor  # Apply liquidity penalty
                    
                    opportunities.append({
                        'ticker': market.ticker,
                        'side': 'buy',
                        'price': quotes['bid_price'],
                        'size': quotes['bid_size'],
                        'ev': adjusted_ev,
                        'reason': f"MM Bid @ {quotes['bid_price']:.0f}¢",
                        'is_mm_quote': True
                    })
                    quote_str = f"BID {quotes['bid_price']:.0f}¢"
                
                # Create ASK opportunity (if not None)
                if quotes['ask_price'] is not None and quotes['ask_size'] > 0:
                    base_ev = quotes['ask_price'] - fair_value
                    adjusted_ev = base_ev * liquidity_factor  # Apply liquidity penalty
                    
                    opportunities.append({
                        'ticker': market.ticker,
                        'side': 'sell',
                        'price': quotes['ask_price'],
                        'size': quotes['ask_size'],
                        'ev': adjusted_ev,
                        'reason': f"MM Ask @ {quotes['ask_price']:.0f}¢",
                        'is_mm_quote': True
                    })
                    if quote_str:
                        quote_str += f" | ASK {quotes['ask_price']:.0f}¢"
                    else:
                        quote_str = f"ASK {quotes['ask_price']:.0f}¢"
            else:
                quote_str = f"Skip: {quotes.get('reason', 'No quote')}"
            
            # Extract market name (e.g., "OKC >11.5")
            ticker_parts = market.ticker.split('-')
            market_name = ticker_parts[-1] if ticker_parts else market.ticker
            
            # Calculate spreads for display
            market_spread = market.yes_ask - market.yes_bid
            ci_width_pct = (ci_upper - ci_lower) * 100  # As percentage
            
            # Our quote spread (if quoting)
            if quotes['should_quote'] and quotes['bid_price'] and quotes['ask_price']:
                our_spread = f"{quotes['bid_price']:.0f}-{quotes['ask_price']:.0f}"
            else:
                our_spread = "-"
            
            # Format display line
            print(f"  {market_name:<12} "
                  f"Mkt:{market.yes_bid:3.0f}-{market.yes_ask:2.0f}¢ "
                  f"({market_spread:2.0f}¢) "
                  f"Model:{fair_value:3.0f}¢±{ci_width_pct:4.1f}% "
                  f"Our:{our_spread:>8} "
                  f"Pos:{pos_str:>3} "
                  f"{quote_str}")
        
        return opportunities
    
    def _calculate_kelly_size(
        self,
        fair_value: float,
        price: float,
        ci_width: float,
        position: int = 0
    ) -> int:
        """
        Calculate optimal position size using Kelly criterion.
        
        Args:
            fair_value: Model prediction in cents (0-100)
            price: Order price in cents
            ci_width: Confidence interval width (0-1)
            position: Current position
            
        Returns:
            Number of contracts (1-10)
        """
        # Get available capital
        bankroll = self.portfolio.get_available_capital()
        
        # Edge in cents
        edge = abs(fair_value - price)
        
        # Avoid division issues
        if price < 1 or price > 99 or bankroll < 0.10:
            return 1
        
        # Kelly percentage: f = (edge/100) / (price/100)
        kelly_pct = (edge / 100.0) / (price / 100.0)
        
        # Use quarter-Kelly for safety (conservative)
        kelly_fraction = 0.25
        
        # Reduce Kelly if uncertain (wide CI)
        if ci_width > 0.25:
            kelly_fraction *= 0.5  # Half-quarter-Kelly for very uncertain
        
        kelly_pct *= kelly_fraction
        
        # Convert to number of contracts
        # Each contract costs price/100 dollars
        dollars_per_contract = price / 100.0
        kelly_dollars = bankroll * kelly_pct
        kelly_size = kelly_dollars / dollars_per_contract
        
        # Round and apply bounds
        size = int(round(kelly_size))
        
        # Reduce if we already have position
        if abs(position) > 3:
            size = max(1, size // 2)
        
        # Hard bounds
        size = max(1, min(10, size))
        
        return size
    
    
    def _generate_two_sided_quotes(
        self,
        ticker: str,
        fair_value: float,      # Model probability in cents (0-100)
        ci_lower: float,        # Lower CI in probability (0-1)
        ci_upper: float,        # Upper CI in probability (0-1)
        market_bid: float,      # Current best bid in cents
        market_ask: float,      # Current best ask in cents
        position: int,          # Current position (-10 to +10)
        seconds_remaining: float
    ) -> dict:
        """
        Generate bid and ask quotes for two-sided market making.
        
        Strategy:
        1. Calculate spread width from confidence interval
        2. Post quotes around fair value
        3. Apply inventory skewing if positioned
        4. Ensure we beat existing market prices
        
        Args:
            ticker: Market ticker
            fair_value: Model fair value in cents (0-100)
            ci_lower: Lower bound of 90% CI (0-1)
            ci_upper: Upper bound of 90% CI (0-1)
            market_bid: Current best bid in cents
            market_ask: Current best ask in cents
            position: Current net position
            seconds_remaining: Time left in game
            
        Returns:
            {
                'should_quote': bool,
                'bid_price': float or None,
                'bid_size': int,
                'ask_price': float or None,
                'ask_size': int,
                'reason': str
            }
        """
        # 1. Calculate theoretical spread from confidence interval
        ci_width = ci_upper - ci_lower
        
        # Continuous spread function: CI 0% → 1¢, CI 50%+ → 10¢
        # Linear scaling with bounds
        half_spread = max(1, min(10, 1 + (ci_width * 18)))
        
        # Tighter spreads in last 5 minutes
        if seconds_remaining < 300:
            half_spread = max(1, half_spread / 2)
        
        
        # 2. Base quotes around fair value
        theo_bid = fair_value - half_spread
        theo_ask = fair_value + half_spread
        
        # 3. Apply inventory skewing
        #    If long → widen bid (less eager to buy), tighten ask (eager to sell)
        #    If short → tighten bid (eager to buy), widen ask (less eager to sell)
        skew = position * 0.5  # 0.5¢ per contract
        
        theo_bid -= skew
        theo_ask -= skew
        
        # 4. Apply price improvement (beat existing market by 1¢)
        tick_improvement = 1
        
        # For bid: want to be better than current bid (higher)
        comp_bid = min(theo_bid, market_bid + tick_improvement)
        
        # For ask: want to be better than current ask (lower)
        comp_ask = max(theo_ask, market_ask - tick_improvement)
        
        # 5. Sanity checks
        
        # Check: Spread not inverted
        if comp_bid >= comp_ask:
            return {
                'should_quote': False,
                'reason': 'Spread inverted'
            }
        
        # Check: Don't cross the spread
        if comp_bid >= market_ask - 1 or comp_ask <= market_bid + 1:
            return {
                'should_quote': False,
                'reason': 'Would cross spread'
            }
        
        # Check: Prices in valid range
        if comp_bid < 1 or comp_ask > 99:
            return {
                'should_quote': False,
                'reason': 'Out of bounds'
            }
        
        # 6. Determine sizes based on position
        MAX_POSITION = 5
        
        if abs(position) >= MAX_POSITION:
            # At position limit - only quote to reduce
            if position > 0:  # Long - only sell
                return {
                    'should_quote': True,
                    'bid_price': None,
                    'bid_size': 0,
                    'ask_price': comp_ask,
                    'ask_size': min(5, abs(position)),
                    'reason': f'Reduce long position ({position})'
                }
            else:  # Short - only buy
                return {
                    'should_quote': True,
                    'bid_price': comp_bid,
                    'bid_size': min(5, abs(position)),
                    'ask_price': None,
                    'ask_size': 0,
                    'reason': f'Reduce short position ({position})'
                }
        
        
        # Calculate optimal sizes using Kelly criterion
        bid_size = self._calculate_kelly_size(fair_value, comp_bid, ci_width, position)
        ask_size = self._calculate_kelly_size(fair_value, comp_ask, ci_width, position)
        
        # Skew sizes based on inventory (reduce less-preferred side)
        if position > 0:  # Long - less eager to buy more
            bid_size = max(1, bid_size // 2)
        elif position < 0:  # Short - less eager to sell more  
            ask_size = max(1, ask_size // 2)
        
        return {
            'should_quote': True,
            'bid_price': comp_bid,
            'bid_size': bid_size,
            'ask_price': comp_ask,
            'ask_size': ask_size,
            'reason': f'Two-sided @ {comp_bid:.0f}-{comp_ask:.0f}¢'
        }
    
    
    def _place_order(self, opp):
        """
        Place order after risk checks.
        
        Args:
            opp: Opportunity dict from evaluation
        """
        print(f"\nBest opportunity: {opp['reason']}")
        print(f"  {opp['side'].upper()} {opp['size']} {opp['ticker']} @ {opp['price']:.1f}¢")
        print(f"  Expected value: {opp['ev']:.1f}¢")
        
        # Risk check
        approved, reason = self.risk_mgr.check_new_order(
            ticker=opp['ticker'],
            side=opp['side'],
            price=opp['price'],
            size=opp['size'],
            current_positions=self.portfolio.positions,
            current_exposure=self.portfolio.get_exposure(),
            game_tickers=self.game_tickers,
            portfolio=self.portfolio  # For accurate game exposure calculation
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
        
        if order_id:
            print(f"  ✓ Order placed: {order_id}")
        else:
            print(f"  ✗ Order failed")


def main():
    """Run live market maker."""
    # Get credentials
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"  # Your hardcoded key
    kalshi_key_path = "key.key"
    
    # Configuration
    DRY_RUN = False  # SET TO FALSE FOR REAL TRADING!
    MAX_EXPOSURE = 10.0
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
