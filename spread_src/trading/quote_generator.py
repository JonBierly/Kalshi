"""
Quote generation for two-sided market making.

Handles spread calculation, inventory skewing, and price improvement.
"""


class QuoteGenerator:
    """Generate bid/ask quotes for market making."""
    
    def __init__(
        self, 
        max_position: int = 5,
        time_urgency_threshold: float = 600.0,
        wide_spread_threshold: float = 20.0,
        tight_spread_threshold: float = 5.0
    ):
        """
        Initialize quote generator.
        
        Args:
            max_position: Maximum position per market (±)
            time_urgency_threshold: Start urgency pricing below this many seconds (default: 600s = 10min)
            wide_spread_threshold: Spread width for full model trust (default: 20¢)
            tight_spread_threshold: Spread width for zero model trust (default: 5¢)
        """
        self.max_position = max_position
        self.time_urgency_threshold = time_urgency_threshold
        self.wide_spread_threshold = wide_spread_threshold
        self.tight_spread_threshold = tight_spread_threshold
    
    def generate_quotes(
        self,
        fair_value: float,
        ci_lower: float,
        ci_upper: float,
        market_bid: float,
        market_ask: float,
        position: int,
        seconds_remaining: float,
        cost_basis: float = None
    ) -> dict:
        """
        Generate quotes with different strategies for opening vs closing positions.
        
        Strategy:
        - Opening positions: Conservative (spread calc, skew, price improvement)
        - Closing positions: Dynamic pricing (time urgency + position risk + market efficiency)
        
        Args:
            fair_value: Model fair value in cents (0-100)
            ci_lower: Lower bound of 90% CI (0-1)
            ci_upper: Upper bound of 90% CI (0-1)
            market_bid: Current best bid in cents
            market_ask: Current best ask in cents
            position: Current net position (+long, -short)
            seconds_remaining: Time left in game
            cost_basis: Average entry price (optional, unused in new logic)
            
        Returns:
            {
                'should_quote': bool,
                'bid_price': float or None,
                'ask_price': float or None,
                'reason': str
            }
        """
        ci_width = ci_upper - ci_lower
        market_spread = market_ask - market_bid
        
        # Check position limits - force closing if at max
        if abs(position) >= self.max_position:
            return self._generate_reducing_quote_dynamic(
                position, fair_value, market_bid, market_ask, 
                seconds_remaining, market_spread, ci_width
            )
        
        # Decide if we're closing or opening a position
        # Note: We only quote ONE side at a time for simplicity
        if position > 0:
            # Long position - offer to close (sell) using dynamic pricing
            close_price = self._calculate_closing_price(
                fair_value=fair_value,
                market_price=market_ask,
                position=position,
                seconds_remaining=seconds_remaining,
                market_spread_width=market_spread,
                ci_width=ci_width,
                side='sell'
            )
            
            # Validate the closing price
            if close_price < 1 or close_price > 99:
                return {
                    'should_quote': False,
                    'bid_price': None,
                    'ask_price': None,
                    'reason': 'Close price out of bounds'
                }
            
            # Also check if we want to increase position (buy more) - be conservative
            theo_bid = self._calculate_opening_bid(
                fair_value, ci_width, market_bid, position, seconds_remaining
            )
            
            return {
                'should_quote': True,
                'bid_price': theo_bid if theo_bid and theo_bid > market_bid else None,
                'ask_price': close_price,
                'reason': f'Close long @ {close_price:.0f}¢, open @ {theo_bid:.0f}¢' if theo_bid else f'Close long @ {close_price:.0f}¢'
            }
            
        elif position < 0:
            # Short position - offer to close (buy) using dynamic pricing
            close_price = self._calculate_closing_price(
                fair_value=fair_value,
                market_price=market_bid,
                position=position,
                seconds_remaining=seconds_remaining,
                market_spread_width=market_spread,
                ci_width=ci_width,
                side='buy'
            )
            
            # Validate the closing price
            if close_price < 1 or close_price > 99:
                return {
                    'should_quote': False,
                    'bid_price': None,
                    'ask_price': None,
                    'reason': 'Close price out of bounds'
                }
            
            # Also check if we want to increase position (sell more) - be conservative
            theo_ask = self._calculate_opening_ask(
                fair_value, ci_width, market_ask, position, seconds_remaining
            )
            
            return {
                'should_quote': True,
                'bid_price': close_price,
                'ask_price': theo_ask if theo_ask and theo_ask < market_ask else None,
                'reason': f'Close short @ {close_price:.0f}¢, open @ {theo_ask:.0f}¢' if theo_ask else f'Close short @ {close_price:.0f}¢'
            }
            
        else:
            # No position - use conservative two-sided opening strategy
            return self._generate_opening_quotes(
                fair_value, ci_width, market_bid, market_ask, 
                position, seconds_remaining
            )
    
    def _calculate_spread(self, ci_width: float, seconds_remaining: float) -> float:
        """
        Calculate half-spread from confidence interval width.
        
        Uses continuous function: CI 0% → 1¢, CI 50%+ → 10¢
        
        Args:
            ci_width: Confidence interval width (0-1)
            seconds_remaining: Seconds left in game
            
        Returns:
            Half-spread in cents
        """
        # Linear scaling with bounds
        half_spread = max(1, min(10, 1 + (ci_width * 18)))
        
        # Tighter spreads in last 5 minutes
        if seconds_remaining < 300:
            half_spread = max(1, half_spread / 2)
        
        return half_spread
    
    def _calculate_urgency_premium(
        self,
        position: int,
        cost_basis: float,
        fair_value: float,
        urgency_factor: float = 0.5,
        max_premium: float = 20.0
    ) -> float:
        """
        Calculate urgency premium for losing positions.
        
        When a position is underwater, add premium to closing quotes
        to increase likelihood of exit.
        
        Args:
            position: Current position (+long, -short)
            cost_basis: Average entry price in cents
            fair_value: Model fair value in cents
            urgency_factor: Fraction of loss to add (0.5 = 50%)
            max_premium: Maximum urgency premium in cents
            
        Returns:
            Urgency premium in cents (always >= 0)
        """
        if position == 0 or cost_basis is None:
            return 0.0
        
        # Calculate mark-to-market P&L per contract
        if position > 0:  # Long position
            pnl_per_contract = fair_value - cost_basis
        else:  # Short position
            pnl_per_contract = cost_basis - fair_value
        
        # Only add urgency for losing positions
        if pnl_per_contract >= 0:
            return 0.0  # Position is profitable, no urgency
        
        # Calculate urgency based on loss severity
        loss_per_contract = abs(pnl_per_contract)
        urgency = loss_per_contract * urgency_factor
        
        # Cap maximum urgency
        return min(urgency, max_premium)
    
    def _apply_inventory_skew(
        self,
        bid: float,
        ask: float,
        position: int,
        urgency_premium: float = 0.0
    ) -> tuple:
        """
        Skew quotes based on position + urgency.
        
        Base skew: 0.5¢ per contract for inventory management
        Urgency: Added to closing side when position is losing
        
        If long → widen bid (less eager to buy), tighten ask (eager to sell)
        If short → tighten bid (eager to buy), widen ask (less eager to sell)
        
        Args:
            bid: Theoretical bid price
            ask: Theoretical ask price
            position: Current position
            urgency_premium: Premium to add to closing side (from urgency calc)
            
        Returns:
            (skewed_bid, skewed_ask)
        """
        base_skew = position * 0.5  # 0.5¢ per contract
        
        # Apply urgency to CLOSING side only
        if position > 0:  # Long - closing side is ASK (sell)
            skewed_bid = bid - base_skew
            skewed_ask = ask - base_skew - urgency_premium  # More aggressive sell
        elif position < 0:  # Short - closing side is BID (buy)
            skewed_bid = bid - base_skew + urgency_premium  # More aggressive buy
            skewed_ask = ask - base_skew
        else:  # No position
            skewed_bid = bid
            skewed_ask = ask
        
        return skewed_bid, skewed_ask
    
    def _apply_price_improvement(
        self,
        theo_bid: float,
        theo_ask: float,
        market_bid: float,
        market_ask: float,
        tick_size: float = 1.0
    ) -> tuple:
        """
        Beat existing market by one tick.
        
        Args:
            theo_bid: Theoretical bid price
            theo_ask: Theoretical ask price
            market_bid: Current market bid
            market_ask: Current market ask
            tick_size: Minimum price increment
            
        Returns:
            (competitive_bid, competitive_ask)
        """
        # Bid: want to be better than current bid (higher)
        comp_bid = min(theo_bid, market_bid + tick_size)
        
        # Ask: want to be better than current ask (lower)
        comp_ask = max(theo_ask, market_ask - tick_size)
        
        return comp_bid, comp_ask
    
    def _validate_quotes(
        self,
        bid: float,
        ask: float,
        market_bid: float,
        market_ask: float
    ) -> dict:
        """
        Validate quotes meet sanity checks.
        
        Args:
            bid: Our bid price
            ask: Our ask price
            market_bid: Current market bid
            market_ask: Current market ask
            
        Returns:
            {'valid': bool, 'reason': str}
        """
        # Check: Spread not inverted
        if bid >= ask:
            return {'valid': False, 'reason': 'Spread inverted'}
        
        # Check: Don't cross the spread
        if bid >= market_ask - 1 or ask <= market_bid + 1:
            return {'valid': False, 'reason': 'Would cross spread'}
        
        # Check: Prices in valid range (1-99¢ for Kalshi)
        if bid < 1 or ask > 99:
            return {'valid': False, 'reason': 'Out of bounds'}
        
        return {'valid': True, 'reason': 'OK'}
    
    def _calculate_competitive_price(
        self,
        fair_value: float,
        market_price: float,
        position: int,
        seconds_remaining: float,
        market_spread_width: float,
        ci_width: float,
        side: str
    ) -> float:
        """
        Calculate competitive quote price using multi-factor model.
        
        Works for both opening AND closing positions.
        Uses three factors (combined via max):
        1. Time urgency: Exponential as game approaches end
        2. Position risk: Linear as position approaches max (0 for opening)
        3. Market efficiency: Based on spread width
        
        Args:
            fair_value: Model fair value in cents
            market_price: Current market bid (if buying) or ask (if selling)
            position: Current position (signed: +long, -short, 0 for opening)
            seconds_remaining: Time left in game
            market_spread_width: Current market spread (ask - bid)
            ci_width: Model confidence interval width (0-1)
            side: 'buy' or 'sell'
            
        Returns:
            Quote price in cents
        """
        # 1. Calculate edge from model confidence
        # Wider CI = less confident = larger edge needed
        edge = ci_width * 50.0  # 10% CI → 5¢ edge
        
        # 2. Time urgency factor (exponential decay in last 10 minutes)
        if seconds_remaining > self.time_urgency_threshold:
            time_factor = 0.0
        else:
            # Exponential: approaches 1.0 as time runs out
            time_factor = 1.0 - (seconds_remaining / self.time_urgency_threshold) ** 2
        
        # 3. Position risk factor (linear scaling)
        # Larger position → more urgency to reduce
        # For opening positions (position=0), this will be 0
        position_factor = abs(position) / self.max_position
        
        # 4. Market efficiency factor (spread-based)
        # Wide spread → market is inefficient → trust our model more
        if market_spread_width >= self.wide_spread_threshold:
            efficiency_factor = 1.0
        elif market_spread_width <= self.tight_spread_threshold:
            efficiency_factor = 0.0
        else:
            # Linear interpolation between thresholds
            efficiency_factor = (market_spread_width - self.tight_spread_threshold) / \
                              (self.wide_spread_threshold - self.tight_spread_threshold)
        
        # 5. Combine factors - take max (most aggressive factor wins)
        trust_factor = max(time_factor, position_factor, efficiency_factor)
        
        # 6. Blend between conservative and aggressive pricing
        # Special case: No liquidity (market_price is 0 or very low)
        # In this case, just post at fair value to create a market
        if market_price <= 1:
            # No existing market - post at fair value (with edge for selling)
            if side == 'sell':
                final_price = max(fair_value + edge, 1.0)
            else:  # buy
                final_price = max(fair_value - edge, 1.0)
        else:
            # Normal case: blend between conservative and aggressive
            if side == 'sell':
                # Selling: conservative = just beat ask, aggressive = fair + edge
                conservative_price = market_price - 1.0
                aggressive_price = fair_value + edge
            else:  # 'buy'
                # Buying: conservative = just beat bid, aggressive = fair - edge
                conservative_price = market_price + 1.0
                aggressive_price = fair_value - edge
            
            # Linear blend based on trust factor
            final_price = (1.0 - trust_factor) * conservative_price + trust_factor * aggressive_price
        
        # Ensure price is in valid range (1-99¢)
        final_price = max(1.0, min(99.0, final_price))
        
        return final_price
    
    def _calculate_closing_price(
        self,
        fair_value: float,
        market_price: float,
        position: int,
        seconds_remaining: float,
        market_spread_width: float,
        ci_width: float,
        side: str
    ) -> float:
        """
        Calculate quote price for closing a position.
        
        Wrapper around _calculate_competitive_price for backwards compatibility.
        """
        return self._calculate_competitive_price(
            fair_value, market_price, position, seconds_remaining,
            market_spread_width, ci_width, side
        )

    
    def _generate_reducing_quote(
        self,
        position: int,
        bid: float,
        ask: float
    ) -> dict:
        """
        Generate quote to reduce position when at limit.
        
        Args:
            position: Current position
            bid: Competitive bid price
            ask: Competitive ask price
            
        Returns:
            Quote dict with only reducing side
        """
        if position > 0:  # Long - only sell
            return {
                'should_quote': True,
                'bid_price': None,
                'ask_price': ask,
                'reason': f'Reduce long position ({position})'
            }
        else:  # Short - only buy
            return {
                'should_quote': True,
                'bid_price': bid,
                'ask_price': None,
                'reason': f'Reduce short position ({position})'
            }

    def _generate_opening_quotes(
        self,
        fair_value: float,
        ci_width: float,
        market_bid: float,
        market_ask: float,
        position: int,
        seconds_remaining: float
    ) -> dict:
        """
        Generate competitive two-sided quotes for opening new positions.
        
        Uses dynamic pricing with time urgency + market efficiency factors.
        Position factor = 0 since we're opening (no existing position).
        """
        market_spread = market_ask - market_bid
        
        # Use competitive pricing for both bid and ask
        # Position = 0 since we're opening
        comp_bid = self._calculate_competitive_price(
            fair_value=fair_value,
            market_price=market_bid,
            position=0,  # No position when opening
            seconds_remaining=seconds_remaining,
            market_spread_width=market_spread,
            ci_width=ci_width,
            side='buy'
        )
        
        comp_ask = self._calculate_competitive_price(
            fair_value=fair_value,
            market_price=market_ask,
            position=0,  # No position when opening
            seconds_remaining=seconds_remaining,
            market_spread_width=market_spread,
            ci_width=ci_width,
            side='sell'
        )
        
        # Sanity checks
        sanity_check = self._validate_quotes(comp_bid, comp_ask, market_bid, market_ask)
        if not sanity_check['valid']:
            return {
                'should_quote': False,
                'bid_price': None,
                'ask_price': None,
                'reason': sanity_check['reason']
            }
        
        return {
            'should_quote': True,
            'bid_price': comp_bid,
            'ask_price': comp_ask,
            'reason': f'Two-sided @ {comp_bid:.0f}-{comp_ask:.0f}¢'
        }
    
    def _calculate_opening_bid(
        self,
        fair_value: float,
        ci_width: float,
        market_bid: float,
        position: int,
        seconds_remaining: float
    ) -> float | None:
        """Calculate conservative bid for opening or increasing position."""
        half_spread = self._calculate_spread(ci_width, seconds_remaining)
        theo_bid = fair_value - half_spread
        
        # Just beat market by 1¢ (conservative)
        comp_bid = min(theo_bid, market_bid + 1.0)
        
        # Only quote if it beats market and is reasonable
        if comp_bid > market_bid and comp_bid >= 1 and comp_bid < fair_value:
            return comp_bid
        return None
    
    def _calculate_opening_ask(
        self,
        fair_value: float,
        ci_width: float,
        market_ask: float,
        position: int,
        seconds_remaining: float
    ) -> float | None:
        """Calculate conservative ask for opening or increasing position."""
        half_spread = self._calculate_spread(ci_width, seconds_remaining)
        theo_ask = fair_value + half_spread
        
        # Just beat market by 1¢ (conservative)
        comp_ask = max(theo_ask, market_ask - 1.0)
        
        # Only quote if it beats market and is reasonable
        if comp_ask < market_ask and comp_ask <= 99 and comp_ask > fair_value:
            return comp_ask
        return None
    
    def _generate_reducing_quote_dynamic(
        self,
        position: int,
        fair_value: float,
        market_bid: float,
        market_ask: float,
        seconds_remaining: float,
        market_spread: float,
        ci_width: float
    ) -> dict:
        """
        Generate quote to reduce position when at limit using dynamic pricing.
        
        At max position, we're desperate to close - use aggressive dynamic pricing.
        """
        if position > 0:  # Long - only sell
            close_price = self._calculate_closing_price(
                fair_value, market_ask, position, seconds_remaining,
                market_spread, ci_width, 'sell'
            )
            return {
                'should_quote': True,
                'bid_price': None,
                'ask_price': close_price,
                'reason': f'Reduce long @ {close_price:.0f}¢ (at limit)'
            }
        else:  # Short - only buy
            close_price = self._calculate_closing_price(
                fair_value, market_bid, position, seconds_remaining,
                market_spread, ci_width, 'buy'
            )
            return {
                'should_quote': True,
                'bid_price': close_price,
                'ask_price': None,
                'reason': f'Reduce short @ {close_price:.0f}¢ (at limit)'
            }
