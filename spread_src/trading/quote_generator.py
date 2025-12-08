"""
Quote generation for two-sided market making.

Handles spread calculation, inventory skewing, and price improvement.
"""


class QuoteGenerator:
    """Generate bid/ask quotes for market making."""
    
    def __init__(self, max_position: int = 5):
        """
        Initialize quote generator.
        
        Args:
            max_position: Maximum position per market (±)
        """
        self.max_position = max_position
    
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
        Generate two-sided quotes with inventory skewing.
        
        Strategy:
        1. Calculate spread width from confidence interval
        2. Post quotes around fair value
        3. Apply inventory skewing if positioned
        4. Ensure we beat existing market prices
        
        Args:
            fair_value: Model fair value in cents (0-100)
            ci_lower: Lower bound of 90% CI (0-1)
            ci_upper: Upper bound of 90% CI (0-1)
            market_bid: Current best bid in cents
            market_ask: Current best ask in cents
            position: Current net position
            seconds_remaining: Time left in game
            cost_basis: Average entry price for urgency calculation (optional)
            
        Returns:
            {
                'should_quote': bool,
                'bid_price': float or None,
                'ask_price': float or None,
                'reason': str
            }
        """
        # 1. Calculate spread width
        ci_width = ci_upper - ci_lower
        half_spread = self._calculate_spread(ci_width, seconds_remaining)
        
        # 2. Base quotes around fair value
        theo_bid = fair_value - half_spread
        theo_ask = fair_value + half_spread
        
        # 3. Calculate urgency premium for losing positions
        urgency = self._calculate_urgency_premium(position, cost_basis, fair_value)
        
        # 4. Apply inventory skewing with urgency
        theo_bid, theo_ask = self._apply_inventory_skew(theo_bid, theo_ask, position, urgency)
        
        # 5. Apply price improvement
        comp_bid, comp_ask = self._apply_price_improvement(
            theo_bid, theo_ask, market_bid, market_ask
        )
        
        # 5. Sanity checks
        sanity_check = self._validate_quotes(comp_bid, comp_ask, market_bid, market_ask)
        if not sanity_check['valid']:
            return {
                'should_quote': False,
                'bid_price': None,
                'ask_price': None,
                'reason': sanity_check['reason']
            }
        
        # Check position limits
        if abs(position) >= self.max_position:
            return self._generate_reducing_quote(position, comp_bid, comp_ask)
        
        return {
            'should_quote': True,
            'bid_price': comp_bid,
            'ask_price': comp_ask,
            'reason': f'Two-sided @ {comp_bid:.0f}-{comp_ask:.0f}¢'
        }
    
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
