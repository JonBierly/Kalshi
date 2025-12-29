"""
Position sizing using Kelly criterion.

Calculates optimal position sizes based on edge, price, and bankroll.
"""



class PositionSizer:
    """Calculate optimal position sizes using Kelly criterion."""
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize position sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
        """
        self.kelly_fraction = kelly_fraction
    
    def calculate_size(
        self,
        fair_value: float,
        price: float,
        bankroll: float,
        action: str = 'buy',
        ci_lower: float = None,
        ci_upper: float = None,
        position: int = 0,
        min_size: int = 1,
        max_size: int = 10
    ) -> int:
        """
        Calculate Kelly-optimal position size using conservative CI bounds.
        
        Formula: f = (conservative_edge / price) × kelly_fraction
        
        For BUYS: conservative_edge = ci_lower - price (worst case: true prob is low)
        For SELLS: conservative_edge = price - ci_upper (worst case: true prob is high)
        
        Args:
            fair_value: Model prediction in cents (0-100)
            price: Order price in cents
            bankroll: Available capital in dollars
            action: 'buy' or 'sell'
            ci_lower: 90% CI lower bound (0-1), or None to use fair_value
            ci_upper: 90% CI upper bound (0-1), or None to use fair_value
            position: Current position
            min_size: Minimum contract size
            max_size: Maximum contract size
            
        Returns:
            Number of contracts (min_size to max_size)
        """
        # Use conservative CI bound based on action
        # BUY: worst case is true prob at ci_lower
        # SELL: worst case is true prob at ci_upper
        if action == 'buy':
            conservative_fair = (ci_lower * 100) if ci_lower is not None else fair_value
            edge = conservative_fair - price  # positive if we have edge
        else:  # sell
            conservative_fair = (ci_upper * 100) if ci_upper is not None else fair_value
            edge = price - conservative_fair  # positive if we have edge
        
        # Calculate CI width for uncertainty adjustment
        if ci_lower is not None and ci_upper is not None:
            ci_width = ci_upper - ci_lower
        else:
            ci_width = 0.15  # fallback default
        
        # No edge on conservative basis = min size
        if edge <= 0:
            return min_size
        
        # Avoid division issues
        if price < 1 or price > 99 or bankroll < 0.10:
            return min_size
        
        # Kelly percentage: f = (edge/100) / (price/100)
        kelly_pct = (edge / 100.0) / (price / 100.0)
        
        # Apply kelly fraction
        kelly_pct = self._adjust_for_uncertainty(kelly_pct, ci_width)
        
        # Convert to number of contracts
        dollars_per_contract = price / 100.0
        kelly_dollars = bankroll * kelly_pct
        kelly_size = kelly_dollars / dollars_per_contract
        
        # Round and apply bounds
        size = int(round(kelly_size))
        
        # Adjust for existing position
        size = self._adjust_for_position(size, position)
        
        # Hard bounds
        size = max(min_size, min(max_size, size))
        
        return size
    
    def _adjust_for_uncertainty(self, kelly_pct: float, ci_width: float) -> float:
        """
        Reduce Kelly percentage if model is uncertain.
        
        Args:
            kelly_pct: Raw Kelly percentage
            ci_width: Confidence interval width (0-1)
            
        Returns:
            Adjusted Kelly percentage
        """
        # Apply configured Kelly fraction
        adjusted = kelly_pct * self.kelly_fraction
        
        # Further reduce if very uncertain (wide CI > 25%)
        if ci_width > 0.25:
            adjusted *= 0.5  # Half-quarter-Kelly
        
        return adjusted
    
    def _adjust_for_position(self, size: int, position: int) -> int:
        """
        Reduce size if already have significant position.
        
        Args:
            size: Calculated Kelly size
            position: Current position
            
        Returns:
            Adjusted size
        """
        # Reduce if already positioned (> 3 contracts)
        if abs(position) > 3:
            size = max(1, size // 2)
        
        return size
