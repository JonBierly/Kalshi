"""
Market making strategy for spread markets.

Generates optimal bid/ask quotes based on model predictions and market conditions.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats


class SpreadMarketMaker:
    """
    Market making strategy generator.
    
    Strategy:
    1. Calculate fair value from model
    2. Set spread based on uncertainty
    3. Apply price improvement vs current market
    4. Adjust for inventory risk
    """
    
    def __init__(self, min_spread=2.0, max_spread=30.0):
        """
        Initialize market maker.
        
        Args:
            min_spread: Minimum spread width in cents
            max_spread: Maximum spread width in cents
        """
        self.min_spread = min_spread
        self.max_spread = max_spread
    
    def calculate_quotes(
        self,
        model_mean: float,
        model_std: float,
        threshold: float,
        market_bid: float,
        market_ask: float,
        position: int = 0,
        seconds_remaining: float = 2880,
    ) -> Tuple[Optional[float], Optional[float], int]:
        """
        Calculate optimal bid/ask quotes.
        
        Args:
            model_mean: Predicted score differential (home - away)
            model_std: Uncertainty (standard deviation)
            threshold: Spread threshold (e.g., 10.5 for ">10.5")
            market_bid: Current market bid in cents
            market_ask: Current market ask in cents
            position: Current position in this market (+ long, - short)
            seconds_remaining: Time left in game
            
        Returns:
            (bid, ask, size) or (None, None, 0) if shouldn't quote
        """
        # 1. Calculate fair value from model
        fair_value = self._calculate_fair_value(model_mean, model_std, threshold)
        
        # 2. Calculate theoretical spread
        theo_spread = self._calculate_theoretical_spread(
            model_std, 
            position, 
            seconds_remaining
        )
        
        # 3. Calculate competitive quotes (price improvement)
        comp_bid, comp_ask = self._calculate_competitive_quotes(
            fair_value,
            theo_spread,
            market_bid,
            market_ask
        )
        
        # 4. Sanity checks
        if not self._should_quote(comp_bid, comp_ask, market_bid, market_ask):
            return None, None, 0
        
        # 5. Calculate size (Kelly sizing with limits)
        size = self._calculate_size(fair_value, model_std, position)
        
        return comp_bid, comp_ask, size
    
    def _calculate_fair_value(self, mean: float, std: float, threshold: float) -> float:
        """
        Calculate P(diff > threshold) based on model distribution.
        
        Args:
            mean: Predicted score differential
            std: Standard deviation
            threshold: Spread threshold
            
        Returns:
            Probability in cents (0-100)
        """
        # Probability that home team wins by > threshold
        dist = stats.norm(loc=mean, scale=std)
        prob = 1 - dist.cdf(threshold)
        
        return prob * 100  # Convert to cents
    
    def _calculate_theoretical_spread(
        self,
        model_std: float,
        position: int,
        seconds_remaining: float
    ) -> float:
        """
        Calculate theoretical spread width.
        
        Components:
        - Base spread (min profit)
        - Uncertainty adjustment
        - Inventory adjustment
        - Time decay
        """
        # Base spread
        base = self.min_spread
        
        # Uncertainty component (wider when uncertain)
        uncertainty = model_std * 100 * 1.5  # 1.5x std in cents
        
        # Inventory risk (wider when skewed)
        inventory = abs(position) * 0.5  # 0.5¢ per contract
        
        # Time decay (tighter near end)
        if seconds_remaining < 120:  # Last 2 minutes
            time_factor = 0.5
        elif seconds_remaining < 600:  # Last 10 minutes
            time_factor = 0.75
        else:
            time_factor = 1.0
        
        spread = (base + uncertainty + inventory) * time_factor
        
        # Clamp to reasonable bounds
        spread = max(self.min_spread, min(self.max_spread, spread))
        
        return spread
    
    def _calculate_competitive_quotes(
        self,
        fair_value: float,
        theo_spread: float,
        market_bid: float,
        market_ask: float
    ) -> Tuple[float, float]:
        """
        Calculate quotes with price improvement.
        
        Strategy:
        - Start with theoretical spread around fair value
        - Apply price improvement (just beat market)
        - Take whichever gives wider spread (more profit)
        """
        # Theoretical quotes
        theo_bid = fair_value - theo_spread / 2
        theo_ask = fair_value + theo_spread / 2
        
        # Competitive quotes (beat market by 1¢)
        tick_size = 1
        comp_bid = market_bid + tick_size
        comp_ask = market_ask - tick_size
        
        # Choose the BETTER pricing (wider spread = more profit)
        # But ensure we're still competitive
        
        # For BID: Use lower (more conservative)
        final_bid = min(theo_bid, comp_bid)
        # But ensure we beat market
        if final_bid <= market_bid:
            final_bid = market_bid + tick_size
        
        # For ASK: Use higher (more profitable)
        final_ask = max(theo_ask, comp_ask)
        # But ensure we beat market
        if final_ask >= market_ask:
            final_ask = market_ask - tick_size
        
        # Final check: ensure spread is reasonable
        actual_spread = final_ask - final_bid
        if actual_spread < theo_spread:
            # Being too aggressive, revert to theoretical
            final_bid = fair_value - theo_spread / 2
            final_ask = fair_value + theo_spread / 2
        
        # Clamp to valid range
        final_bid = max(1, min(99, final_bid))
        final_ask = max(1, min(99, final_ask))
        
        return final_bid, final_ask
    
    def _should_quote(
        self,
        bid: float,
        ask: float,
        market_bid: float,
        market_ask: float
    ) -> bool:
        """
        Check if we should post these quotes.
        
        Reasons to skip:
        - Quotes don't beat market
        - Spread is inverted
        - Prices out of bounds
        """
        # Check bounds
        if bid < 1 or bid > 99 or ask < 1 or ask > 99:
            return False
        
        # Check spread isn't inverted
        if ask <= bid:
            return False
        
        # Check we're actually competitive
        if bid <= market_bid and ask >= market_ask:
            # We're not better on either side
            return False
        
        # Check spread isn't too wide
        if ask - bid > 50:  # More than 50¢ spread seems unreasonable
            return False
        
        return True
    
    def _calculate_size(
        self,
        fair_value: float,
        model_std: float,
        position: int
    ) -> int:
        """
        Calculate order size using Kelly criterion with limits.
        
        Args:
            fair_value: Model probability (cents)
            model_std: Model uncertainty
            position: Current position
            
        Returns:
            Number of contracts (conservative)
        """
        # Base size (conservative)
        base_size = 10  # Default 10 contracts
        
        # Reduce if uncertain
        if model_std > 10:  # High uncertainty
            base_size = 5
        elif model_std < 5:  # Very certain
            base_size = 15
        
        # Reduce if we have inventory
        if abs(position) > 10:
            base_size = max(1, base_size // 2)
        
        return int(base_size)
    
    def calculate_ev(
        self,
        fair_value: float,
        price: float,
        side: str
    ) -> float:
        """
        Calculate expected value of a trade.
        
        Args:
            fair_value: Model probability (cents)
            price: Trade price (cents)
            side: 'buy' or 'sell'
            
        Returns:
            Expected value in cents per contract
        """
        if side == 'buy':
            # EV = (prob × $1) - price
            return (fair_value / 100) * 100 - price
        else:  # sell
            # EV = price - ((1 - prob) × $1)
            return price - (1 - fair_value / 100) * 100
