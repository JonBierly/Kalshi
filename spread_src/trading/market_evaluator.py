"""
Market evaluation for generating trading opportunities.

Evaluates markets and generates bid/ask opportunities using model predictions.
"""

from typing import List, Dict, Optional
from scipy import stats as scipy_stats


class MarketEvaluator:
    """Evaluate markets and generate trading opportunities."""
    
    def __init__(self, spread_model, quote_generator, position_sizer):
        """
        Initialize market evaluator.
        
        Args:
            spread_model: SpreadDistributionModel for predictions
            quote_generator: QuoteGenerator for quote generation  
            position_sizer: PositionSizer for sizing calculations
        """
        self.spread_model = spread_model
        self.quote_generator = quote_generator
        self.position_sizer = position_sizer
    
    def create_opportunities_from_quotes(
        self,
        ticker: str,
        fair_value: float,
        quotes: dict,
        market_spread: float,
        position: int,
        bankroll: float,
        ci_width: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        seconds_remaining: Optional[int] = None
    ) -> List[Dict]:
        """
        Convert quotes into trading opportunities.
        
        Args:
            ticker: Market ticker
            fair_value: Model fair value in cents
            quotes: Quote dict from QuoteGenerator
            market_spread: Current market spread (for liquidity penalty)
            position: Current position
            bankroll: Available capital
            ci_width: Confidence interval width
            ci_lower: Lower CI bound (0-1)
            ci_upper: Upper CI bound (0-1)
            seconds_remaining: Seconds left in game
            
        Returns:
            List of opportunity dicts
        """
        if not quotes['should_quote']:
            return []
        
        opportunities = []
        
        # Calculate liquidity factor for EV adjustment
        liquidity_factor = self._calculate_liquidity_factor(market_spread)
        
        # Create BID opportunity
        if quotes['bid_price'] is not None:
            bid_size = self.position_sizer.calculate_size(
                fair_value=fair_value,
                price=quotes['bid_price'],
                ci_width=ci_width,
                bankroll=bankroll,
                position=position
            )
            
            # Skew size based on position
            if position > 0:  # Long - less eager to buy
                bid_size = max(1, bid_size // 2)
            
            base_ev = fair_value - quotes['bid_price']
            adjusted_ev = base_ev * liquidity_factor
            
            opportunities.append({
                'ticker': ticker,
                'side': 'buy',
                'price': quotes['bid_price'],
                'size': bid_size,
                'ev': adjusted_ev,
                'reason': f"MM Bid @ {quotes['bid_price']:.0f}¢",
                'is_mm_quote': True,
                # Model context for logging
                'model_fair': fair_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'market_spread': market_spread,
                'seconds_remaining': seconds_remaining
            })
        
        # Create ASK opportunity
        if quotes['ask_price'] is not None:
            ask_size = self.position_sizer.calculate_size(
                fair_value=fair_value,
                price=quotes['ask_price'],
                ci_width=ci_width,
                bankroll=bankroll,
                position=position
            )
            
            # Skew size based on position
            if position < 0:  # Short - less eager to sell
                ask_size = max(1, ask_size // 2)
            
            base_ev = quotes['ask_price'] - fair_value
            adjusted_ev = base_ev * liquidity_factor
            
            opportunities.append({
                'ticker': ticker,
                'side': 'sell',
                'price': quotes['ask_price'],
                'size': ask_size,
                'ev': adjusted_ev,
                'reason': f"MM Ask @ {quotes['ask_price']:.0f}¢",
                'is_mm_quote': True,
                # Model context for logging
                'model_fair': fair_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'market_spread': market_spread,
                'seconds_remaining': seconds_remaining
            })
        
        return opportunities
    
    def _calculate_liquidity_factor(self, market_spread: float) -> float:
        """
        Calculate liquidity penalty based on market spread width.
        
        Wide spreads = illiquid = lower priority
        Smooth penalty: 1.0 at 0¢ spread, 0.5 at 100¢+ spread
        
        Args:
            market_spread: Current bid-ask spread in cents
            
        Returns:
            Liquidity factor (0.5 to 1.0)
        """
        return max(0.5, 1.0 - (market_spread / 200.0))
