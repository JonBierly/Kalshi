"""
Simple Edge Trader - +EV edge-based trading strategy.

Strategy:
1. Model predicts P(spread > threshold)
2. Compare to market bid/ask
3. If edge >= 4%, place order at beat-by-1 price
4. Cancel orders when edge drops or price uncompetitive
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class TradeOpportunity:
    """A trading opportunity with edge."""
    ticker: str
    side: str  # 'yes' or 'no'
    action: str  # 'buy' or 'sell'
    price: int  # in cents
    model_prob: float  # model's probability for YES
    market_price: float  # market price we're comparing to (in cents)
    edge: float  # model_prob - market_implied_prob
    ev_cents: float  # expected value in cents per contract
    
    # Optional fields for logging/analytics
    ci_lower: Optional[float] = None  # 90% CI lower bound (0-1)
    ci_upper: Optional[float] = None  # 90% CI upper bound (0-1)
    market_spread: Optional[float] = None  # bid-ask spread in cents
    game_id: Optional[str] = None  # game identifier
    seconds_remaining: Optional[float] = None  # time left in game
    
    def __repr__(self):
        return f"{self.ticker} {self.action.upper()} {self.side.upper()} @ {self.price}¢ (edge: {self.edge:.1%}, EV: {self.ev_cents:.1f}¢)"


class SimpleEdgeTrader:
    """
    Simple +EV trading strategy.
    
    - Places orders when edge >= min_edge
    - Beats current best price by 1 cent to get filled first
    - Cancels orders when edge drops below cancel_threshold
    """
    
    def __init__(self, min_edge: float = 0.04, cancel_threshold: float = 0.02):
        """
        Args:
            min_edge: Minimum edge to place order (default 4%)
            cancel_threshold: Edge below which to cancel order (default 2%)
        """
        self.min_edge = min_edge
        self.cancel_threshold = cancel_threshold
    
    def evaluate_market(
        self,
        ticker: str,
        model_prob: float,
        bid: Optional[int],
        ask: Optional[int],
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None
    ) -> List[TradeOpportunity]:
        """
        Evaluate a market for trading opportunities.
        
        Strategy: Beat-by-1 pricing with fair value edge calculation.
        - BUY at bid+1 if edge (fair - price) >= min_edge
        - SELL at ask-1 if edge (price - fair) >= min_edge
        
        Uses CI bounds conservatively: only trade if CI bound also shows edge.
        
        Args:
            ticker: Market ticker
            model_prob: Model's probability that YES wins (0-1)
            bid: Best bid price in cents
            ask: Best ask price in cents
            ci_lower: 90% CI lower bound (0-1)
            ci_upper: 90% CI upper bound (0-1)
            
        Returns:
            List of TradeOpportunity objects (0-2 opportunities)
        """
        opportunities = []
        
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return opportunities
        
        # Default CI to point estimate if not provided
        if ci_lower is None:
            ci_lower = model_prob
        if ci_upper is None:
            ci_upper = model_prob
        
        fair_cents = model_prob * 100
        ci_lower_cents = ci_lower * 100
        ci_upper_cents = ci_upper * 100
        
        # BUY opportunity: bid at (bid + 1)
        # Edge = what we think it's worth - what we pay
        # Conservative: use ci_lower to ensure even pessimistic estimate shows edge
        buy_price = bid + 1
        if buy_price < fair_cents:  # Only buy below fair value
            edge = (ci_lower_cents - buy_price) / 100  # Conservative edge
            ev = model_prob * (100 - buy_price) - (1 - model_prob) * buy_price
            
            if edge >= self.min_edge:
                opportunities.append(TradeOpportunity(
                    ticker=ticker,
                    side='yes',
                    action='buy',
                    price=buy_price,
                    model_prob=model_prob,
                    market_price=ask,
                    edge=edge,
                    ev_cents=ev,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    market_spread=ask - bid
                ))
        
        # SELL opportunity: offer at (ask - 1)
        # Edge = what we receive - what we think it's worth
        # Conservative: use ci_upper to ensure even optimistic estimate shows edge
        sell_price = ask - 1
        if sell_price > fair_cents:  # Only sell above fair value
            edge = (sell_price - ci_upper_cents) / 100  # Conservative edge
            ev = (1 - model_prob) * sell_price - model_prob * (100 - sell_price)
            
            if edge >= self.min_edge:
                opportunities.append(TradeOpportunity(
                    ticker=ticker,
                    side='yes',
                    action='sell',
                    price=sell_price,
                    model_prob=model_prob,
                    market_price=bid,
                    edge=edge,
                    ev_cents=ev,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    market_spread=ask - bid
                ))
        
        return opportunities
    
    def should_cancel_order(
        self,
        order_side: str,  # 'yes' or 'no'
        order_action: str,  # 'buy' or 'sell'
        order_price: int,
        model_prob: float,
        current_bid: Optional[int],
        current_ask: Optional[int]
    ) -> tuple[bool, str]:
        """
        Determine if an existing order should be cancelled.
        
        Returns:
            (should_cancel, reason)
        """
        # Calculate current edge
        if order_action == 'buy' and order_side == 'yes':
            # We're trying to buy YES
            current_edge = model_prob - (order_price / 100)
            if current_edge < self.cancel_threshold:
                return True, f"Edge dropped to {current_edge:.1%}"
            # Check if our price is still competitive (at or above best bid)
            if current_bid and order_price < current_bid:
                return True, f"Price {order_price}¢ below best bid {current_bid}¢"
                
        elif order_action == 'sell' and order_side == 'yes':
            # We're trying to sell YES
            current_edge = (order_price / 100) - model_prob
            if current_edge < self.cancel_threshold:
                return True, f"Edge dropped to {current_edge:.1%}"
            # Check if our price is still competitive (at or below best ask)
            if current_ask and order_price > current_ask:
                return True, f"Price {order_price}¢ above best ask {current_ask}¢"
        
        return False, ""
    
    def get_order_size(
        self,
        price: int,
        max_exposure: float,
        current_exposure: float
    ) -> int:
        """
        Calculate order size based on remaining exposure room.
        
        Args:
            price: Order price in cents
            max_exposure: Maximum allowed exposure in dollars
            current_exposure: Current exposure in dollars
            
        Returns:
            Number of contracts to order
        """
        remaining = max_exposure - current_exposure
        if remaining <= 0:
            return 0
        
        # Cost per contract = price / 100 dollars
        cost_per_contract = price / 100
        max_contracts = int(remaining / cost_per_contract)
        
        return max(0, min(max_contracts, 100))  # Cap at 100 contracts
