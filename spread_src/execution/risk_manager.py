"""
Risk management for market making.

Enforces all position and capital limits before order placement.
"""

from typing import Dict, Tuple, Optional


class RiskManager:
    """
    Pre-trade risk checks to prevent over-exposure.
    
    Hard limits:
    - Total portfolio exposure: $20
    - Per-game exposure: $5
    - Max contracts per order: 20
    """
    
    def __init__(self, max_total_exposure=20.0, max_game_exposure=5.0, max_order_size=20):
        """
        Initialize risk manager.
        
        Args:
            max_total_exposure: Maximum $ at risk across all positions
            max_game_exposure: Maximum $ at risk in single game
            max_order_size: Maximum contracts in single order
        """
        self.max_total_exposure = max_total_exposure
        self.max_game_exposure = max_game_exposure
        self.max_order_size = max_order_size
        
        # Absolute hard limits (with 5% buffer)
        self.ABSOLUTE_MAX_EXPOSURE = max_total_exposure * 1.05
        self.ABSOLUTE_MAX_GAME = max_game_exposure * 1.1
        self.ABSOLUTE_MAX_CONTRACTS = max_order_size
    
    def check_new_order(
        self,
        ticker: str,
        side: str,
        price: float,
        size: int,
        current_positions: Dict[str, int],
        current_exposure: float,
        game_tickers: Dict[str, list],  # game_id -> [tickers]
        portfolio=None  # Optional: for accurate exposure calculation
    ) -> Tuple[bool, str]:
        """
        Check if new order passes risk limits.
        
        Args:
            ticker: Market ticker
            side: 'buy' or 'sell'
            price: Price in cents (1-99)
            size: Number of contracts
            current_positions: Current positions {ticker: contracts}
            current_exposure: Current total exposure in $
            game_tickers: Map of game_id to list of tickers
            portfolio: Portfolio object (optional, for accurate exposure)
            
        Returns:
            (approved: bool, reason: str)
        """
        # Check 1: Price bounds
        if price < 1 or price > 99:
            return False, f"Price {price}¢ out of bounds [1, 99]"
        
        # Check 2: Order size
        if size <= 0:
            return False, f"Invalid size {size}"
        if size > self.ABSOLUTE_MAX_CONTRACTS:
            return False, f"Size {size} exceeds max {self.ABSOLUTE_MAX_CONTRACTS}"
        
        # Check 3: Calculate new exposure
        order_cost = self._calculate_order_exposure(side, price, size)
        new_total_exposure = current_exposure + order_cost
        
        if new_total_exposure > self.ABSOLUTE_MAX_EXPOSURE:
            return False, f"Total exposure ${new_total_exposure:.2f} exceeds ${self.ABSOLUTE_MAX_EXPOSURE}"
        
        # Check 4: Game-level exposure
        game_id = self._extract_game_id(ticker)
        game_exposure = self._calculate_game_exposure(
            game_id, 
            ticker, 
            side, 
            price, 
            size,
            current_positions,
            game_tickers,
            portfolio
        )
        
        if game_exposure > self.ABSOLUTE_MAX_GAME:
            return False, f"Game exposure ${game_exposure:.2f} exceeds ${self.ABSOLUTE_MAX_GAME}"
        
        # Check 5: Position limit per market (for two-sided market making)
        current_pos = current_positions.get(ticker, 0)
        if side == 'buy':
            new_pos = current_pos + size
        else:
            new_pos = current_pos - size
        
        MAX_POSITION_PER_MARKET = 10  # Tight limit for market making
        
        if abs(new_pos) > MAX_POSITION_PER_MARKET:
            return False, f"Position would be {new_pos:+d}, exceeds limit ±{MAX_POSITION_PER_MARKET}"
        
        # All checks passed!
        return True, "OK"
    
    def _calculate_order_exposure(self, side: str, price: float, size: int) -> float:
        """
        Calculate $ at risk for this order.
        
        For limit orders:
        - Buy: You pay price * size (max loss)
        - Sell: You receive price, but could lose (100 - price) * size
        """
        if side == 'buy':
            # Max loss: you paid this amount
            return (price / 100.0) * size
        else:  # sell
            # Max loss: contract pays $1, you got price
            return ((100 - price) / 100.0) * size
    
    def _extract_game_id(self, ticker: str) -> str:
        """Extract game identifier from ticker."""
        # Format: KXNBASPREAD-25DEC02WASPHI-WAS7
        # Game ID: 25DEC02WASPHI
        parts = ticker.split('-')
        if len(parts) >= 2:
            return parts[1][:13]  # YYMMMDDTEAMTEAM
        return ticker
    
    def _calculate_game_exposure(
        self,
        game_id: str,
        new_ticker: str,
        new_side: str,
        new_price: float,
        new_size: int,
        current_positions: Dict[str, int],
        game_tickers: Dict[str, list],
        portfolio=None
    ) -> float:
        """
        Calculate total exposure for this game including new order.
        """
        exposure = 0.0
        
        # Get all tickers for this game
        tickers_in_game = []
        for gid, tickers in game_tickers.items():
            if game_id in gid:
                tickers_in_game.extend(tickers)
        
        # Add exposure from existing positions
        for ticker in tickers_in_game:
            pos = current_positions.get(ticker, 0)
            if pos != 0:
                # Use actual cost basis if portfolio available
                if portfolio and ticker in portfolio.cost_basis:
                    cost = portfolio.cost_basis[ticker]
                    if pos > 0:  # Long position
                        exposure += (cost / 100.0) * pos
                    else:  # Short position
                        exposure += ((100 - cost) / 100.0) * abs(pos)
                else:
                    # Fallback: estimate exposure (use 50¢ as average price)
                    exposure += abs(pos) * 0.50
        
        # Add new order exposure
        exposure += self._calculate_order_exposure(new_side, new_price, new_size)
        
        print(f"  DEBUG: Game {game_id[:13]} exposure: ${exposure:.2f} (limit: ${self.ABSOLUTE_MAX_GAME:.2f})")
        
        return exposure
    
    def get_available_capital(self, current_exposure: float) -> float:
        """Get remaining capital available for trading."""
        return max(0, self.max_total_exposure - current_exposure)
    
    def get_max_order_size(self, price: float, current_exposure: float) -> int:
        """
        Calculate max contracts we can trade at this price.
        
        Args:
            price: Order price in cents
            current_exposure: Current total exposure
            
        Returns:
            Max contracts (limited by capital and size limits)
        """
        available = self.get_available_capital(current_exposure)
        
        # Max based on capital
        max_by_capital = int(available / (price / 100.0))
        
        # Max based on size limits
        max_by_limit = self.max_order_size
        
        return min(max_by_capital, max_by_limit)
