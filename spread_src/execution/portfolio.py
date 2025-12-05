"""
Portfolio state tracking for market making.

Tracks positions, exposure, P&L, and cash.
"""

from typing import Dict, Optional
from datetime import datetime
from spread_src.execution.trade_logger import TradeLogger


class Portfolio:
    """
    Tracks portfolio state for market maker.
    
    State:
    - Positions: {ticker: contracts} (+ long, - short)
    - Cash: Available capital
    - Exposure: Total $ at risk
    - P&L: Realized + unrealized
    """
    
    def __init__(self, initial_capital=20.0, db_path='data/nba_data.db'):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting cash in dollars
            db_path: Path to database for logging
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Positions: ticker -> contracts (+ long, - short)
        self.positions: Dict[str, int] = {}
        
        # Cost basis: ticker -> average cost per contract
        self.cost_basis: Dict[str, float] = {}
        
        # Realized P&L (from closed positions)
        self.realized_pnl = 0.0
        
        # Trade history
        self.trade_history = []
        
        # Database logger
        self.logger = TradeLogger(db_path)
        
        # Track trade_ids for fills
        self.trade_ids: Dict[str, int] = {}  # ticker -> trade_id
    
    def update_fill(self, ticker: str, side: str, price: float, size: int, timestamp=None, trade_id=None):
        """
        Update position and cash after fill.
        
        Args:
            ticker: Market ticker
            side: 'buy' or 'sell'
            price: Fill price in cents
            size: Contracts filled
            timestamp: Fill time (optional)
            trade_id: Trade ID from database (optional)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update position
        current_pos = self.positions.get(ticker, 0)
        
        if side == 'buy':
            new_pos = current_pos + size
            cash_flow = -(price / 100.0) * size  # You pay
        else:  # sell
            new_pos = current_pos - size
            cash_flow = (price / 100.0) * size  # You receive
        
        self.positions[ticker] = new_pos
        self.cash += cash_flow
        
        # Update cost basis
        self._update_cost_basis(ticker, side, price, size, current_pos)
        
        # Log fill to database
        if trade_id:
            self.logger.log_order_filled(
                trade_id=trade_id,
                fill_price=price,
                position_after=new_pos,
                timestamp=timestamp
            )
        
        # Log trade
        self.trade_history.append({
            'timestamp': timestamp,
            'ticker': ticker,
            'side': side,
            'price': price,
            'size': size,
            'position_after': new_pos,
            'cash_after': self.cash
        })
        
        print(f"Fill: {side.upper()} {size} {ticker} @ {price}¢")
        print(f"  Position: {current_pos} → {new_pos}")
        print(f"  Cash: ${self.cash:.2f}")
    
    def _update_cost_basis(self, ticker: str, side: str, price: float, size: int, old_pos: int):
        """Update average cost basis for position."""
        if ticker not in self.cost_basis:
            self.cost_basis[ticker] = 0.0
        
        old_basis = self.cost_basis[ticker]
        
        if side == 'buy':
            # Adding to long or reducing short
            if old_pos >= 0:  # Increasing long
                # Weighted average
                total_contracts = abs(old_pos) + size
                self.cost_basis[ticker] = (old_basis * abs(old_pos) + price * size) / total_contracts
            else:  # Reducing short
                # Keep old basis if still short, else reset
                if old_pos + size < 0:
                    pass  # Still short, keep basis
                else:
                    self.cost_basis[ticker] = price
        else:  # sell
            # Adding to short or reducing long
            if old_pos <= 0:  # Increasing short
                total_contracts = abs(old_pos) + size
                self.cost_basis[ticker] = (old_basis * abs(old_pos) + price * size) / total_contracts
            else:  # Reducing long
                if old_pos - size > 0:
                    pass  # Still long, keep basis
                else:
                    self.cost_basis[ticker] = price
    
    def settle_market(self, ticker: str, outcome: bool):
        """
        Settle market at expiration.
        
        Args:
            ticker: Market ticker
            outcome: True if YES won, False if NO won
        """
        if ticker not in self.positions or self.positions[ticker] == 0:
            return
        
        position = self.positions[ticker]
        
        if outcome:  # YES wins, pays $1
            payout = position * 1.0  # $1 per long contract
        else:  # NO wins, YES pays $0
            payout = 0.0
        
        # Realized P&L
        cost = self.cost_basis.get(ticker, 0.0) * abs(position)
        if position > 0:  # Long
            realized = payout - cost
        else:  # Short
            realized = cost - payout
        
        self.realized_pnl += realized
        self.cash += payout
        
        # Clear position
        self.positions[ticker] = 0
        
        print(f"\nSettled {ticker}: {'YES' if outcome else 'NO'}")
        print(f"  Position: {position} → 0")
        print(f"  Payout: ${payout:.2f}")
        print(f"  Realized P&L: ${realized:+.2f}")
    
    def get_exposure(self) -> float:
        """
        Calculate total exposure ($ at risk).
        
        For each position:
        - Long: cost is exposure
        - Short: (100 - cost) is exposure
        """
        exposure = 0.0
        
        for ticker, position in self.positions.items():
            if position == 0:
                continue
            
            cost = self.cost_basis.get(ticker, 50.0)  # Default 50¢ if unknown
            
            if position > 0:  # Long
                exposure += (cost / 100.0) * position
            else:  # Short
                exposure += ((100 - cost) / 100.0) * abs(position)
        
        return exposure
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate unrealized P&L based on current market prices.
        
        Args:
            current_prices: {ticker: mid_price} in cents
            
        Returns:
            Unrealized P&L in dollars
        """
        unrealized = 0.0
        
        for ticker, position in self.positions.items():
            if position == 0:
                continue
            
            current_price = current_prices.get(ticker, 50.0)  # Default mid
            cost = self.cost_basis.get(ticker, 50.0)
            
            if position > 0:  # Long
                unrealized += ((current_price - cost) / 100.0) * position
            else:  # Short
                unrealized += ((cost - current_price) / 100.0) * abs(position)
        
        return unrealized
    
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Total P&L = Realized + Unrealized."""
        return self.realized_pnl + self.get_unrealized_pnl(current_prices)
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Total equity = Cash + Unrealized P&L."""
        return self.cash + self.get_unrealized_pnl(current_prices)
    
    def can_trade(self, ticker: str, side: str, price: float, size: int) -> bool:
        """
        Check if we have enough capital for this trade.
        
        Args:
            ticker: Market ticker
            side: 'buy' or 'sell'
            price: Price in cents
            size: Contracts
            
        Returns:
            True if we can afford it
        """
        if side == 'buy':
            cost = (price / 100.0) * size
            return self.cash >= cost
        else:  # sell (margin requirement)
            margin = ((100 - price) / 100.0) * size
            return self.cash >= margin
    
    def get_position_summary(self) -> str:
        """Get formatted summary of positions."""
        lines = ["\n=== PORTFOLIO ==="]
        lines.append(f"Cash: ${self.cash:.2f}")
        lines.append(f"Exposure: ${self.get_exposure():.2f}")
        lines.append(f"Realized P&L: ${self.realized_pnl:+.2f}")
        lines.append(f"\nPositions:")
        
        if not self.positions or all(p == 0 for p in self.positions.values()):
            lines.append("  (none)")
        else:
            for ticker, pos in self.positions.items():
                if pos != 0:
                    cost = self.cost_basis.get(ticker, 0.0)
                    lines.append(f"  {ticker}: {pos:+d} @ {cost:.1f}¢")
        
        return "\n".join(lines)
    
    def get_available_capital(self) -> float:
        """
        Get available capital for new trades.
        Returns current cash (could be adjusted to account for margin requirements).
        """
        return self.cash
    
    def sync_positions(self, kalshi_client):
        """
        Sync portfolio with actual Kalshi positions.
        
        Uses fill history to calculate accurate cost basis.
        
        Args:
            kalshi_client: KalshiClient instance
        """
        print("\n📥 Syncing positions from Kalshi...")
        
        try:
            # Get all unsettled positions
            response = kalshi_client.get_positions(settlement_status='unsettled', limit=1000)
            market_positions = response.get('market_positions', [])
            
            if not market_positions:
                print("  ✓ No existing positions")
                return
            
            # Get fill history for accurate cost basis
            fills = kalshi_client.get_fills(limit=1000)
            
            # Group fills by ticker
            fills_by_ticker = {}
            for fill in fills:
                ticker = fill.get('ticker')
                if ticker not in fills_by_ticker:
                    fills_by_ticker[ticker] = []
                fills_by_ticker[ticker].append(fill)
            
            # Load each position
            for pos_data in market_positions:
                ticker = pos_data.get('ticker')
                position = pos_data.get('position', 0)  # Net position
                
                if position == 0:
                    continue
                
                # Calculate cost basis from actual fills
                if ticker in fills_by_ticker:
                    ticker_fills = fills_by_ticker[ticker]
                    
                    # Sort by timestamp
                    ticker_fills.sort(key=lambda x: x.get('created_time', ''))
                    
                    # Calculate weighted average cost
                    total_cost = 0
                    total_contracts = 0
                    
                    for fill in ticker_fills:
                        side = fill.get('side', '').lower()
                        price = fill.get('yes_price', 50)  # Price in cents
                        count = fill.get('count', 0)  # Number of contracts
                        
                        if side == 'yes':  # Bought YES
                            total_cost += price * count
                            total_contracts += count
                        elif side == 'no':  # Sold YES (short)
                            total_cost += price * count
                            total_contracts += count
                    
                    if total_contracts > 0:
                        avg_cost = total_cost / total_contracts
                    else:
                        avg_cost = 50.0
                else:
                    # No fill history, use position data estimate
                    total_cost = pos_data.get('total_cost', 0)
                    if position != 0:
                        avg_cost = total_cost / abs(position)
                    else:
                        avg_cost = 50.0
                
                # Update portfolio
                self.positions[ticker] = position
                self.cost_basis[ticker] = avg_cost
                
                print(f"  ✓ Loaded: {ticker[-20:]:>20} → {position:+3d} @ {avg_cost:>5.1f}¢")
            
            # Calculate exposure
            total_exposure = self.get_exposure()
            num_positions = len([p for p in self.positions.values() if p != 0])
            print(f"\n  📊 Positions: {num_positions} | Exposure: ${total_exposure:.2f}")
            
        except Exception as e:
            print(f"  ⚠️  Error syncing positions: {e}")
            print("  Continuing with empty portfolio...")
    
    def should_close_position(
        self,
        ticker: str,
        position: int,
        fair_value: float,
        market_bid: float,
        market_ask: float
    ) -> bool:
        """
        Determine if we should close a position.
        
        Args:
            ticker: Market ticker
            position: Current position (+long, -short)
            fair_value: Model fair value in cents
            market_bid: Current best bid
            market_ask: Current best ask
            
        Returns:
            True if we should close
        """
        if position == 0:
            return False
        
        # Close if we can lock in profit of 5¢+ per contract
        cost = self.cost_basis.get(ticker, 50.0)
        
        if position > 0:  # Long - would sell at bid
            edge = market_bid - cost
            return edge >= 5.0
        else:  # Short - would buy at ask
            edge = cost - market_ask
            return edge >= 5.0
