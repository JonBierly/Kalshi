"""
Portfolio state tracking for market making.

Tracks positions, exposure, P&L, and cash.
"""

from typing import Dict, Optional
from datetime import datetime
from spread_src.execution.trade_logger import TradeLogger


class Portfolio:
    """
    Stateless portfolio query layer for market maker.
    
    Fetches all data from Kalshi on every query (no RAM caching).
    This ensures data is always fresh and never stale.
    """
    
    def __init__(self, max_exposure=20.0, db_path='data/nba_data.db'):
        """
        Initialize portfolio configuration.
        
        Args:
            max_exposure: Maximum exposure limit in dollars (risk limit)
            db_path: Path to database for logging
        """
        self.max_exposure = max_exposure
        self.logger = TradeLogger(db_path)
        
        # Temporary cache (refreshed each iteration from Kalshi)
        self.positions: Dict[str, int] = {}
        self.cost_basis: Dict[str, float] = {}
        self.cash = None
        self.realized_pnl = 0.0
        
        # Trade history (for logging)
        self.trade_history = []
    
    def get_live_state(self, kalshi_client):
        """
        Fetch current portfolio state from Kalshi (no caching).
        
        Returns fresh data on every call to avoid stale state.
        
        Args:
            kalshi_client: KalshiClient instance
            
        Returns:
            {
                'balance': float,  # Current cash balance
                'positions': {ticker: count},  # Net positions
                'cost_basis': {ticker: avg_price},  # Cost per contract
                'exposure': float,  # Total $ at risk
                'available_capital': float  # Balance - margin
            }
        """
        # Fetch balance
        balance_data = kalshi_client.get_balance()
        balance = balance_data['balance'] if balance_data else self.max_exposure
        
        # Fetch unsettled positions
        response = kalshi_client.get_positions(settlement_status='unsettled', limit=1000)
        market_positions = response.get('market_positions', [])
        
        positions = {}
        cost_basis = {}
        
        # Get fill history for cost basis calculation
        fills_response = kalshi_client.get_fills(limit=1000)
        fills = fills_response if isinstance(fills_response, list) else []
        
        # Group fills by ticker
        fills_by_ticker = {}
        for fill in fills:
            ticker = fill.get('ticker')
            if ticker:
                if ticker not in fills_by_ticker:
                    fills_by_ticker[ticker] = []
                fills_by_ticker[ticker].append(fill)
        
        # Parse positions and calculate cost basis
        for pos_data in market_positions:
            ticker = pos_data.get('ticker')
            position = pos_data.get('position', 0)
            
            if position == 0:
                continue
                
            positions[ticker] = position
            
            # Try to calculate cost basis from fills (most accurate)
            if ticker in fills_by_ticker:
                ticker_fills = fills_by_ticker[ticker]
                ticker_fills.sort(key=lambda x: x.get('created_time', ''))
                
                # Calculate weighted average cost
                total_cost = 0
                total_contracts = 0
                
                for fill in ticker_fills:
                    side = fill.get('side', '').lower()
                    price = fill.get('yes_price', 50)  # Price in cents
                    count = fill.get('count', 0)  # Number of contracts
                    
                    if side in ['yes', 'no']:
                        total_cost += price * count
                        total_contracts += count
                
                if total_contracts > 0:
                    cost_basis[ticker] = total_cost / total_contracts
                else:
                    cost_basis[ticker] = 50.0
            else:
                # Fallback: try total_cost from position data
                total_cost_cents = pos_data.get('total_cost', 0)
                if total_cost_cents != 0:
                    cost_basis[ticker] = abs(total_cost_cents) / abs(position)
                else:
                    cost_basis[ticker] = 50.0
        
        # Calculate exposure
        exposure = self._calculate_exposure_from_state(positions, cost_basis)
        
        # Calculate available capital (balance - margin held for shorts)
        margin_held = 0.0
        for ticker, pos in positions.items():
            if pos < 0:  # Short position
                margin_held += abs(pos) * 1.00  # $1 per contract margin
        
        available = balance - margin_held
        
        return {
            'balance': balance,
            'positions': positions,
            'cost_basis': cost_basis,
            'exposure': exposure,
            'available_capital': max(0.0, available)
        }
    
    def _calculate_exposure_from_state(self, positions: dict, cost_basis: dict) -> float:
        """
        Calculate total exposure from position state.
        
        Args:
            positions: {ticker: count}
            cost_basis: {ticker: avg_price_cents}
            
        Returns:
            Total exposure in dollars
        """
        exposure = 0.0
        
        for ticker, position in positions.items():
            if position == 0:
                continue
            
            cost = cost_basis.get(ticker, 50.0)
            
            if position > 0:  # Long
                exposure += (cost / 100.0) * position
            else:  # Short
                exposure += ((100 - cost) / 100.0) * abs(position)
        
        return exposure
    
    def refresh_state(self, kalshi_client):
        """
        Refresh cached state from Kalshi.
        
        Call this at the START of each trading iteration to ensure
        all data is fresh. State is then cached for the iteration.
        
        Args:
            kalshi_client: KalshiClient instance
        """
        state = self.get_live_state(kalshi_client)
        
        # Update cache
        self.cash = state['balance']
        self.positions = state['positions']
        self.cost_basis = state['cost_basis']
        
        # Calculate realized P&L from Kalshi
        self.realized_pnl = self.get_realized_pnl(kalshi_client)
        
        print(f"\n💰 Refreshed state from Kalshi")
        print(f"  Cash: ${self.cash:.2f}")
        print(f"  Positions: {len(self.positions)}")
        print(f"  Exposure: ${state['exposure']:.2f}")
        print(f"  Realized P&L: ${self.realized_pnl:+.2f}")
    
    def get_realized_pnl(self, kalshi_client, since_timestamp=None):
        """
        Calculate realized P&L from Kalshi settled positions.
        
        Args:
            kalshi_client: KalshiClient instance
            since_timestamp: Optional start time (default: today 4am)
            
        Returns:
            Total realized P&L in dollars
        """
        if not since_timestamp:
            # Default to today 4am
            from datetime import datetime, timedelta
            now = datetime.now()
            if now.hour < 4:
                start = (now - timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
            else:
                start = now.replace(hour=4, minute=0, second=0, microsecond=0)
            since_timestamp = int(start.timestamp() * 1000)  # Kalshi uses milliseconds
        
        try:
            # Fetch settled positions since timestamp
            response = kalshi_client.get_positions(
                settlement_status='settled',
                limit=1000
            )
            
            total_pnl = 0.0
            for pos in response.get('market_positions', []):
                # Check if within time range
                settled_time = pos.get('settled_time')
                # For now, include all settled positions
                # TODO: filter by timestamp when API supports it
                
                # Kalshi provides realized_pnl in cents
                real_pnl = pos.get('realized_pnl', 0)  # In cents
                total_pnl += real_pnl / 100.0  # Convert to dollars
            
            return total_pnl
        except Exception as e:
            print(f"  ⚠️  Error calculating realized P&L: {e}")
            return 0.0
    
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
    
    def settle_unsettled_trades(self, kalshi_client, trade_logger):
        """
        Check Kalshi for finalized markets and settle all unsettled trades.
        
        This method:
        1. Queries database for all filled but unsettled trades
        2. Checks each market's status via Kalshi API
        3. Calculates P&L with Kalshi fees for finalized markets
        4. Updates database and portfolio state
        
        Args:
            kalshi_client: KalshiClient instance for API calls
            trade_logger: TradeLogger instance for database updates
            
        Returns:
            Number of trades settled
        """
        import sqlite3
        from spread_src.utils.kalshi_fees import calculate_kalshi_fee
        
        # Get unsettled trades from database
        conn = sqlite3.connect(self.logger.db_path) # Changed self.trade_logger to self.logger
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trade_id, ticker, side, fill_price, size
            FROM trades
            WHERE status = 'filled'
            AND closed_at IS NULL
            ORDER BY trade_id
        """)
        
        unsettled_trades = cursor.fetchall()
        conn.close()
        
        if not unsettled_trades:
            return 0  # Nothing to settle
        
        print(f"\n🔍 Checking {len(unsettled_trades)} unsettled trades...")
        
        settled_count = 0
        
        for trade_id, ticker, side, fill_price, size in unsettled_trades:
            try:
                # Query Kalshi for market status
                market = kalshi_client.get_market_details(ticker)
                
                if not market:
                    continue
                
                status = market.get('status')
                
                # Kalshi uses 'finalized' for settled markets
                if status in ['finalized', 'settled']:
                    result = market.get('result')  # 'yes' or 'no'
                    
                    if not result:
                        continue
                    
                    is_yes_result = (result.lower() == 'yes')
                    
                    # Calculate P&L for THIS specific trade
                    if side == 'buy':
                        # Bought YES contracts
                        cost = (fill_price / 100.0) * size
                        payout = 1.0 * size if is_yes_result else 0.0
                        pnl_before_fee = payout - cost
                    elif side == 'sell':
                        # Sold YES contracts (went short)
                        revenue = (fill_price / 100.0) * size
                        cost = 1.0 * size if is_yes_result else 0.0
                        pnl_before_fee = revenue - cost
                    else:
                        pnl_before_fee = 0.0
                    
                    # Deduct Kalshi fee
                    fee = calculate_kalshi_fee(fill_price, size)
                    realized_pnl = pnl_before_fee - fee
                    
                    # Update database
                    self.logger.log_position_closed( # Changed trade_logger to self.logger
                        trade_id=trade_id,
                        realized_pnl=realized_pnl
                    )
                    
                    # Update portfolio positions
                    # This logic assumes that each 'trade' in the database represents a single fill
                    # and that closing a position means reversing the effect of that fill on the portfolio.
                    # This is a simplified approach for settling individual trades.
                    # The `settle_market` method handles full market settlement and clearing the entire position.
                    # For `settle_unsettled_trades`, we are just marking individual fills as closed.
                    # The portfolio's `positions` and `cost_basis` are primarily updated by `update_fill`
                    # and `sync_positions`. This method doesn't directly modify `self.positions` or `self.cost_basis`
                    # in a way that would reflect the *current* state of the portfolio after settlement,
                    # but rather marks the *trade* as settled.
                    # The provided code snippet for updating portfolio positions here seems to be
                    # attempting to reverse the effect of the trade on the portfolio's net position,
                    # which might be redundant or conflict with `sync_positions` if not carefully managed.
                    # For now, I'll keep the provided logic for `self.positions` and `self.cost_basis` updates.
                    if ticker in self.positions:
                        current_pos = self.positions[ticker]
                        # Adjust position based on this trade
                        if side == 'buy':
                            self.positions[ticker] = current_pos - size
                        else:
                            self.positions[ticker] = current_pos + size
                        
                        # If position is now 0, remove it
                        if self.positions[ticker] == 0:
                            del self.positions[ticker]
                            if ticker in self.cost_basis:
                                del self.cost_basis[ticker]
                    
                    self.realized_pnl += realized_pnl # Add realized P&L to portfolio total
                    self.cash += realized_pnl # Adjust cash for realized P&L
                    
                    settled_count += 1
                    
                    print(f"  🏁 {ticker[-10:]}: {size} @ {fill_price:.1f}¢ → ${realized_pnl:+.2f}")
            
            except Exception as e:
                print(f"  ⚠️  Error settling trade {trade_id} for {ticker}: {e}") # Added error message
                continue
        
        if settled_count > 0:
            print(f"✅ Settled {settled_count} trades")
        
        return settled_count
    
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
                    # Extract market name from ticker (e.g., KXNBASPREAD-25DEC07ORLNYK-NYK2 -> NYK2)
                    market_name = ticker.split('-')[-1] if '-' in ticker else ticker[-10:]
                    lines.append(f"  {market_name}: {pos:+d} @ {cost:.1f}¢")
        
        return "\n".join(lines)
    
    def get_available_capital(self) -> float:
        """
        Get available capital for new trades.
        
        Accounts for margin held by Kalshi on short positions.
        When you sell YES contracts, Kalshi holds $1.00/contract as margin.
        """
        # Start with total cash
        available = self.cash
        
        # Subtract margin held for short positions
        # For each short position, Kalshi holds $1.00 per contract
        for ticker, position in self.positions.items():
            if position < 0:  # Short position
                margin_held = abs(position) * 1.00  # $1 per contract
                available -= margin_held
        
        return max(0.0, available)  # Can't be negative
    
    def sync_balance(self, kalshi_client):
        """
        Sync cash balance from Kalshi account.
        
        Args:
            kalshi_client: KalshiClient instance
        """
        print("\n💰 Syncing balance from Kalshi...")
        
        try:
            balance_data = kalshi_client.get_balance()
            if balance_data:
                self.cash = balance_data['balance']
                portfolio_value = balance_data['portfolio_value']
                print(f"  ✓ Balance: ${self.cash:.2f}")
                print(f"  ✓ Portfolio Value: ${portfolio_value:.2f}")
            else:
                print("  ⚠️  Failed to fetch balance, using max_exposure as fallback")
                self.cash = self.max_exposure
        except Exception as e:
            print(f"  ⚠️  Error syncing balance: {e}")
            print(f"  Using max_exposure (${self.max_exposure:.2f}) as fallback")
            self.cash = self.max_exposure
    
    def sync_positions(self, kalshi_client):
        """
        Sync portfolio with actual Kalshi positions.
        
        Uses fill history to calculate accurate cost basis.
        Note: Cash is synced separately via sync_balance().
        
        Args:
            kalshi_client: KalshiClient instance
        """
        print("\n📥 Syncing positions from Kalshi...")
        
        try:
            # Get all unsettled positions
            response = kalshi_client.get_positions(settlement_status='unsettled', limit=1000)
            market_positions = response.get('market_positions', [])
            
            # CRITICAL: Clear existing positions before syncing
            # This ensures settled positions are removed
            self.positions.clear()
            self.cost_basis.clear()
            
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
