"""
Order manager for Kalshi market making.

Handles order placement, cancellation, and tracking.
"""

from typing import List, Dict, Optional
from datetime import datetime


class Order:
    """Represents a single order."""
    
    def __init__(self, order_id: str, ticker: str, side: str, price: float, size: int, status: str = 'pending'):
        self.order_id = order_id
        self.ticker = ticker
        self.side = side  # 'buy' or 'sell'
        self.price = price  # cents
        self.size = size  # contracts
        self.status = status  # 'pending', 'filled', 'canceled'
        self.submitted_at = datetime.now()
        self.filled_at = None
        self.filled_size = 0


class Fill:
    """Represents a filled order."""
    
    def __init__(self, ticker: str, side: str, price: float, size: int):
        self.ticker = ticker
        self.side = side
        self.price = price
        self.size = size
        self.timestamp = datetime.now()


class OrderManager:
    """
    Manages order lifecycle and Kalshi API interaction.
    
    Responsibilities:
    - Submit orders to Kalshi
    - Track order status  
    - Cancel unfilled orders
    - Detect fills
    """
    
    def __init__(self, kalshi_client, dry_run=True):
        """
        Initialize order manager.
        
        Args:
            kalshi_client: KalshiClient instance
            dry_run: If True, log orders without executing
        """
        self.kalshi = kalshi_client
        self.dry_run = dry_run
        
        # Active orders
        self.orders: Dict[str, Order] = {}
        
        # Fill history
        self.fills: List[Fill] = []
        
        # Order counter for dry-run
        self._order_counter = 0
    
    def place_limit_order(
        self,
        ticker: str,
        side: str,
        price: float,
        size: int
    ) -> Optional[str]:
        """
        Place limit order on Kalshi.
        
        Args:
            ticker: Market ticker
            side: 'buy' or 'sell'
            price: Price in cents
            size: Number of contracts
            
        Returns:
            order_id if successful, None otherwise
        """
        print(f"\n{'[DRY-RUN] ' if self.dry_run else ''}Placing order:")
        print(f"  {side.upper()} {size} {ticker} @ {price:.1f}¢")
        
        if self.dry_run:
            # Simulate order placement
            order_id = f"DRY_{self._order_counter}"
            self._order_counter += 1
        else:
            # Real Kalshi API call
            # Determine if we're buying or selling YES
            # side='buy' means BUY, side='sell' means SELL
            # For spread markets we're always trading YES side
            order_data = self.kalshi.place_order(
                ticker=ticker,
                side='yes',  # Trading YES side of spread
                action=side,  # 'buy' or 'sell'
                count=size,
                price=price,
                order_type='limit'
            )
            
            if not order_data:
                print("  [ERROR] Order placement failed")
                return None
            
            order_id = order_data.get('order_id')
        
        # Track order
        order = Order(order_id, ticker, side, price, size)
        self.orders[order_id] = order
        
        print(f"  Order ID: {order_id}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        if order_id not in self.orders:
            print(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != 'pending':
            print(f"Order {order_id} already {order.status}")
            return False
        
        print(f"{'[DRY-RUN] ' if self.dry_run else ''}Canceling order {order_id}")
        
        if not self.dry_run:
            # Real Kalshi API call
            self.kalshi.cancel_order(order_id)
        
        order.status = 'canceled'
        return True
    
    def amend_order(self, order_id: str, new_price: float = None, new_size: int = None) -> bool:
        """
        Amend an existing order (change price and/or size).
        
        Args:
            order_id: Order ID to amend
            new_price: New price in cents (optional)
            new_size: New size in contracts (optional)
            
        Returns:
            True if successful
        """
        if order_id not in self.orders:
            print(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != 'pending':
            print(f"Order {order_id} already {order.status}")
            return False
        
        print(f"{'[DRY-RUN] ' if self.dry_run else ''}Amending order {order_id}:")
        print(f"  Old: {order.side} {order.size} @ {order.price:.1f}¢")
        print(f"  New: {order.side} {new_size or order.size} @ {new_price or order.price:.1f}¢")
        
        if self.dry_run:
            # Simulate amendment
            if new_price:
                order.price = new_price
            if new_size:
                order.size = new_size
        else:
            # Real Kalshi API call
            # Only pass parameters we're actually changing
            amend_params = {
                'order_id': order_id,
                'ticker': order.ticker,
                'side': 'yes',
                'action': order.side
            }
            
            if new_price is not None:
                amend_params['new_price'] = new_price
            
            if new_size is not None:
                amend_params['new_count'] = new_size
            
            result = self.kalshi.amend_order(**amend_params)
            
            if not result:
                print("  [ERROR] Amendment failed")
                return False
            
            # Update tracked order
            if new_price:
                order.price = new_price
            if new_size:
                order.size = new_size
        
        return True
    
    def manage_orders(self, opportunities: List[Dict]) -> None:
        """
        Smart order management: keep good orders, amend if needed, cancel bad ones.
        
        Args:
            opportunities: List of current best opportunities
                Each: {'ticker', 'side', 'price', 'size', 'ev'}
        """
        pending_orders = self.get_open_orders()
        
        if not pending_orders:
            return
        
        print(f"\nManaging {len(pending_orders)} pending orders...")
        
        # Create lookup of opportunities by ticker+side
        opp_map = {}
        for opp in opportunities:
            key = (opp['ticker'], opp['side'])
            opp_map[key] = opp
        
        to_cancel = []
        
        for order in pending_orders:
            key = (order.ticker, order.side)
            
            # Check if this market+side still has an opportunity
            if key not in opp_map:
                # No longer a good opportunity - cancel
                print(f"  {order.order_id}: No longer profitable, canceling")
                to_cancel.append(order.order_id)
                continue
            
            opp = opp_map[key]
            
            # Check if price needs adjustment (>2¢ difference)
            price_diff = abs(opp['price'] - order.price)
            
            if price_diff > 2:
                # Amend to new price
                print(f"  {order.order_id}: Price changed by {price_diff:.1f}¢, amending")
                self.amend_order(order.order_id, new_price=opp['price'])
            else:
                # Price is still good, keep it
                print(f"  {order.order_id}: Still good, keeping")
        
        # Cancel orders that are no longer profitable
        for order_id in to_cancel:
            self.cancel_order(order_id)
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders.
        
        Returns:
            Number of orders canceled
        """
        pending = [oid for oid, o in self.orders.items() if o.status == 'pending']
        
        if not pending:
            return 0
        
        print(f"\n{'[DRY-RUN] ' if self.dry_run else ''}Canceling {len(pending)} pending orders...")
        
        for order_id in pending:
            self.cancel_order(order_id)
        
        return len(pending)
    
    def get_open_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [o for o in self.orders.values() if o.status == 'pending']
    
    def get_fills(self) -> List[Fill]:
        """Get fill history."""
        return self.fills
    
    def check_for_fills(self) -> List[Fill]:
        """
        Check for new fills on pending orders.
        
        In production, this would:
        1. Query Kalshi API for order status
        2. Detect any fills
        3. Update order status
        4. Return new fills
        
        Returns:
            List of new fills since last check
        """
        if self.dry_run:
            # In dry-run, simulate no fills
            return []
        
        # Real Kalshi API call - get recent fills
        new_fills = []
        
        try:
            # Get all recent fills
            fills_data = self.kalshi.get_fills(limit=50)
            
            for fill in fills_data:
                order_id = fill.get('order_id')
                
                # Check if this is one of our orders
                if order_id in self.orders:
                    order = self.orders[order_id]
                    
                    # Only process if not already marked as filled
                    if order.status != 'filled':
                        order.status = 'filled'
                        order.filled_at = datetime.now()
                        order.filled_size = fill.get('count', 0)
                        
                        # Create fill object
                        fill_obj = Fill(
                            ticker=fill.get('ticker'),
                            side=fill.get('action'),  # 'buy' or 'sell'
                            price=fill.get('yes_price', fill.get('no_price', 0)),
                            size=fill.get('count', 0)
                        )
                        
                        self.fills.append(fill_obj)
                        new_fills.append(fill_obj)
        except Exception as e:
            print(f"Error checking fills: {e}")
        
        return new_fills
    
    def get_order_summary(self) -> str:
        """Get formatted summary of orders."""
        lines = ["\n=== ORDERS ==="]
        
        pending = [o for o in self.orders.values() if o.status == 'pending']
        filled = [o for o in self.orders.values() if o.status == 'filled']
        canceled = [o for o in self.orders.values() if o.status == 'canceled']
        
        lines.append(f"Pending: {len(pending)}")
        for order in pending:
            lines.append(f"  {order.order_id}: {order.side} {order.size} {order.ticker} @ {order.price:.1f}¢")
        
        if filled:
            lines.append(f"\nFilled: {len(filled)}")
            for order in filled:
                lines.append(f"  {order.ticker}: {order.side} {order.size} @ {order.price:.1f}¢")
        
        if canceled:
            lines.append(f"\nCanceled: {len(canceled)}")
        
        return "\n".join(lines)
