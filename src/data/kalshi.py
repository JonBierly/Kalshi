import requests
import json
import time
import base64
from datetime import datetime
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

class KalshiClient:
    def __init__(self, key_id, key_file_path='key.key', environment='prod'):
        self.key_id = key_id
        self.key_file_path = key_file_path
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2" if environment == 'prod' else "https://demo-api.kalshi.co/trade-api/v2"
        
        # Load Private Key
        try:
            with open(self.key_file_path, "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
        except Exception as e:
            print(f"Error loading private key: {e}")
            self.private_key = None

    def _get_headers(self, method, path):
        """Generates headers with signature for API Key auth."""
        if not self.private_key:
            return {}
            
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method}{path}".encode('utf-8')
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        sig_b64 = base64.b64encode(signature).decode('utf-8')
        
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

    def login(self):
        """No explicit login needed for API Key auth, but we can verify connection."""
        # We can try to fetch balance or something simple to verify
        return True

    def get_nba_markets(self):
        """Fetches active NBA markets."""
        # Search for daily NBA games series
        endpoint = "/events"
        params = {
            "series_ticker": "KXNBAGAME",
            "status": "open",
            "limit": 100
        }
        
        path = "/trade-api/v2/events"
        headers = self._get_headers("GET", path)
        
        try:
            print(f"Fetching events from: {self.base_url}{endpoint}")
            response = requests.get(f"{self.base_url}{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            events = data.get('events', [])
            print(f"DEBUG: Fetched {len(events)} KXNBAGAME events.")
            return events
        except Exception as e:
            print(f"Error fetching NBA markets: {e}")
            if 'response' in locals():
                print(response.text)
            return []

    def get_event_markets(self, event_ticker):
        """Fetches markets for a specific event."""
        endpoint = "/markets"
        params = {"event_ticker": event_ticker}
        
        # For GET with params, path is just /trade-api/v2/markets
        path = "/trade-api/v2/markets"
        headers = self._get_headers("GET", path)
        
        try:
            print(f"DEBUG: Fetching markets for event {event_ticker}")
            response = requests.get(f"{self.base_url}{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            markets = data.get('markets', [])
            print(f"DEBUG: Found {len(markets)} markets for {event_ticker}")
            return markets
        except Exception as e:
            print(f"Error fetching markets for event {event_ticker}: {e}")
            return []

    def get_market_details(self, ticker):
        """Get specific market details including current price."""
        endpoint = f"/markets/{ticker}"
        path = f"/trade-api/v2/markets/{ticker}"
        headers = self._get_headers("GET", path)
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json().get('market')
        except Exception as e:
            print(f"Error fetching market {ticker}: {e}")
            return None
            
    def get_orderbook(self, ticker):
        """Get orderbook for a market."""
        endpoint = f"/markets/{ticker}/orderbook"
        path = f"/trade-api/v2/markets/{ticker}/orderbook"
        headers = self._get_headers("GET", path)
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json().get('orderbook')
        except Exception as e:
            print(f"Error fetching orderbook for {ticker}: {e}")
            return None
    
    # ==================== ORDER MANAGEMENT ====================
    
    def place_order(self, ticker, side, action, count, price=None, order_type='limit'):
        """
        Place an order on Kalshi.
        
        Args:
            ticker: Market ticker (e.g., 'KXNBASPREAD-25DEC02OKCGSW-OKC5')
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts (must be >= 1)
            price: Price in cents (1-99) for limit orders
            order_type: 'limit' or 'market'
            
        Returns:
            Order dict with order_id, or None if failed
        """
        endpoint = "/portfolio/orders"
        path = "/trade-api/v2/portfolio/orders"
        
        # Build request body
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type
        }
        
        # Add price for limit orders
        if order_type == 'limit' and price is not None:
            if side == 'yes':
                body['yes_price'] = int(price)
            else:
                body['no_price'] = int(price)
        
        headers = self._get_headers("POST", path)
        
        try:
            print(f"Placing order: {action.upper()} {count} {ticker} {side.upper()} @ {price}¢")
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            order_data = response.json().get('order', {})
            print(f"  ✓ Order placed: {order_data.get('order_id')}")
            return order_data
        except Exception as e:
            print(f"  ✗ Error placing order: {e}")
            if 'response' in locals():
                try:
                    print(f"  Response: {response.text}")
                except:
                    pass
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Canceled order dict, or None if failed
        """
        endpoint = f"/portfolio/orders/{order_id}"
        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        headers = self._get_headers("DELETE", path)
        
        try:
            print(f"Canceling order: {order_id}")
            response = requests.delete(f"{self.base_url}{endpoint}", headers=headers)
            response.raise_for_status()
            order_data = response.json().get('order', {})
            print(f"  ✓ Order canceled")
            return order_data
        except Exception as e:
            print(f"  ✗ Error canceling order: {e}")
            return None
    
    def amend_order(self, order_id, ticker, side, action, new_price=None, new_count=None):
        """
        Amend an existing order (change price and/or size).
        
        Args:
            order_id: Order ID to amend
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            new_price: New price in cents (optional)
            new_count: New contract count (optional)
            
        Returns:
            Amended order dict, or None if failed
        """
        endpoint = f"/portfolio/orders/{order_id}/amend"
        path = f"/trade-api/v2/portfolio/orders/{order_id}/amend"
        
        body = {
            "ticker": ticker,
            "side": side,
            "action": action
        }
        
        if new_price is not None:
            if side == 'yes':
                body['yes_price'] = int(new_price)
            else:
                body['no_price'] = int(new_price)
        
        if new_count is not None:
            body['count'] = new_count
        
        headers = self._get_headers("POST", path)
        
        try:
            print(f"Amending order {order_id}: price={new_price}, count={new_count}")
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            order_data = response.json().get('order', {})
            print(f"  ✓ Order amended")
            return order_data
        except Exception as e:
            print(f"  ✗ Error amending order: {e}")
            if 'response' in locals():
                try:
                    print(f"  Response: {response.text}")
                except:
                    pass
            return None
    
    def get_orders(self, status=None, ticker=None, limit=100):
        """
        Get your orders.
        
        Args:
            status: Filter by status ('resting', 'canceled', 'executed')
            ticker: Filter by market ticker
            limit: Max results (1-200, default 100)
            
        Returns:
            List of order dicts
        """
        endpoint = "/portfolio/orders"
        path = "/trade-api/v2/portfolio/orders"
        
        params = {"limit": limit}
        if status:
            params['status'] = status
        if ticker:
            params['ticker'] = ticker
        
        headers = self._get_headers("GET", path)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json().get('orders', [])
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []
    
    def get_fills(self, ticker=None, order_id=None, min_ts=None, max_ts=None, limit=100):
        """
        Get fill history (executed trades).
        
        Args:
            ticker: Filter by market ticker
            order_id: Filter by order ID
            min_ts: Filter after this Unix timestamp
            max_ts: Filter before this Unix timestamp
            limit: Max results (1-200, default 100)
            
        Returns:
            List of fill dicts
        """
        endpoint = "/portfolio/fills"
        path = "/trade-api/v2/portfolio/fills"
        
        params = {"limit": limit}
        if ticker:
            params['ticker'] = ticker
        if order_id:
            params['order_id'] = order_id
        if min_ts:
            params['min_ts'] = min_ts
        if max_ts:
            params['max_ts'] = max_ts
        
        headers = self._get_headers("GET", path)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json().get('fills', [])
        except Exception as e:
            print(f"Error fetching fills: {e}")
            return []
    
    def get_positions(self, ticker=None, settlement_status='unsettled', limit=100):
        """
        Get current positions.
        
        Args:
            ticker: Filter by market ticker
            settlement_status: 'all', 'unsettled', or 'settled'
            limit: Max results (1-1000, default 100)
            
        Returns:
            Dict with 'market_positions' and 'event_positions'
        """
        endpoint = "/portfolio/positions"
        path = "/trade-api/v2/portfolio/positions"
        
        params = {
            "limit": limit,
            "settlement_status": settlement_status
        }
        if ticker:
            params['ticker'] = ticker
        
        headers = self._get_headers("GET", path)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
            return {
                'market_positions': data.get('market_positions', []),
                'event_positions': data.get('event_positions', [])
            }
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {'market_positions': [], 'event_positions': []}
