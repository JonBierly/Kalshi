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
