import os
import sys
import json
import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient

def debug_events():
    print("Initializing Kalshi Client...")
    try:
        kalshi = KalshiClient("3048039d-2104-4e20-801b-c7eb07519142", "key.key")
    except Exception as e:
        print(f"Failed to initialize KalshiClient: {e}")
        return

    print("\nFetching ONE NBA Spread Event...")
    endpoint = "/events"
    params = {
        "series_ticker": "KXNBASPREAD",
        "status": "open",
        "limit": 1
    }
    path = "/trade-api/v2/events"
    headers = kalshi._get_headers("GET", path)
    
    try:
        resp = requests.get(f"{kalshi.base_url}{endpoint}", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        events = data.get('events', [])
        
        if events:
            print("Event Structure:")
            print(json.dumps(events[0], indent=2))
        else:
            print("No events found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_events()
