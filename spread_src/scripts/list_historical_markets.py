import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.kalshi import KalshiClient

def list_historical_markets():
    # Use the same key as in fetch_kalshi_data.py
    kalshi = KalshiClient("a40ff1c6-12ac-4a6c-9669-ffe12f3de235", "key.key")
    
    # We want events for KXNBAGAME (which containment SPREAD markets)
    # or just search markets directly with series_ticker KXNBASPREAD
    
    series_ticker = "KXNBASPREAD"
    print(f"Searching for {series_ticker} markets...")
    
    # Try fetching events first to see if we can find the relevant ones
    endpoint = "/events"
    path = "/trade-api/v2/events"
    params = {
        "series_ticker": "KXNBASPREAD",
        "limit": 100
    }
    
    headers = kalshi._get_headers("GET", path)
    import requests
    response = requests.get(f"{kalshi.base_url}{endpoint}", headers=headers, params=params)
    events = response.json().get('events', [])
    if events:
        print(f"Sample event keys: {events[0].keys()}")
        print(f"Sample event: {events[0]}")
    
    print(f"Found {len(events)} KXNBAGAME events.")
    
    relevant_tickers = []
    
    # Filter events by date: Dec 29 (25DEC29) to Jan 3 (26JAN03)
    target_dates = ["25DEC29", "25DEC30", "25DEC31", "26JAN01", "26JAN02", "26JAN03"]
    
    for event in events:
        ticker = event.get('event_ticker', '')
        if any(d in ticker for d in target_dates):
            print(f"Processing event: {ticker}")
            # Get markets for this event
            markets = kalshi.get_event_markets(ticker)
            for m in markets:
                m_ticker = m.get('ticker', '')
                print(f"    Market ticker: {m_ticker}")
                if "SPREAD" in m_ticker:
                    relevant_tickers.append(m_ticker)
                    print(f"  Found spread market: {m_ticker}")

    with open("data/historical_spread_tickers.json", "w") as f:
        json.dump(relevant_tickers, f, indent=4)
    
    print(f"\nSaved {len(relevant_tickers)} spread tickers to data/historical_spread_tickers.json")

if __name__ == "__main__":
    list_historical_markets()
