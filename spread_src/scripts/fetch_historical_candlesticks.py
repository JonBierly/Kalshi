import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.kalshi import KalshiClient

def fetch_all_candlesticks():
    kalshi = KalshiClient("a40ff1c6-12ac-4a6c-9669-ffe12f3de235", "key.key")
    
    with open("data/historical_spread_tickers.json", "r") as f:
        tickers = json.load(f)
    
    os.makedirs("data/candlesticks", exist_ok=True)
    
    print(f"Fetching candlesticks for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        output_file = f"data/candlesticks/{ticker}.json"
        
        # Skip if already exists
        if os.path.exists(output_file):
            # print(f"  {i+1}/{len(tickers)} Skipping {ticker} (already exists)")
            continue
            
        print(f"  {i+1}/{len(tickers)} Fetching {ticker}...")
        
        # 1. Get market details to find open/close times
        market = kalshi.get_market_details(ticker)
        if not market:
            print(f"    Failed to get details for {ticker}")
            continue
            
        # Parse times: "2026-01-03T02:05:00Z"
        # Since they are strings, we can convert to timestamp
        def parse_kalshi_time(t_str):
            if not t_str: return None
            # Handle possible decimals in seconds
            if '.' in t_str:
                t_str = t_str.split('.')[0] + 'Z'
            return int(time.mktime(time.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")))

        start_ts = parse_kalshi_time(market.get('open_time'))
        # Use close_time or expected_expiration_time
        end_ts = parse_kalshi_time(market.get('close_time') or market.get('expected_expiration_time'))
        
        if not start_ts or not end_ts:
            print(f"    Missing open/close time for {ticker}")
            continue

        # 2. Fetch candlesticks in chunks of 5000 if needed
        # (Though most NBA markets are open for < 5000 minutes)
        all_candles = []
        current_start = start_ts
        while current_start < end_ts:
            chunk_end = min(current_start + 4500 * 60, end_ts)
            candles = kalshi.get_candlesticks(ticker, interval=1, start_ts=current_start, end_ts=chunk_end)
            if candles:
                all_candles.extend(candles)
                # Next start is the time of the last candle + 60s
                # Actually, Kalshi might include start/end. Let's just increment.
                current_start = chunk_end
            else:
                break
            time.sleep(0.2)
        
        if all_candles:
            with open(output_file, "w") as f:
                json.dump(all_candles, f)
            print(f"    Saved {len(all_candles)} candles.")
        else:
            print(f"    No candles found for {ticker}.")
            
        # Rate limiting - let's be responsible
        time.sleep(0.5)

if __name__ == "__main__":
    fetch_all_candlesticks()
