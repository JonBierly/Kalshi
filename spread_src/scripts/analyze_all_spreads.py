import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient

def analyze_spreads():
    # 1. Initialize Client
    print("Initializing Kalshi Client...")
    try:
        kalshi = KalshiClient("3048039d-2104-4e20-801b-c7eb07519142", "key.key")
    except Exception as e:
        print(f"Failed to initialize KalshiClient: {e}")
        return

    # 2. Fetch Events (NBA Spreads)
    print("\nFetching NBA Spread Events...")
    # Using 'KXNBASPREAD' series ticker based on `simple_live_trader.py` usage
    # Note: `simple_live_trader.py` constructs tickers like `KXNBASPREAD-{date}{away}{home}`
    # We'll search for events with series_ticker="KXNBASPREAD" if possible, or just "KXNBASPREAD"
    
    # Kalshi API usually takes series_ticker for events
    events = []
    try:
        # Fetching all events for the series
        response = kalshi.get_nba_markets() # This fetches KXNBAGAME, let's try a direct call for SPREADS if possible
        # Actually, get_nba_markets uses KXNBAGAME. We need KXNBASPREAD.
        # Let's use the underlying _get_headers and requests if needed, 
        # or just modify the call.
        # Actually, let's just use the client's generic get_events if it had one, 
        # but it doesn't. 
        # We'll reproduce the logic from `get_nba_markets` but for spreads.
        
        endpoint = "/events"
        params = {
            "series_ticker": "KXNBASPREAD",
            "status": "open",
            "limit": 100 # Fetch plenty
        }
        path = "/trade-api/v2/events"
        headers = kalshi._get_headers("GET", path)
        
        import requests
        resp = requests.get(f"{kalshi.base_url}{endpoint}", headers=headers, params=params)
        resp.raise_for_status()
        events = resp.json().get('events', [])
        print(f"Found {len(events)} spread events.")
        
    except Exception as e:
        print(f"Error fetching spread events: {e}")
        return

    if not events:
        print("No events found.")
        return

    # 3. Fetch Markets & Analyze
    market_data = []
    
    print(f"\nAnalyzing {len(events)} events...")
    for event in events:
        ticker = event['event_ticker']
        
        # Parse start time from ticker: KXNBASPREAD-26FEB19DENLAC
        # Format: KXNBASPREAD-YYMMMdd...
        try:
            # Extract date part
            parts = ticker.split('-')
            if len(parts) < 2:
                continue
                
            date_str = parts[1][:7] # e.g. 26FEB19
            # Parse date
            start_dt = datetime.strptime(date_str, "%y%b%d").replace(tzinfo=ZoneInfo("UTC"))
            
            # Set time to roughly 7 PM ET (00:00 UTC next day) as a placeholder since we don't have exact time
            # Actually, let's just use the date for "Days Out" calculation
            # But wait, NBA games are at night. 
            # If current time is Feb 17 10am, and game is Feb 17, it's 0 days out.
            
            now_dt = datetime.now(ZoneInfo("UTC"))
            
            # normalize to date only for diff
            days_to_start = (start_dt.date() - now_dt.date()).days
            
            # Approximate hours (assuming 7pm ET start = 00:00 UTC)
            # This is rough but fine for "days out" view
            hours_to_start = days_to_start * 24.0
            
        except Exception as e:
            print(f"Error parsing date from {ticker}: {e}")
            hours_to_start = 0
            days_to_start = 0
            start_dt = datetime.now()
            
        print(f"  Fetching {ticker} (Starts in ~{days_to_start} days)...")
        
        try:
            markets = kalshi.get_event_markets(ticker)
        except:
            print(f"    Failed to fetch markets for {ticker}")
            continue
            
        if not markets:
            continue
            
        for m in markets:
            # We only care about spread markets
            # format: KXNBASPREAD-YYMMDDTEAM-TEAM{spread}
            # The market ticker is usually the event ticker + suffix
            
            yes_bid = m.get('yes_bid', 0)
            yes_ask = m.get('yes_ask', 100) # Default to 100 if no ask? No, 0 usually means empty
            
            # If empty orderbook, skip
            if yes_bid == 0 and yes_ask == 0:
                continue
                
            # Valid market?
            # Spread = Ask - Bid. 
            # If ask is missing (0), we can't calculate tight spread, likely 100 or essentially infinite.
            # Let's treat 0 ask as 100 (max price) for visualization if bid exists?
            # Actually, usually 'yes_ask' is the lowest ask price. 
            # If no one is selling, yes_ask is effectively 100 (or None).
            # Kalshi returns 0 if empty.
            
            final_ask = yes_ask if yes_ask > 0 else 100
            
            spread_width = final_ask - yes_bid
            mid_price = (final_ask + yes_bid) / 2
            
            # Parse the line from ticker if possible
            # e.g. KXNBASPREAD-231225LALBOS-BOS5.5
            line_val = 0
            try:
                parts = m['ticker'].split('-')
                last_part = parts[-1]
                # Filter digits
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", last_part)
                if nums:
                    line_val = float(nums[0])
            except:
                pass
            
            market_data.append({
                'event': ticker,
                'market_ticker': m['ticker'],
                'line': line_val,
                'bid': yes_bid,
                'ask': yes_ask,
                'spread_width': spread_width,
                'mid_price': mid_price,
                'hours_to_start': hours_to_start,
                'days_to_start': days_to_start,
                'start_time': start_dt,
                'volume': m.get('volume', 0),
                'open_interest': m.get('open_interest', 0)
            })

    # 4. Create DataFrame & Visualize
    df = pd.DataFrame(market_data)
    
    if df.empty:
        print("No market data found.")
        return

    print(f"\nCollected {len(df)} market data points.")
    
    # Filter out empty markets (spread = 100 usually means empty)
    # Actually, lets look at "tradable" markets where spread < 100
    df_active = df[df['spread_width'] < 100].copy()
    
    if df_active.empty:
        print("No active markets (all spreads = 100).")
        return

    # Sort by spread width descending to find the widest ones
    print("\n=== TOP 20 WIDEST SPREADS (Active) ===")
    print(df_active.sort_values('spread_width', ascending=False)[['market_ticker', 'days_to_start', 'bid', 'ask', 'spread_width']].head(20).to_string())

    print("\n=== SPREAD STATISTICS BY DAYS OUT ===")
    # Bin days to start
    df_active['day_bin'] = df_active['days_to_start'].apply(lambda x: int(x))
    stats = df_active.groupby('day_bin')['spread_width'].describe()
    print(stats)

    # VISUALIZATION
    print("\nGenerating plots...")
    plt.figure(figsize=(18, 12))
    
    # 1. Spread Width vs Hours to Start
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=df_active, x='hours_to_start', y='spread_width', hue='day_bin', palette='viridis')
    plt.title('Spread Width vs. Hours to Start')
    plt.xlabel('Hours to Start')
    plt.ylabel('Spread (cents)')
    
    # 2. Spread Width vs Volume (New)
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df_active, x='volume', y='spread_width', hue='day_bin', palette='viridis')
    plt.title('Spread Width vs. Volume')
    plt.xlabel('Volume (Contracts)')
    plt.ylabel('Spread (cents)')
    plt.xscale('log') # Log scale for volume
    
    # 3. Spread Width vs Open Interest (New)
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=df_active, x='open_interest', y='spread_width', hue='day_bin', palette='viridis')
    plt.title('Spread Width vs. Open Interest')
    plt.xlabel('Open Interest')
    plt.ylabel('Spread (cents)')
    
    # 4. Volume vs Days Out
    plt.subplot(2, 3, 4)
    sns.barplot(data=df_active, x='day_bin', y='volume')
    plt.title('Avg Volume by Days Until Game')
    
    # 5. Open Interest vs Days Out
    plt.subplot(2, 3, 5)
    sns.barplot(data=df_active, x='day_bin', y='open_interest')
    plt.title('Avg Open Interest by Days Until Game')
    
    plt.tight_layout()
    plt.savefig('spread_volume_analysis.png')
    print("Saved spread_volume_analysis.png")
    
    # Liquidity Analysis
    print("\n=== LIQUIDITY ANALYSIS ===")
    # Define "Weak" as Low Vol (< 100) AND Wide Spread (> 10)
    # These are markets where price discovery is likely poor
    weak_markets = df_active[(df_active['volume'] < 100) & (df_active['spread_width'] > 10)]
    print(f"Found {len(weak_markets)} potentially 'weak' markets (Vol < 100, Spread > 10c)")
    
    if not weak_markets.empty:
        print(weak_markets.sort_values('spread_width', ascending=False)[['market_ticker', 'days_to_start', 'volume', 'open_interest', 'spread_width']].head(10).to_string())

if __name__ == "__main__":
    analyze_spreads()
