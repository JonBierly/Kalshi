"""
Check LIVE Kalshi spread markets and compare to model predictions.

For any live NBA games:
1. Get spread markets from Kalshi
2. Get live game data  
3. Generate model prediction
4. Compare and find edges
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.kalshi import KalshiClient
from spread_src.data.spread_markets import parse_spread_ticker, SpreadMarket, check_spread_arbitrage
from spread_src.models.spread_model import SpreadDistributionModel
import joblib
from scipy import stats
import numpy as np
import requests


def get_live_nba_games():
    """Get currently live NBA games."""
    try:
        # NBA API endpoint for today's games
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        
        url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('scoreboard', {}).get('games', [])
            
            # Filter for live games only
            live_games = [g for g in games if g.get('gameStatus') == 2]
            
            return live_games
        else:
            print(f"Failed to fetch NBA games: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching NBA games: {e}")
        return []


def check_live_spread_opportunities():
    """
    Main function: Check live games for spread betting opportunities.
    """
    print("=" * 80)
    print("LIVE SPREAD OPPORTUNITY FINDER")
    print("=" * 80)
    
    # Step 1: Check for live NBA games
    print("\nStep 1: Checking for live NBA games...")
    live_games = get_live_nba_games()
    
    if not live_games:
        print("✗ No live NBA games right now")
        print("\nCome back when games are in progress!")
        print("Typical NBA game times: 7:00 PM - 10:30 PM ET")
        return
    
    print(f"✓ Found {len(live_games)} live game(s)")


def check_live_spread_opportunities():
    """
    Main function: Check live games for spread betting opportunities.
    """
    print("=" * 80)
    print("LIVE SPREAD OPPORTUNITY FINDER")
    print("=" * 80)
    
    # Step 1: Check for live NBA games
    print("\nStep 1: Checking for live NBA games...")
    live_games = get_live_nba_games()
    
    if not live_games:
        print("✗ No live NBA games right now")
        print("\nCome back when games are in progress!")
        print("Typical NBA game times: 7:00 PM - 10:30 PM ET")
        return
    
    print(f"✓ Found {len(live_games)} live game(s)")
    
    for game in live_games:
        home_team = game.get('homeTeam', {}).get('teamTricode', 'UNK')
        away_team = game.get('awayTeam', {}).get('teamTricode', 'UNK')
        period = game.get('period', 0)
        game_clock = game.get('gameClock', '')
        
        home_score = game.get('homeTeam', {}).get('score', 0)
        away_score = game.get('awayTeam', {}).get('score', 0)
        
        print(f"\n  {away_team} @ {home_team}")
        print(f"  Q{period} - {game_clock}")
        print(f"  Score: {away_team} {away_score}, {home_team} {home_score}")
    
    # Step 2: Check Kalshi for spread markets
    print("\n" + "=" * 80)
    print("Step 2: Fetching Kalshi spread markets...")
    print("=" * 80)
    
    try:
        client = KalshiClient()
        print("✓ Connected to Kalshi")
    except Exception as e:
        print(f"✗ Failed to connect to Kalshi: {e}")
        print("\nMake sure you have credentials in environment:")
        print("  export KALSHI_API_KEY_ID='your_key'")
        print("  export KALSHI_PRIVATE_KEY='your_private_key'")
        return
    
    try:
        # Try to get spread markets
        markets = client.get_markets(event_ticker="KXNBASPREAD")
        
        if not markets or len(markets) == 0:
            print("✗ No spread markets found on Kalshi")
            print("\nPossible reasons:")
            print("  - Kalshi doesn't offer spread markets for today's games")
            print("  - Markets haven't been created yet")
            print("  - Markets already settled")
            return
        
        print(f"✓ Found {len(markets)} spread market(s)")
        
        # Show sample markets
        print("\nSample markets:")
        for market in markets[:5]:
            ticker = market.get('ticker', 'N/A')
            title = market.get('title', 'N/A')
            yes_ask = market.get('yes_ask', 0)
            
            print(f"  {ticker}")
            print(f"    {title}")
            print(f"    Price: {yes_ask}¢")
        
    except Exception as e:
        print(f"✗ Error fetching markets: {e}")
        return
    
    # Step 3: For each live game, check for opportunities
    print("\n" + "=" * 80)
    print("Step 3: Analyzing opportunities...")
    print("=" * 80)
    
    print("\n⚠️  TO COMPLETE THIS:")
    print("Need to:")
    print("  1. Match Kalshi markets to live games (by team names)")
    print("  2. Get live game features (efg, turnovers, etc.)")
    print("  3. Generate model predictions")
    print("  4. Compare model prob vs market price")
    
    print("\nFor now, showing what markets exist:")
    
    # Group markets by game
    games_dict = {}
    for market in markets:
        try:
            parsed = parse_spread_ticker(market['ticker'])
            game_key = f"{parsed['away_team']}@{parsed['home_team']}"
            
            if game_key not in games_dict:
                games_dict[game_key] = []
            
            spread_market = SpreadMarket(
                ticker=market['ticker'],
                team=parsed['spread_team'],
                spread=parsed['spread_value'],
                yes_bid=market.get('yes_bid', 0),
                yes_ask=market.get('yes_ask', 0),
                no_bid=market.get('no_bid', 0),
                no_ask=market.get('no_ask', 0),
                subtitle=market.get('title', '')
            )
            
            games_dict[game_key].append(spread_market)
        except:
            continue
    
    print(f"\nFound markets for {len(games_dict)} game(s):")
    
    for game_key, spread_markets in games_dict.items():
        print(f"\n{game_key}:")
        
        # Show markets
        for sm in spread_markets:
            print(f"  {sm.team} >{sm.spread}: ${sm.yes_ask/100:.2f}")
        
        # Check arbitrage
        arb = check_spread_arbitrage(spread_markets)
        if arb:
            print("\n  ⚡ ARBITRAGE FOUND:")
            for opp in arb:
                print(f"    {opp['strategy']}")
                print(f"    Profit: {opp['arbitrage_cents']:.1f}¢")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nTo find prediction-based edges, need to:")
    print("1. Build live game tracker (like your binary tracker)")
    print("2. Match games to Kalshi markets")
    print("3. Generate spread predictions in real-time")
    print("4. Compare and execute trades")
    
    print("\nFor now:")
    print("- Look for ARBITRAGE opportunities (shown above)")
    print("- These are risk-free regardless of model predictions!")


if __name__ == "__main__":
    check_live_spread_opportunities()
