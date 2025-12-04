"""
Simple spread market checker - just shows what's available on Kalshi.

No authentication needed (uses demo mode).
"""

import sys
import os
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def get_live_nba_games():
    """Get currently live NBA games from NBA API."""
    try:
        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('scoreboard', {}).get('games', [])
            live_games = [g for g in games if g.get('gameStatus') == 2]
            return live_games
        return []
    except Exception as e:
        print(f"Error fetching NBA games: {e}")
        return []


def main():
    print("=" * 80)
    print("LIVE NBA GAMES &  SPREAD MARKETS")
    print("=" * 80)
    
    # Step 1: Live games
    print("\nFetching live NBA games...")
    live_games = get_live_nba_games()
    
    if not live_games:
        print("✗ No live NBA games right now")
        print("\nTypical NBA game times: 7:00 PM - 10:30 PM ET")
        return
    
    print(f"✓ Found {len(live_games)} live games:\n")
    
    for game in live_games:
        home = game.get('homeTeam', {})
        away = game.get('awayTeam', {})
        period = game.get('period', 0)
        clock = game.get('gameClock', '')
        
        home_score = home.get('score', 0)
        away_score = away.get('score', 0)
        home_team = home.get('teamTricode', 'UNK')
        away_team = away.get('teamTricode', 'UNK')
        
        print(f"  {away_team} @ {home_team}")
        print(f"  Q{period} - {clock}")
        print(f"  Score: {away_score} - {home_score}")
        print()
    
    # Step 2: Check for spread markets (simplified)
    print("=" * 80)
    print("KALSHI SPREAD MARKETS")
    print("=" * 80)
    
    print("\nTo check Kalshi spread markets, you need:")
    print("  1. Kalshi API credentials")
    print("  2. Active API key")
    
    print("\nFor now, here's what to look for:")
    print("  - Check Kalshi website for KXNBASPREAD markets")
    print("  - Look for markets like: 'DAL wins by over 6.5 Points'")
    print("  - Compare market prices to model predictions")
    
    print("\n" + "=" * 80)
    print("MANUAL EXPERIMENT")
    print("=" * 80)
    
    print("\nTo test if edges exist:")
    print("\n1. Go to Kalshi.com")
    print("2. Find NBA spread markets for the games above")
    print("3. For each market, note:")
    print("   - Team")
    print("   - Spread (e.g., >6.5 points)")
    print("   - Price (in cents)")
    
    print("\n4. Then run your Ridge model on these games:")
    print("   python -m scripts.paper_trade")
    
    print("\n5. Compare:")
    print("   - Model P(diff > 6.5 points) = ?")
    print("   - Kalshi price = ?")
    print("   - Edge = Model - Market")
    
    print("\n6. If edge > 5%, that's a betting opportunity!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
