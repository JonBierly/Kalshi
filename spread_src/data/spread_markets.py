"""
Spread market utilities - parse and understand Kalshi spread markets.

Market format: "Atlanta wins by over 3.5 Points"
Ticker format: KXNBASPREAD-25DEC01ATLDET-DET6
    - Date: 25DEC01
    - Teams: ATLDET (away @ home) 
    - Spread: DET6 (Detroit -6.5)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpreadMarket:
    """Represents a Kalshi spread market."""
    ticker: str
    team: str  # Team that must cover the spread
    spread: float  # Spread value (e.g., 3.5, 6.5)
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    subtitle: str
    
    @property
    def is_home(self) -> bool:
        """True if this is a home team spread."""
        # Parse ticker to determine
        return self._extract_team_from_ticker() == self._extract_home_from_ticker()
    
    def _extract_team_from_ticker(self) -> str:
        """Extract team from ticker (e.g., DET from KXNBASPREAD-25DEC01ATLDET-DET6)."""
        match = re.search(r'-([A-Z]{3})(\d+\.?\d*)$', self.ticker)
        if match:
            return match.group(1)
        return None
    
    def _extract_home_from_ticker(self) -> str:
        """Extract home team from ticker."""
        # Format: KXNBASPREAD-{date}{away}{home}-{team}{spread}
        # E.g., KXNBASPREAD-25DEC01ATLDET
        match = re.search(r'-\d{7}[A-Z]{3}([A-Z]{3})-', self.ticker)
        if match:
            return match.group(1)
        return None


def parse_spread_ticker(ticker: str) -> Dict[str, any]:
    """
    Parse Kalshi spread ticker.
    
    Example: KXNBASPREAD-25DEC01HOUUTA-UTA3
    Means: "UTA wins by over 3 points"
    
    Returns:
        {
            'date': '25DEC01',
            'away_team': 'HOU',
            'home_team': 'UTA', 
            'spread_team': 'UTA',
            'spread_value': 3.0  # Integer in ticker, but could be 3.5 in reality
        }
    """
    # Pattern: KXNBASPREAD-{date}{away}{home}-{team}{spread}
    # Note: spread in ticker is INTEGER (e.g., 3, 6, 9)
    # Date is alphanumeric: 25DEC01
    pattern = r'KXNBASPREAD-([A-Z0-9]{7})([A-Z]{3})([A-Z]{3})-([A-Z]{3})(\d+)'
    
    match = re.match(pattern, ticker)
    if not match:
        raise ValueError(f"Invalid spread ticker format: {ticker}")
    
    date, away, home, spread_team, spread_int = match.groups()
    
    # IMPORTANT: Ticker shows integer but actual line is integer + 0.5
    # "UTA3" means "UTA by over 3.5 points" (not 3.0)
    # "PHX10" means "PHX by over 10.5 points" (not 10.0)
    
    return {
        'date': date,
        'away_team': away,
        'home_team': home,
        'spread_team': spread_team,
        'spread_value': float(spread_int) + 0.5  # Add 0.5 for actual line
    }


def parse_spread_subtitle(subtitle: str) -> Dict[str, any]:
    """
    Parse subtitle like "Atlanta wins by over 3.5 Points"
    
    Returns:
        {
            'team': 'Atlanta',
            'spread': 3.5
        }
    """
    pattern = r'(.+?)\s+wins by over\s+([\d.]+)\s+Points?'
    match = re.match(pattern, subtitle)
    
    if not match:
        raise ValueError(f"Invalid subtitle format: {subtitle}")
    
    team, spread = match.groups()
    
    return {
        'team': team,
        'spread': float(spread)
    }


def group_spread_markets_by_game(markets: List[SpreadMarket]) -> Dict[str, List[SpreadMarket]]:
    """
    Group spread markets by game.
    
    Returns:
        Dict mapping game identifier (date+away+home) to list of spread markets
    """
    games = {}
    
    for market in markets:
        try:
            parsed = parse_spread_ticker(market.ticker)
            game_key = f"{parsed['date']}_{parsed['away_team']}_{parsed['home_team']}"
            
            if game_key not in games:
                games[game_key] = {
                    'away_team': parsed['away_team'],
                    'home_team': parsed['home_team'],
                    'date': parsed['date'],
                    'markets': []
                }
            
            games[game_key]['markets'].append(market)
        except ValueError:
            continue
    
    return games


def check_spread_arbitrage(markets: List[SpreadMarket]) -> Optional[Dict]:
    """
    Check for arbitrage in spread markets.
    
    Invariant: If team wins by >6.5, they also win by >3.5
    So: P(>6.5) <= P(>3.5) must hold
    
    Arbitrage opportunity: Buy lower spread, Sell higher spread
    - Buy lower YES at yes_ask
    - Sell higher YES at yes_bid (or equivalently buy NO at no_ask)
    
    Returns:
        Arbitrage opportunity dict or None
    """
    # Group by team
    markets_by_team = {}
    for market in markets:
        info = parse_spread_ticker(market.ticker)
        team = info['spread_team']
        
        if team not in markets_by_team:
            markets_by_team[team] = []
        
        markets_by_team[team].append({
            'spread': info['spread_value'],
            'yes_ask': market.yes_ask,
            'yes_bid': market.yes_bid,
            'no_ask': market.no_ask,
            'no_bid': market.no_bid,
            'market': market
        })
    
    # Check ordering for each team
    arbitrage_opportunities = []
    
    for team, team_markets in markets_by_team.items():
        # Sort by spread (ascending)
        sorted_markets = sorted(team_markets, key=lambda x: x['spread'])
        
        # Check if probabilities are monotonically decreasing
        for i in range(len(sorted_markets) - 1):
            lower_spread = sorted_markets[i]
            higher_spread = sorted_markets[i + 1]
            
            # Arbitrage strategy: Buy lower YES, Sell higher YES
            # Cost: Pay yes_ask for lower spread
            # Revenue: Receive yes_bid for higher spread
            # 
            # If both hit: +1 on lower, +1 on higher = net 0 (both pay $1)
            # If only lower hits: +1 on lower, 0 on higher = net +1
            # If neither hit: 0 on both
            #
            # Profit = yes_bid(higher) - yes_ask(lower)
            # For arbitrage to exist: yes_bid(higher) > yes_ask(lower)
            
            buy_price = lower_spread['yes_ask']  # Cost to buy lower spread
            sell_price = higher_spread['yes_bid']  # Revenue from selling higher spread
            
            if sell_price > buy_price:
                # Arbitrage exists!
                arbitrage_opportunities.append({
                    'type': 'SPREAD_ORDERING_VIOLATION',
                    'team': team,
                    'lower_spread': lower_spread['spread'],
                    'higher_spread': higher_spread['spread'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'arbitrage_cents': sell_price - buy_price,
                    'strategy': f"Buy {team} >{lower_spread['spread']} YES @ {buy_price}¢, Sell {team} >{higher_spread['spread']} YES @ {sell_price}¢"
                })
    
    return arbitrage_opportunities if arbitrage_opportunities else None


# Example usage
if __name__ == "__main__":
    # Test parsing
    ticker = "KXNBASPREAD-25DEC01ATLDET-DET6"
    parsed = parse_spread_ticker(ticker)
    print(f"Parsed ticker: {parsed}")
    # {'date': '25DEC01', 'away_team': 'ATL', 'home_team': 'DET', 
    #  'spread_team': 'DET', 'spread_value': 6.5}
    
    subtitle = "Atlanta wins by over 3.5 Points"
    parsed_sub = parse_spread_subtitle(subtitle)
    print(f"Parsed subtitle: {parsed_sub}")
    # {'team': 'Atlanta', 'spread': 3.5}
    
    # Test arbitrage detection
    fake_markets = [
        SpreadMarket("KXNBASPREAD-25DEC01ATLDET-DET3", "DET", 3.5, 30, 35, 65, 70, "Detroit wins by over 3.5 Points"),
        SpreadMarket("KXNBASPREAD-25DEC01ATLDET-DET6", "DET", 6.5, 35, 40, 60, 65, "Detroit wins by over 6.5 Points"),  # WRONG! Should be lower
    ]
    
    arb = check_spread_arbitrage(fake_markets)
    if arb:
        print(f"\n⚡ ARBITRAGE FOUND: {arb}")
