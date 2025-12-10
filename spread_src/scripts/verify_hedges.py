
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime, timedelta
from data.kalshi import KalshiClient
import argparse

API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-12-09')
    args = parser.parse_args()
    
    target_date = datetime.strptime(args.date, '%Y-%m-%d')
    # 4am to 4am
    start_ts = int(target_date.replace(hour=4, minute=0, second=0).timestamp() * 1000)
    end_ts = int((target_date + timedelta(days=1)).replace(hour=4, minute=0, second=0).timestamp() * 1000)
    
    print(f"\n🔎 CONFLICT CHECK (Effective Side Analysis) for {args.date}")
    
    try:
        kalshi = KalshiClient(API_KEY, KEY_PATH)
        fills = kalshi.get_fills(min_ts=start_ts, max_ts=end_ts, limit=1000)
    except Exception as e:
        print(f"Error connecting: {e}")
        return

    print(f"Total Fills: {len(fills)}")

    ticker_sides = {}

    for f in fills:
        tick = f['ticker']
        action = f['action']     # buy/sell
        side = f['side']         # yes/no
        
        # Calculate Effective Side (Direction)
        # Long YES = 1
        # Short YES = -1 (Long NO)
        
        if action == 'buy':
            eff_side = 1 if side == 'yes' else -1
        else: # sell
            eff_side = -1 if side == 'yes' else 1
            
        if tick not in ticker_sides:
            ticker_sides[tick] = {'sides': set(), 'actions': set(), 'details': []}
        
        ticker_sides[tick]['sides'].add(eff_side)
        ticker_sides[tick]['actions'].add(f"{action.upper()} {side.upper()}")
        ticker_sides[tick]['details'].append(f"{action.upper()} {side.upper()} @ {f['price']} (Yes:{f.get('yes_price')} No:{f.get('no_price')})")

    hedged_tickers = []
    one_way_tickers = []

    print(f"\n{'TICKER':<40} | {'DIRS':<10} | {'ACTIONS'}")
    print('-'*80)

    for tick, data in ticker_sides.items():
        sides = data['sides']
        actions = sorted(list(data['actions']))
        
        is_hedged = (1 in sides) and (-1 in sides)
        
        dir_str = 'MIXED' if is_hedged else ('BULLISH' if 1 in sides else 'BEARISH')
        
        print(f"{tick[-25:]:<40} | {dir_str:<10} | {actions}")
        
        if len(data['details']) < 5:
            for d in data['details']:
                print(f"    -> {d}")
        else:
             print(f"    -> (First 3): {data['details'][:3]}")
        
        if is_hedged:
            hedged_tickers.append(tick)
        else:
            one_way_tickers.append(tick)

    print(f"\nSUMMARY:")
    print(f"  One-Way Tickers (Doubling Down): {len(one_way_tickers)}")
    print(f"  Hedged Tickers (Round Trips):    {len(hedged_tickers)}")

    if len(hedged_tickers) == 0:
        print("\n✅ VERIFIED: You never took an opposing position on any ticker.")
        print("   Every trade you made was in the SAME direction (accumulating exposure).")
    else:
        print("\n❌ FOUND HEDGES: The analysis script might be missing something!")
        print(f"   Check these tickers: {hedged_tickers}")

if __name__ == "__main__":
    main()
