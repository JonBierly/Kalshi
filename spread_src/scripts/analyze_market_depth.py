#!/usr/bin/env python
"""
Market Depth Analyzer — Snapshot analysis of Kalshi NBA spread order books.

Pulls live orderbooks for all active NBA spread markets and analyzes:
1. Spread width (bid-ask gap) by spread level
2. Book depth (total contracts near top-of-book) by spread level
3. Depth asymmetry (more buyers or sellers?)
4. Volume comparison across spread levels
5. "Sophistication" indicators (tight spreads + deep books → pro activity)

Usage:
    python -m spread_src.scripts.analyze_market_depth
"""

import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient
from spread_src.data.spread_markets import parse_spread_ticker, SpreadMarket


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_orderbook(ob: dict) -> dict:
    """
    Parse Kalshi orderbook response into useful metrics.
    
    Kalshi returns: {'yes': [[price, qty], ...], 'no': [[price, qty], ...]}
    All entries are BIDS. A YES bid at price P is equivalent to a NO ask at (100-P).
    """
    yes_bids = ob.get('yes', []) or []
    no_bids = ob.get('no', []) or []
    
    # Best YES bid = highest price in yes_bids array
    best_yes_bid = max([p for p, q in yes_bids], default=0) if yes_bids else 0
    # Best NO bid = highest price in no_bids array
    best_no_bid = max([p for p, q in no_bids], default=0) if no_bids else 0
    
    # YES ask = 100 - best_no_bid (the best NO bid IS the YES ask)
    # NO ask = 100 - best_yes_bid
    yes_ask = (100 - best_no_bid) if best_no_bid > 0 else 0
    
    spread = yes_ask - best_yes_bid if (best_yes_bid > 0 and yes_ask > 0) else None
    midpoint = (best_yes_bid + yes_ask) / 2 if spread is not None else None
    
    # Depth within 5¢ of best bid/ask
    depth_near_bid = sum(q for p, q in yes_bids if p >= best_yes_bid - 5) if yes_bids else 0
    depth_near_ask = sum(q for p, q in no_bids if p >= best_no_bid - 5) if no_bids else 0
    
    # Total depth across all levels
    total_yes_depth = sum(q for p, q in yes_bids)
    total_no_depth = sum(q for p, q in no_bids)
    
    # Number of price levels with orders
    yes_levels = len(yes_bids)
    no_levels = len(no_bids)
    
    return {
        'best_yes_bid': best_yes_bid,
        'yes_ask': yes_ask,
        'best_no_bid': best_no_bid,
        'spread': spread,
        'midpoint': midpoint,
        'depth_near_bid': depth_near_bid,
        'depth_near_ask': depth_near_ask,
        'total_yes_depth': total_yes_depth,
        'total_no_depth': total_no_depth,
        'yes_levels': yes_levels,
        'no_levels': no_levels,
        'raw_yes_bids': yes_bids,
        'raw_no_bids': no_bids,
    }


def classify_spread_level(spread_value: float) -> str:
    """Categorize spread into buckets for comparison."""
    if spread_value <= 3.5:
        return "Tight (≤3.5)"
    elif spread_value <= 8.5:
        return "Mid (4.5-8.5)"
    else:
        return "Wide (≥9.5)"


def liquidity_score(spread, depth_near_bid, depth_near_ask) -> float:
    """
    Simple liquidity score: higher = more liquid.
    score = total_near_depth / spread
    """
    if spread is None or spread <= 0:
        return 0.0
    total_depth = depth_near_bid + depth_near_ask
    return total_depth / spread


# ── Main Analysis ────────────────────────────────────────────────────────────

def fetch_all_spread_data(client: KalshiClient) -> list:
    """Fetch orderbook + market details for every active NBA spread market."""
    
    print("=" * 80)
    print("  Kalshi NBA Spread Market Depth Analyzer")
    print(f"  Snapshot time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Find active NBA spread events
    print("\n📡 Fetching active NBA events...")
    events = client.get_nba_markets()
    
    if not events:
        print("No active NBA events found.")
        return []
    
    # 2. Get all spread markets for each event
    all_data = []
    
    for event in events:
        event_ticker = event.get('event_ticker', '')
        title = event.get('title', event_ticker)
        
        # Only process spread events
        if 'SPREAD' not in event_ticker.upper():
            # Try to find the spread sub-event
            spread_ticker = event_ticker.replace('KXNBAGAME', 'KXNBASPREAD')
            markets = client.get_event_markets(spread_ticker)
        else:
            markets = client.get_event_markets(event_ticker)
        
        if not markets:
            continue
            
        print(f"\n🏀 {title}")
        print(f"   Found {len(markets)} spread markets")
        
        for market in markets:
            ticker = market.get('ticker', '')
            
            # Only process spread markets
            if 'SPREAD' not in ticker:
                continue
            
            try:
                parsed = parse_spread_ticker(ticker)
            except ValueError:
                continue
            
            # Fetch orderbook
            ob = client.get_orderbook(ticker)
            time.sleep(0.05)  # Rate limit
            
            if not ob:
                continue
            
            ob_metrics = parse_orderbook(ob)
            
            # Get market details for volume
            details = market  # Already have details from event markets
            volume = details.get('volume', 0)
            volume_24h = details.get('volume_24h', 0)
            open_interest = details.get('open_interest', 0)
            
            # Get yes_bid/yes_ask from market details as backup
            detail_yes_bid = details.get('yes_bid', 0) or 0
            detail_yes_ask = details.get('yes_ask', 0) or 0
            
            row = {
                'ticker': ticker,
                'event': title,
                'game': f"{parsed['away_team']}@{parsed['home_team']}",
                'spread_team': parsed['spread_team'],
                'spread_value': parsed['spread_value'],
                'spread_category': classify_spread_level(parsed['spread_value']),
                'is_home': parsed['spread_team'] == parsed['home_team'],
                
                # Price data
                'yes_bid': ob_metrics['best_yes_bid'] or detail_yes_bid,
                'yes_ask': ob_metrics['yes_ask'] or detail_yes_ask,
                'bid_ask_spread': ob_metrics['spread'],
                'midpoint': ob_metrics['midpoint'],
                
                # Depth data
                'depth_near_bid': ob_metrics['depth_near_bid'],
                'depth_near_ask': ob_metrics['depth_near_ask'],
                'total_yes_depth': ob_metrics['total_yes_depth'],
                'total_no_depth': ob_metrics['total_no_depth'],
                'yes_levels': ob_metrics['yes_levels'],
                'no_levels': ob_metrics['no_levels'],
                
                # Volume data
                'volume': volume,
                'volume_24h': volume_24h,
                'open_interest': open_interest,
                
                # Liquidity score
                'liq_score': liquidity_score(ob_metrics['spread'], 
                                              ob_metrics['depth_near_bid'],
                                              ob_metrics['depth_near_ask']),
                
                # Raw book for later analysis
                'raw_yes_bids': ob_metrics['raw_yes_bids'],
                'raw_no_bids': ob_metrics['raw_no_bids'],
            }
            
            all_data.append(row)
    
    return all_data


def print_market_table(df: pd.DataFrame):
    """Print a per-market summary table."""
    
    print("\n" + "=" * 110)
    print("  PER-MARKET SNAPSHOT")
    print("=" * 110)
    
    cols = ['game', 'spread_team', 'spread_value', 'yes_bid', 'yes_ask', 
            'bid_ask_spread', 'depth_near_bid', 'depth_near_ask', 'volume', 'liq_score']
    
    display_df = df[cols].copy()
    display_df.columns = ['Game', 'Team', 'Spread', 'Bid', 'Ask', 
                          'B/A Width', 'Bid Depth', 'Ask Depth', 'Volume', 'Liq Score']
    
    # Sort by game then spread
    display_df = display_df.sort_values(['Game', 'Spread'])
    
    print(display_df.to_string(index=False, float_format='%.1f'))


def print_spread_category_analysis(df: pd.DataFrame):
    """Compare metrics across spread level categories."""
    
    print("\n" + "=" * 110)
    print("  ANALYSIS BY SPREAD LEVEL")
    print("  (Are tighter spreads more liquid? Do wider spreads attract less sophisticated money?)")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    
    if valid.empty:
        print("  No markets with valid bid-ask data.")
        return
    
    grouped = valid.groupby('spread_category').agg({
        'bid_ask_spread': ['mean', 'median', 'min', 'max'],
        'depth_near_bid': ['mean', 'sum'],
        'depth_near_ask': ['mean', 'sum'],
        'total_yes_depth': ['mean', 'sum'],
        'volume': ['mean', 'sum'],
        'liq_score': ['mean', 'median'],
        'ticker': 'count',
    })
    
    # Flatten columns
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
    grouped = grouped.rename(columns={'ticker_count': 'n_markets'})
    
    print(f"\n{'Category':<16} {'Mkt Count':>10} {'Avg Spread':>11} {'Med Spread':>11} "
          f"{'Avg Bid Depth':>14} {'Avg Ask Depth':>14} {'Avg Volume':>11} {'Avg Liq Score':>14}")
    print("-" * 110)
    
    for cat in ["Tight (≤3.5)", "Mid (4.5-8.5)", "Wide (≥9.5)"]:
        if cat not in grouped.index:
            continue
        row = grouped.loc[cat]
        print(f"{cat:<16} {int(row['n_markets']):>10} "
              f"{row['bid_ask_spread_mean']:>10.1f}¢ {row['bid_ask_spread_median']:>10.1f}¢ "
              f"{row['depth_near_bid_mean']:>13.1f} {row['depth_near_ask_mean']:>13.1f} "
              f"{row['volume_mean']:>10.0f} {row['liq_score_mean']:>13.2f}")


def print_sophistication_analysis(df: pd.DataFrame):
    """
    Identify which markets show signs of professional activity:
    - Tight spreads (2-4¢) suggest competing market makers
    - Deep books suggest institutional capital  
    - High volume suggests active trading
    """
    
    print("\n" + "=" * 110)
    print("  SOPHISTICATION INDICATORS")
    print("  (Tight spread + deep book = likely professional market making)")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    
    if valid.empty:
        return
    
    # Classify each market's sophistication
    def classify_sophistication(row):
        spread = row['bid_ask_spread']
        depth = row['depth_near_bid'] + row['depth_near_ask']
        volume = row['volume']
        
        if spread <= 4 and depth >= 20:
            return "🟢 PROFESSIONAL (tight spread, deep book)"
        elif spread <= 6 and depth >= 10:
            return "🟡 SEMI-PRO (moderate spread, some depth)"
        elif spread <= 8 and depth >= 5:
            return "🟠 RETAIL-MIXED (wider spread, thin depth)"
        else:
            return "🔴 RETAIL/DEAD (wide spread and/or no depth)"
    
    valid['sophistication'] = valid.apply(classify_sophistication, axis=1)
    
    # Summary
    soph_counts = valid['sophistication'].value_counts()
    print(f"\nMarket Classification:")
    for soph, count in soph_counts.items():
        pct = 100 * count / len(valid)
        print(f"  {soph}: {count} markets ({pct:.0f}%)")
    
    # Show pro markets in detail
    pro_markets = valid[valid['sophistication'].str.contains('PROFESSIONAL')]
    if not pro_markets.empty:
        print(f"\n  🟢 Professional Markets (tight spreads, deep books):")
        for _, row in pro_markets.iterrows():
            print(f"     {row['game']} {row['spread_team']}-{row['spread_value']:.1f} "
                  f"| Spread: {row['bid_ask_spread']:.0f}¢ "
                  f"| Depth: {row['depth_near_bid']:.0f}b/{row['depth_near_ask']:.0f}a "
                  f"| Vol: {row['volume']}")
    
    # Show dead markets
    dead_markets = valid[valid['sophistication'].str.contains('DEAD')]
    if not dead_markets.empty and len(dead_markets) <= 15:
        print(f"\n  🔴 Retail/Dead Markets (wide spreads, thin books):")
        for _, row in dead_markets.iterrows():
            spread_str = f"{row['bid_ask_spread']:.0f}¢" if pd.notna(row['bid_ask_spread']) else "N/A"
            print(f"     {row['game']} {row['spread_team']}-{row['spread_value']:.1f} "
                  f"| Spread: {spread_str} "
                  f"| Depth: {row['depth_near_bid']:.0f}b/{row['depth_near_ask']:.0f}a "
                  f"| Vol: {row['volume']}")


def print_depth_profile(df: pd.DataFrame):
    """Show how depth is distributed across the book for sample markets."""
    
    print("\n" + "=" * 110)
    print("  ORDER BOOK DEPTH PROFILES (Sample Markets)")
    print("  (Shows contract quantity at each price level)")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    if valid.empty:
        return
    
    # Pick the most liquid and least liquid markets to compare
    valid_sorted = valid.sort_values('liq_score', ascending=False)
    
    samples = []
    if len(valid_sorted) >= 1:
        samples.append(('MOST LIQUID', valid_sorted.iloc[0]))
    if len(valid_sorted) >= 2:
        samples.append(('2ND MOST LIQUID', valid_sorted.iloc[1]))
    
    # Also pick the least liquid non-dead market
    non_dead = valid_sorted[valid_sorted['total_yes_depth'] + valid_sorted['total_no_depth'] > 0]
    if len(non_dead) >= 1:
        samples.append(('LEAST LIQUID (active)', non_dead.iloc[-1]))
    
    for label, row in samples:
        ticker_short = f"{row['game']} {row['spread_team']}-{row['spread_value']:.1f}"
        print(f"\n  📊 {label}: {ticker_short}")
        print(f"     Best Bid: {row['yes_bid']:.0f}¢ | Ask: {row['yes_ask']:.0f}¢ | "
              f"Spread: {row['bid_ask_spread']:.0f}¢")
        
        # YES bids (buy side)
        yes_bids = row['raw_yes_bids']
        no_bids = row['raw_no_bids']
        
        if yes_bids:
            sorted_yes = sorted(yes_bids, key=lambda x: x[0], reverse=True)[:8]
            bid_str = " | ".join([f"{int(p)}¢×{int(q)}" for p, q in sorted_yes])
            print(f"     YES bids (buy side):  {bid_str}")
        else:
            print(f"     YES bids: EMPTY")
            
        if no_bids:
            sorted_no = sorted(no_bids, key=lambda x: x[0], reverse=True)[:8]
            ask_str = " | ".join([f"{int(p)}¢×{int(q)}" for p, q in sorted_no])
            print(f"     NO bids  (sell side): {ask_str}")
        else:
            print(f"     NO bids: EMPTY")


def print_home_vs_away(df: pd.DataFrame):
    """Compare liquidity for home vs away team spreads."""
    
    print("\n" + "=" * 110)
    print("  HOME vs AWAY SPREAD LIQUIDITY")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    if valid.empty:
        return
    
    for side, label in [(True, 'Home Team Spreads'), (False, 'Away Team Spreads')]:
        subset = valid[valid['is_home'] == side]
        if subset.empty:
            continue
        avg_spread = subset['bid_ask_spread'].mean()
        avg_depth = (subset['depth_near_bid'] + subset['depth_near_ask']).mean()
        avg_vol = subset['volume'].mean()
        avg_liq = subset['liq_score'].mean()
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    Avg B/A spread: {avg_spread:.1f}¢ | Avg depth near TOB: {avg_depth:.1f} "
              f"| Avg volume: {avg_vol:.0f} | Avg liq score: {avg_liq:.2f}")


def print_game_comparison(df: pd.DataFrame):
    """Compare overall liquidity across games."""
    
    print("\n" + "=" * 110)
    print("  GAME-LEVEL LIQUIDITY COMPARISON")
    print("  (Which games attract the most trading activity?)")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    if valid.empty:
        return
    
    game_stats = valid.groupby('game').agg({
        'bid_ask_spread': 'mean',
        'depth_near_bid': 'sum',
        'depth_near_ask': 'sum',
        'volume': 'sum',
        'liq_score': 'mean',
        'ticker': 'count'
    }).rename(columns={'ticker': 'n_markets'})
    
    game_stats['total_depth'] = game_stats['depth_near_bid'] + game_stats['depth_near_ask']
    game_stats = game_stats.sort_values('volume', ascending=False)
    
    print(f"\n{'Game':<16} {'Markets':>8} {'Avg Spread':>11} {'Total Depth':>12} "
          f"{'Total Volume':>13} {'Avg Liq Score':>14}")
    print("-" * 80)
    
    for game, row in game_stats.iterrows():
        print(f"{game:<16} {int(row['n_markets']):>8} "
              f"{row['bid_ask_spread']:>10.1f}¢ {int(row['total_depth']):>11} "
              f"{int(row['volume']):>12} {row['liq_score']:>13.2f}")


def print_summary_insights(df: pd.DataFrame):
    """Print key takeaways."""
    
    print("\n" + "=" * 110)
    print("  KEY INSIGHTS")
    print("=" * 110)
    
    valid = df[df['bid_ask_spread'].notna()].copy()
    if valid.empty:
        print("  No data to analyze.")
        return
    
    overall_spread = valid['bid_ask_spread'].median()
    overall_depth = (valid['depth_near_bid'] + valid['depth_near_ask']).median()
    
    # Where depth concentrates
    depth_by_cat = valid.groupby('spread_category')[['depth_near_bid', 'depth_near_ask']].sum()
    depth_by_cat['total'] = depth_by_cat['depth_near_bid'] + depth_by_cat['depth_near_ask']
    
    tight_depth = depth_by_cat.loc['Tight (≤3.5)', 'total'] if 'Tight (≤3.5)' in depth_by_cat.index else 0
    mid_depth = depth_by_cat.loc['Mid (4.5-8.5)', 'total'] if 'Mid (4.5-8.5)' in depth_by_cat.index else 0
    wide_depth = depth_by_cat.loc['Wide (≥9.5)', 'total'] if 'Wide (≥9.5)' in depth_by_cat.index else 0
    total_depth = tight_depth + mid_depth + wide_depth
    
    print(f"""
  📊 Overall:
    • Median bid-ask spread: {overall_spread:.0f}¢
    • Median depth near TOB: {overall_depth:.0f} contracts
    • Total markets analyzed: {len(valid)}
    
  📈 Where the liquidity lives:
    • Tight spreads (≤3.5):  {tight_depth:.0f} contracts ({100*tight_depth/max(total_depth,1):.0f}% of total)
    • Mid spreads (4.5-8.5): {mid_depth:.0f} contracts ({100*mid_depth/max(total_depth,1):.0f}% of total)  
    • Wide spreads (≥9.5):   {wide_depth:.0f} contracts ({100*wide_depth/max(total_depth,1):.0f}% of total)
  
  🎯 What this means for your trading:
    • Markets with spread ≤ 4¢ and depth ≥ 20 likely have professional participation
    • Markets with spread ≥ 10¢ are retail-dominated — your model edge persists longer
    • Consider concentrating on mid-range spreads: enough edge to exploit, 
      enough liquidity to enter/exit
""")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
    kalshi_key_path = "key.key"
    
    client = KalshiClient(kalshi_key_id, kalshi_key_path)
    
    # Fetch all data
    all_data = fetch_all_spread_data(client)
    
    if not all_data:
        print("\n❌ No spread market data found. Is there an active NBA game tonight?")
        return
    
    df = pd.DataFrame(all_data)
    
    # Drop raw book columns for display
    display_cols = [c for c in df.columns if not c.startswith('raw_')]
    
    # Run all analyses
    print_market_table(df)
    print_spread_category_analysis(df)
    print_sophistication_analysis(df)
    print_depth_profile(df)
    print_home_vs_away(df)
    print_game_comparison(df)
    print_summary_insights(df)
    
    # Save raw data
    save_path = f"reports/market_depth_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    os.makedirs('reports', exist_ok=True)
    
    # Convert to serializable format
    save_data = []
    for row in all_data:
        r = {k: v for k, v in row.items() if not k.startswith('raw_')}
        # Convert numpy types
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                r[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                r[k] = float(v)
        save_data.append(r)
    
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_markets': len(save_data),
            'markets': save_data
        }, f, indent=2)
    
    print(f"\n💾 Raw data saved to {save_path}")


if __name__ == "__main__":
    main()
