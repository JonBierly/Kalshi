#!/usr/bin/env python
"""
Analyze Kalshi account balance and P&L history using API data.

Reconstructs cash flow by fetching:
1. Fills (Entry costs)
2. Settled Positions (Payouts/Realized P&L)

Also analyzes trade characteristics to understand P&L drivers.
"""

import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

# Add parent directory to path to import data.kalshi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.kalshi import KalshiClient

# Credentials
API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Kalshi P&L from API.")
    parser.add_argument("--days", type=int, default=3, help="Number of days to look back")
    parser.add_argument("--date", type=str, default=None, help="Specific date to analyze (YYYY-MM-DD)")
    parser.add_argument("--detail", action="store_true", help="Show detailed trade breakdown")
    return parser.parse_args()

def fetch_data(client, days_back):
    """Fetch fills and settled positions."""
    start_dt = datetime.now() - timedelta(days=days_back)
    min_ts = int(start_dt.timestamp() * 1000)
    
    print(f"📥 Fetching data since {start_dt.strftime('%Y-%m-%d')}...")
    
    # 1. Fetch Fills (Trade Costs)
    fills = client.get_fills(min_ts=min_ts, limit=1000)
    print(f"  ✓ Fetched {len(fills)} fills")
    
    # 2. Fetch Settled Positions (Payouts)
    settled = client.get_positions(settlement_status='settled', limit=1000)
    market_positions = settled.get('market_positions', [])
    print(f"  ✓ Fetched {len(market_positions)} settled positions")
    
    return fills, market_positions

def extract_game_date(ticker):
    """Extract game date from KXNBASPREAD ticker."""
    if not ticker or not ticker.startswith('KXNBASPREAD'):
        return None
    match = re.search(r'-(\d{2}[A-Z]{3}\d{2})', ticker)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%y%b%d').date()
        except:
            pass
    return None

def extract_spread_line(ticker):
    """Extract spread line from ticker (e.g., OKC5 -> 5, NYK18 -> 18)."""
    if not ticker:
        return None
    # Pattern: team abbreviation followed by number at the end
    match = re.search(r'-([A-Z]+)(\d+)$', ticker)
    if match:
        return int(match.group(2))
    return None

def extract_matchup(ticker):
    """Extract matchup from ticker."""
    if not ticker or not ticker.startswith('KXNBASPREAD'):
        return None
    # Pattern: KXNBASPREAD-25DEC09NYKTOR-NYK18
    match = re.search(r'-\d{2}[A-Z]{3}\d{2}([A-Z]+)([A-Z]+)-', ticker)
    if match:
        return f"{match.group(1)} vs {match.group(2)}"
    return None

def process_settled_positions(settled_positions):
    """Process settled positions into a DataFrame with enriched data."""
    pnl_events = []
    
    for pos in settled_positions:
        ts_raw = pos.get('last_updated_ts')
        if not ts_raw:
            continue
        
        try:
            ts = float(ts_raw)
            if ts > 100000000000:
                dt = datetime.fromtimestamp(ts / 1000.0)
            else:
                dt = datetime.fromtimestamp(ts)
        except ValueError:
            try:
                dt = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
            except:
                continue
        
        ticker = pos.get('ticker')
        realized_pnl = pos.get('realized_pnl', 0) / 100.0
        fees_paid = pos.get('fees_paid', 0) / 100.0
        total_traded = pos.get('total_traded', 0)
        
        # Extract game date for NBA spread markets
        game_date = extract_game_date(ticker)
        if game_date:
            dt = datetime.combine(game_date, datetime.min.time())
        
        spread_line = extract_spread_line(ticker)
        matchup = extract_matchup(ticker)
        is_nba_spread = ticker and ticker.startswith('KXNBASPREAD')
        
        pnl_events.append({
            'date': dt.date(),
            'datetime': dt,
            'ticker': ticker,
            'pnl': realized_pnl,
            'fees': fees_paid,
            'total_traded': total_traded,
            'spread_line': spread_line,
            'matchup': matchup,
            'is_nba_spread': is_nba_spread,
            'type': 'settlement'
        })
        
    return pd.DataFrame(pnl_events)

def process_fills(fills):
    """Process fills into a DataFrame with trade details."""
    fill_data = []
    
    for fill in fills:
        ts_str = fill.get('created_time')
        if not ts_str:
            continue
        
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        ticker = fill.get('ticker')
        
        side = fill.get('side', '')
        action = fill.get('action', '')
        count = fill.get('count', 0)
        yes_price = fill.get('yes_price', 0)
        no_price = fill.get('no_price', 0)
        is_taker = fill.get('is_taker', False)
        
        # Calculate the market spread at fill time
        market_spread = abs(100 - yes_price - no_price)  # In cents
        
        # Determine our entry price
        if side == 'yes':
            entry_price = yes_price
        else:
            entry_price = no_price
        
        game_date = extract_game_date(ticker)
        spread_line = extract_spread_line(ticker)
        matchup = extract_matchup(ticker)
        
        fill_data.append({
            'datetime': dt,
            'date': dt.date(),
            'game_date': game_date,
            'ticker': ticker,
            'side': side,
            'action': action,
            'count': count,
            'yes_price': yes_price,
            'no_price': no_price,
            'entry_price': entry_price,
            'market_spread_cents': market_spread,
            'is_taker': is_taker,
            'spread_line': spread_line,
            'matchup': matchup,
            'is_nba_spread': ticker and ticker.startswith('KXNBASPREAD'),
        })
    
    return pd.DataFrame(fill_data)

def analyze_pnl(df, date_filter=None):
    """Analyze P&L summary."""
    if df.empty:
        print("No settled P&L data found.")
        return
    
    # Filter by date if specified
    if date_filter:
        df = df[df['date'] == date_filter]
        if df.empty:
            print(f"No data for {date_filter}")
            return
    
    print("\n📊 Daily Realized P&L (By Game Date):")
    daily = df.groupby('date')['pnl'].sum()
    print(daily)
    
    print(f"\nTotal Realized P&L (Period): ${df['pnl'].sum():.2f}")
    
    # Breakdown by market type
    if 'is_nba_spread' in df.columns:
        nba_pnl = df[df['is_nba_spread']]['pnl'].sum()
        other_pnl = df[~df['is_nba_spread']]['pnl'].sum()
        print(f"\n  NBA Spreads: ${nba_pnl:.2f}")
        print(f"  Other Markets: ${other_pnl:.2f}")
    
    # Save plot
    if len(daily) > 1:
        plt.figure(figsize=(10, 6))
        colors = ['green' if v >= 0 else 'red' for v in daily.values]
        sns.barplot(x=[str(d) for d in daily.index], y=daily.values, palette=colors)
        plt.title('Daily Realized P&L (By Game Date)')
        plt.ylabel('P&L ($)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('kalshi_api_pnl.png')
        print("\n✅ Saved chart to kalshi_api_pnl.png")

def analyze_trade_characteristics(fills_df, positions_df, date_filter=None):
    """Analyze what trade characteristics correlate with wins/losses."""
    
    if positions_df.empty:
        print("\nNo settled positions to analyze.")
        return
    
    # Filter to NBA spreads only for this analysis
    nba_positions = positions_df[positions_df['is_nba_spread']].copy()
    
    if date_filter:
        nba_positions = nba_positions[nba_positions['date'] == date_filter]
    
    if nba_positions.empty:
        print("\nNo NBA spread positions for the selected period.")
        return
    
    print("\n" + "="*60)
    print("📈 TRADE CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Aggregate fills by ticker to get trading characteristics
    if not fills_df.empty:
        nba_fills = fills_df[fills_df['is_nba_spread']].copy()
        
        if date_filter:
            nba_fills = nba_fills[nba_fills['game_date'] == date_filter]
        
        if not nba_fills.empty:
            # Group fills by ticker
            ticker_stats = nba_fills.groupby('ticker').agg({
                'count': 'sum',
                'entry_price': 'mean',
                'market_spread_cents': 'mean',
                'is_taker': 'mean',  # % of fills as taker
            }).reset_index()
            ticker_stats.columns = ['ticker', 'total_contracts', 'avg_entry_price', 'avg_market_spread', 'taker_ratio']
            
            # Merge with P&L
            merged = nba_positions.merge(ticker_stats, on='ticker', how='left')
            
            print("\n📉 LOSING TRADES (NBA Spreads):")
            losers = merged[merged['pnl'] < 0].sort_values('pnl')
            if not losers.empty:
                for _, row in losers.iterrows():
                    print(f"\n  {row['ticker']}")
                    print(f"    P&L: ${row['pnl']:.2f} | Spread Line: {row['spread_line']}")
                    if pd.notna(row.get('avg_entry_price')):
                        print(f"    Avg Entry: {row['avg_entry_price']:.0f}¢ | Mkt Spread: {row['avg_market_spread']:.1f}¢")
                        print(f"    Contracts: {row['total_contracts']:.0f} | Taker %: {row['taker_ratio']*100:.0f}%")
                
                # Summary stats for losers
                print(f"\n  Summary of losing trades:")
                print(f"    Total loss: ${losers['pnl'].sum():.2f} across {len(losers)} positions")
                if 'avg_market_spread' in losers.columns:
                    avg_spread = losers['avg_market_spread'].mean()
                    if pd.notna(avg_spread):
                        print(f"    Avg market spread: {avg_spread:.1f}¢")
            else:
                print("  No losing trades!")
            
            print("\n📈 WINNING TRADES (NBA Spreads):")
            winners = merged[merged['pnl'] > 0].sort_values('pnl', ascending=False)
            if not winners.empty:
                for _, row in winners.head(5).iterrows():
                    print(f"\n  {row['ticker']}")
                    print(f"    P&L: ${row['pnl']:.2f} | Spread Line: {row['spread_line']}")
                    if pd.notna(row.get('avg_entry_price')):
                        print(f"    Avg Entry: {row['avg_entry_price']:.0f}¢ | Mkt Spread: {row['avg_market_spread']:.1f}¢")
                
                if len(winners) > 5:
                    print(f"\n  ... and {len(winners) - 5} more winning trades")
                
                # Summary stats for winners
                print(f"\n  Summary of winning trades:")
                print(f"    Total profit: ${winners['pnl'].sum():.2f} across {len(winners)} positions")
                if 'avg_market_spread' in winners.columns:
                    avg_spread = winners['avg_market_spread'].mean()
                    if pd.notna(avg_spread):
                        print(f"    Avg market spread: {avg_spread:.1f}¢")
            else:
                print("  No winning trades!")
            
            # Compare winners vs losers
            if not losers.empty and not winners.empty:
                print("\n📊 WINNERS vs LOSERS COMPARISON:")
                if 'avg_market_spread' in merged.columns:
                    win_spread = winners['avg_market_spread'].mean()
                    lose_spread = losers['avg_market_spread'].mean()
                    if pd.notna(win_spread) and pd.notna(lose_spread):
                        print(f"    Avg Mkt Spread - Winners: {win_spread:.1f}¢ | Losers: {lose_spread:.1f}¢")
                
                if 'avg_entry_price' in merged.columns:
                    win_entry = winners['avg_entry_price'].mean()
                    lose_entry = losers['avg_entry_price'].mean()
                    if pd.notna(win_entry) and pd.notna(lose_entry):
                        print(f"    Avg Entry Price - Winners: {win_entry:.0f}¢ | Losers: {lose_entry:.0f}¢")
                
                if 'spread_line' in merged.columns:
                    win_line = winners['spread_line'].mean()
                    lose_line = losers['spread_line'].mean()
                    if pd.notna(win_line) and pd.notna(lose_line):
                        print(f"    Avg Spread Line - Winners: {win_line:.1f} | Losers: {lose_line:.1f}")

def check_current_standing(client):
    """Check current balance vs portfolio value."""
    print("\n💰 Current Account Standing:")
    balance_data = client.get_balance()
    if not balance_data:
        print("  Failed to fetch balance.")
        return
        
    cash = balance_data['balance']
    value = balance_data['portfolio_value']
    unrealized = value - cash
    
    print(f"  Cash Balance:    ${cash:>8.2f}")
    print(f"  Portfolio Value: ${value:>8.2f}")
    print(f"  Unrealized P&L:  ${unrealized:>8.2f}")
    
    # Check open positions
    positions = client.get_positions(settlement_status='unsettled')
    market_pos = positions.get('market_positions', [])
    
    print(f"\n📦 Open Positions: {len(market_pos)}")
    for pos in market_pos:
        ticker = pos.get('ticker')
        count = pos.get('position')
        cost = pos.get('total_cost', 0) / 100.0
        if count != 0:
            print(f"  {ticker}: {count} contracts (Cost: ${cost:.2f})")
    
    if unrealized < -10:
        print("\n⚠️  LARGE UNREALIZED LOSS DETECTED!")
        print("  This explains why you feel you lost money despite flat settled P&L.")
        print("  You are holding losing positions that have not yet settled.")

def main():
    args = parse_args()
    
    client = KalshiClient(API_KEY, KEY_PATH)
    
    # Check current standing first
    check_current_standing(client)
    
    # Parse date filter
    date_filter = None
    if args.date:
        try:
            date_filter = datetime.strptime(args.date, '%Y-%m-%d').date()
            print(f"\n🔍 Filtering to date: {date_filter}")
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return
    
    fills, settled = fetch_data(client, args.days)
    
    # Process data
    positions_df = process_settled_positions(settled)
    fills_df = process_fills(fills)
    
    # Analyze P&L
    analyze_pnl(positions_df, date_filter)
    
    # Detailed trade analysis
    if args.detail or date_filter:
        analyze_trade_characteristics(fills_df, positions_df, date_filter)


if __name__ == "__main__":
    main()
