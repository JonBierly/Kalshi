#!/usr/bin/env python
"""
Fetch actual trading data from Kalshi API to compare with our database.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.kalshi import KalshiClient
import json
from datetime import datetime

def fetch_kalshi_data():
    """Fetch fills, positions, and balance from Kalshi."""
    print("\n" + "=" * 60)
    print("📡 FETCHING KALSHI TRADING DATA")
    print("=" * 60)
    kalshi = KalshiClient("a40ff1c6-12ac-4a6c-9669-ffe12f3de235", "key.key")
    
    # 1. Get account balance
    print("\n💰 Account Balance:")
    try:
        balance = kalshi.get_balance()
        print(f"  Balance: ${balance.get('balance', 0) / 100:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. Get recent fills
    print("\n📋 Recent Fills:")
    try:
        fills = kalshi.get_fills(limit=100)
        print(f"  Total fills fetched: {len(fills)}")
        
        # Filter to Dec 5
        dec5_fills = [f for f in fills if '2025-12-05' in f.get('created_time', '')]
        print(f"  Dec 5 fills: {len(dec5_fills)}")
        
        if dec5_fills:
            print(f"\n  First 5 Dec 5 fills:")
            for i, fill in enumerate(dec5_fills[:5]):
                ticker = fill.get('ticker', 'Unknown')
                side = fill.get('action', 'unknown')  # 'buy' or 'sell'
                price = fill.get('yes_price', 0) / 100.0
                count = fill.get('count', 0)
                created = fill.get('created_time', '')
                
                print(f"  {i+1}. {ticker[-10:]}: {side} {count} @ ${price:.2f} - {created[:19]}")
        
        # Calculate total from fills
        total_bought = sum(f.get('count', 0) * f.get('yes_price', 0) / 100.0 
                          for f in dec5_fills if f.get('action') == 'buy')
        total_sold = sum(f.get('count', 0) * f.get('yes_price', 0) / 100.0 
                        for f in dec5_fills if f.get('action') == 'sell')
        
        print(f"\n  Dec 5 Trading Volume:")
        print(f"    Total Bought: ${total_bought:.2f}")
        print(f"    Total Sold: ${total_sold:.2f}")
        
    except Exception as e:
        print(f"  Error fetching fills: {e}")
    
    # 3. Get positions (settled)
    print("\n📊 Settled Positions:")
    try:
        settled = kalshi.get_positions(settlement_status='settled', limit=100)
        
        # Kalshi returns positions differently - let's see what we get
        print(f"  Type of settled data: {type(settled)}")
        
        if isinstance(settled, list):
            print(f"  Total settled positions: {len(settled)}")
            
            # Try to filter to Dec 5
            dec5_settled = []
            for pos in settled:
                if isinstance(pos, dict):
                    # Check various time fields
                    time_str = (pos.get('last_update_time', '') or 
                              pos.get('created_time', '') or 
                              pos.get('closed_time', ''))
                    if '2025-12-05' in time_str:
                        dec5_settled.append(pos)
            
            print(f"  Dec 5 settled: {len(dec5_settled)}")
            
            if dec5_settled:
                print(f"\n  Sample settled positions:")
                for i, pos in enumerate(dec5_settled[:5]):
                    ticker = pos.get('ticker', 'Unknown')
                    total_cost = pos.get('total_cost', 0) / 100.0
                    realized_pnl = pos.get('realized_pnl', 0) / 100.0
                    
                    print(f"  {i+1}. {ticker[-10:]}: Cost ${total_cost:.2f}, P&L ${realized_pnl:+.2f}")
                
                # Calculate total P&L from Kalshi
                total_kalshi_pnl = sum(p.get('realized_pnl', 0) / 100.0 for p in dec5_settled)
                print(f"\n  📊 Total Dec 5 P&L (Kalshi): ${total_kalshi_pnl:+.2f}")
        else:
            print(f"  Unexpected format: {settled}")
            
    except Exception as e:
        print(f"  Error fetching positions: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Get portfolio events
    print("\n📅 Portfolio Events:")
    try:
        # This might give us settlement info
        events = kalshi.get_portfolio_events(limit=100)
        print(f"  Type: {type(events)}")
        if isinstance(events, list):
            print(f"  Total events: {len(events)}")
            
            # Show first few
            for i, event in enumerate(events[:3]):
                print(f"  {i+1}. {json.dumps(event, indent=4)}")
    except Exception as e:
        print(f"  Error (might not exist): {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    fetch_kalshi_data()
