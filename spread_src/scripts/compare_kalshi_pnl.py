#!/usr/bin/env python
"""
Compare our P&L calculations with Kalshi's actual settled positions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.kalshi import KalshiClient
from spread_src.utils.kalshi_fees import calculate_kalshi_fee
import sqlite3
from datetime import datetime

def compare_with_kalshi():
    """Fetch Kalshi's actual P&L and compare with our calculations."""
    print("\n" + "=" * 60)
    print("📊 KALSHI P&L COMPARISON")
    print("=" * 60)
    
    # Initialize Kalshi client
    kalshi = KalshiClient("your_key_id", "key.key")
    
    # Get settled positions from Kalshi
    print("\n📥 Fetching settled positions from Kalshi...")
    try:
        settled_positions = kalshi.get_positions(settlement_status='settled')
        print(f"✓ Found {len(settled_positions)} settled positions")
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        return
    
    # Filter to Dec 5 positions
    dec5_positions = []
    for pos in settled_positions:
        # Check if position was opened/closed on Dec 5
        if '2025-12-05' in pos.get('last_update_time', ''):
            dec5_positions.append(pos)
    
    print(f"✓ {len(dec5_positions)} positions from Dec 5")
    
    # Calculate Kalshi's reported P&L
    kalshi_pnl = 0.0
    kalshi_fees = 0.0
    
    print("\n📋 Kalshi's Settled Positions:")
    print("-" * 60)
    
    for i, pos in enumerate(dec5_positions[:10]):  # Show first 10
        ticker = pos.get('ticker', 'Unknown')
        quantity = pos.get('total_traded', 0)
        realized_pnl = pos.get('realized_pnl', 0) / 100.0  # Convert cents to dollars
        
        kalshi_pnl += realized_pnl
        
        print(f"{i+1}. {ticker[-10:]}: {quantity} contracts → ${realized_pnl:+.2f}")
    
    if len(dec5_positions) > 10:
        print(f"... and {len(dec5_positions) - 10} more")
    
    # Get our calculations
    conn = sqlite3.connect('data/nba_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ticker,
            side,
            fill_price,
            size,
            realized_pnl
        FROM trades
        WHERE DATE(filled_at) = '2025-12-05'
        AND fill_price IS NOT NULL
        AND closed_at IS NOT NULL
    """)
    
    our_trades = cursor.fetchall()
    
    # Calculate fees for our trades
    our_pnl_before_fees = sum(pnl for _, _, _, _, pnl in our_trades if pnl)
    our_total_fees = 0.0
    
    print("\n📋 Our Calculated Fees:")
    print("-" * 60)
    
    for ticker, side, fill_price, size, pnl in our_trades[:10]:
        fee = calculate_kalshi_fee(fill_price, size)
        our_total_fees += fee
        print(f"{ticker[-10:]}: {size} @ {fill_price:.0f}¢ → Fee: ${fee:.2f}")
    
    if len(our_trades) > 10:
        # Calculate remaining fees
        for ticker, side, fill_price, size, pnl in our_trades[10:]:
            fee = calculate_kalshi_fee(fill_price, size)
            our_total_fees += fee
        print(f"... and {len(our_trades) - 10} more trades")
    
    our_pnl_after_fees = our_pnl_before_fees - our_total_fees
    
    conn.close()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 COMPARISON")
    print("=" * 60)
    
    print(f"\n🔹 Our Calculations:")
    print(f"  P&L Before Fees: ${our_pnl_before_fees:+.2f}")
    print(f"  Total Fees: ${our_total_fees:.2f}")
    print(f"  P&L After Fees: ${our_pnl_after_fees:+.2f}")
    
    print(f"\n🔹 Kalshi Reports:")
    print(f"  Total P&L: ${kalshi_pnl:+.2f}")
    
    print(f"\n🔹 Difference:")
    difference = our_pnl_after_fees - kalshi_pnl
    print(f"  ${difference:+.2f}")
    
    if abs(difference) < 1.0:
        print("  ✅ Very close! Likely rounding differences.")
    elif abs(difference) < 3.0:
        print("  ⚠️  Small discrepancy - possibly fee calculation or timing.")
    else:
        print("  ❌ Large discrepancy - investigate further!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    compare_with_kalshi()
