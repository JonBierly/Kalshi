#!/usr/bin/env python
"""
Debug P&L discrepancy between our calculation and Kalshi.
"""

import sqlite3
from datetime import datetime

DB_PATH = 'data/nba_data.db'

def analyze_pnl_calculation():
    """Check our P&L calculations step by step."""
    print("=" * 60)
    print("P&L CALCULATION DEBUG")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get Dec 5 trades
    cursor.execute("""
        SELECT 
            trade_id,
            ticker,
            side,
            fill_price,
            size,
            position_after,
            realized_pnl,
            datetime(filled_at) as filled_at
        FROM trades
        WHERE DATE(filled_at) = '2025-12-05'
        AND fill_price IS NOT NULL
        ORDER BY filled_at
    """)
    
    trades = cursor.fetchall()
    print(f"\nAnalyzing {len(trades)} trades from Dec 5...")
    
    total_our_pnl = 0
    
    # Show first 10 trades with calculation breakdown
    print("\n📊 Sample Trade Calculations (first 10):")
    print("-" * 60)
    
    for i, (trade_id, ticker, side, fill_price, size, pos_after, pnl, filled_at) in enumerate(trades[:10]):
        total_our_pnl += pnl if pnl else 0
        
        print(f"\nTrade {trade_id}: {ticker[-6:]}")
        print(f"  Side: {side}, Size: {size}, Fill: {fill_price}¢")
        print(f"  Position After: {pos_after:+d}")
        print(f"  Our P&L: ${pnl:.2f}" if pnl else "  Our P&L: Not settled")
    
    # Total P&L
    cursor.execute("""
        SELECT SUM(realized_pnl)
        FROM trades
        WHERE DATE(filled_at) = '2025-12-05'
        AND fill_price IS NOT NULL
    """)
    total_pnl = cursor.fetchone()[0]
    
    print("\n" + "=" * 60)
    print("TOTALS:")
    print(f"  Our Calculated P&L: ${total_pnl:.2f}")
    print(f"  Number of Fills: {len(trades)}")
    
    # Calculate total contracts traded (for fee estimation)
    cursor.execute("""
        SELECT SUM(ABS(position_after))
        FROM trades
        WHERE DATE(filled_at) = '2025-12-05'
        AND fill_price IS NOT NULL
    """)
    total_contracts = cursor.fetchone()[0]
    
    estimated_fees = total_contracts * 0.01 if total_contracts else 0
    
    print(f"\n💡 Analysis:")
    print(f"  Total Contracts: {total_contracts}")
    print(f"  Est. Kalshi Fees ($0.01/contract): ${estimated_fees:.2f}")
    print(f"  Expected Net P&L: ${total_pnl - estimated_fees:.2f}")
    print(f"\n  Kalshi Reports: ~$6.00")
    print(f"  Discrepancy: ${total_pnl - estimated_fees - 6:.2f}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("POSSIBLE ISSUES:")
    print("1. Position_after might not be accurate")
    print("2. We might be double-counting some trades")
    print("3. Settlement outcomes might be wrong")
    print("4. Kalshi has additional fees we're not accounting for")
    print("=" * 60)

if __name__ == "__main__":
    analyze_pnl_calculation()
