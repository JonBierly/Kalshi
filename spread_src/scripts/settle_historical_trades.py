#!/usr/bin/env python
"""
Settle Historical Trades

Retroactively settles all filled trades that haven't been closed yet.
Run this anytime to update P&L for completed markets.

Usage:
    python spread_src/scripts/settle_historical_trades.py
"""

import sys
import os
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient
from spread_src.execution.trade_logger import TradeLogger
from spread_src.utils.kalshi_fees import calculate_kalshi_fee

# Configuration
DB_PATH = 'data/nba_data.db'
API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"


def settle_historical_trades():
    """Settle all filled but unclosed trades."""
    print("=" * 60)
    print("SETTLE HISTORICAL TRADES")
    print("=" * 60)
    
    # Initialize clients
    kalshi = KalshiClient(API_KEY, KEY_PATH)
    logger = TradeLogger(DB_PATH)
    
    # Get unsettled trades
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT trade_id, ticker, side, fill_price, size, created_at
        FROM trades
        WHERE status = 'filled'
        AND closed_at IS NULL
        ORDER BY created_at
    """)
    
    unsettled = cursor.fetchall()
    conn.close()
    
    if not unsettled:
        print("\n✅ No unsettled trades found!")
        return 0
    
    print(f"\n📋 Found {len(unsettled)} unsettled trades")
    print("-" * 60)
    
    settled_count = 0
    total_pnl = 0.0
    
    for trade_id, ticker, side, fill_price, size, created_at in unsettled:
        try:
            # Query Kalshi for market status
            market = kalshi.get_market_details(ticker)
            
            if not market:
                print(f"  ⚠️  {ticker[-15:]}: Market not found")
                continue
            
            status = market.get('status')
            
            if status not in ['finalized', 'settled']:
                print(f"  ⏳ {ticker[-15:]}: Still active ({status})")
                continue
            
            result = market.get('result')
            if not result:
                print(f"  ⚠️  {ticker[-15:]}: No result yet")
                continue
            
            is_yes_result = (result.lower() == 'yes')
            
            # Calculate P&L
            if side == 'buy':
                cost = (fill_price / 100.0) * size
                payout = 1.0 * size if is_yes_result else 0.0
                pnl_before_fee = payout - cost
            elif side == 'sell':
                revenue = (fill_price / 100.0) * size
                cost = 1.0 * size if is_yes_result else 0.0
                pnl_before_fee = revenue - cost
            else:
                continue
            
            # Deduct fee
            fee = calculate_kalshi_fee(fill_price, size)
            realized_pnl = pnl_before_fee - fee
            
            # Update database
            logger.log_position_closed(trade_id, realized_pnl)
            
            settled_count += 1
            total_pnl += realized_pnl
            
            result_emoji = "✅" if realized_pnl >= 0 else "❌"
            print(f"  {result_emoji} {ticker[-15:]}: {side.upper()} {size} @ {fill_price:.0f}¢ → ${realized_pnl:+.2f}")
            
        except Exception as e:
            print(f"  ⚠️  {ticker[-15:]}: Error - {e}")
            continue
    
    print("-" * 60)
    print(f"\n📊 SUMMARY")
    print(f"  Settled: {settled_count}/{len(unsettled)} trades")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    
    return settled_count


if __name__ == "__main__":
    settle_historical_trades()
