#!/usr/bin/env python
"""
Settle all open positions by checking Kalshi for final results.

Run this script to manually settle positions after games complete.
"""

import sys
import os
from datetime import datetime
import sqlite3
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.kalshi import KalshiClient
from spread_src.execution.trade_logger import TradeLogger


def calculate_kalshi_fee(price_cents: float, num_contracts: int) -> float:
    """
    Calculate Kalshi trading fee: round_up(0.0175 × C × P × (1-P))
    
    Args:
        price_cents: Fill price in cents
        num_contracts: Number of contracts
        
    Returns:
        Fee in dollars
    """
    P = price_cents / 100.0
    C = num_contracts
    fee = 0.0175 * C * P * (1 - P)
    return math.ceil(fee * 100) / 100.0




def settle_all_positions(dry_run=False):
    """
    Check all filled trades and settle those with finalized markets.
    
    Args:
        dry_run: If True, don't update database, just show what would happen
    """
    print("\n" + "=" * 60)
    print("🏁 POSITION SETTLEMENT SCRIPT")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Time: {datetime.now()}")
    
    # Initialize clients
    kalshi = KalshiClient("your_key_id", "key.key")  # Replace with actual path
    logger = TradeLogger('data/nba_data.db')
    
    # Get all filled but not closed trades from database
    conn = sqlite3.connect('data/nba_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT trade_id, ticker, side, fill_price, size, position_after
        FROM trades
        WHERE status = 'filled'
        AND closed_at IS NULL
        ORDER BY trade_id
    """)
    
    open_trades = cursor.fetchall()
    print(f"\n📊 Found {len(open_trades)} open positions to check...")
    
    settled_count = 0
    total_pnl = 0.0
    
    for trade_id, ticker, side, fill_price, size, position_after in open_trades:
        try:
            # Query Kalshi for market status
            market = kalshi.get_market_details(ticker)
            
            if not market:
                print(f"⚠️  {ticker}: Could not fetch market details")
                continue
            
            status = market.get('status')
            
            if status in ['finalized', 'settled']:  # Kalshi uses 'finalized'
                result = market.get('result')  # 'yes' or 'no'
                
                if not result:
                    print(f"⚠️  {ticker}: Settled but no result")
                    continue
                
                # Calculate P&L
                # For YES contracts:
                # - If result is YES: contract pays $1.00
                # - If result is NO: contract pays $0.00
                
                # Position_after tells us net position after this fill
                # If we bought (side='buy'), we're long
                # If we sold (side='sell'), we're short
                
                is_yes_result = (result.lower() == 'yes')
                
                # CRITICAL FIX: Use 'size' (this trade), not 'position_after' (cumulative)!
                # Calculate P&L for THIS specific trade only
                
                if side == 'buy':  # We bought YES contracts
                    # Cost: what we paid
                    cost = (fill_price / 100.0) * size
                    # Payout: $1 if YES, $0 if NO
                    payout = 1.0 * size if is_yes_result else 0.0
                    realized_pnl = payout - cost
                    
                elif side == 'sell':  # We sold YES contracts (went short)
                    # Revenue: what we received
                    revenue = (fill_price / 100.0) * size
                    # Cost: $1 per contract if YES, $0 if NO
                    cost = 1.0 * size if is_yes_result else 0.0
                    realized_pnl = revenue - cost
                    
                else:
                    realized_pnl = 0.0
                
                # Deduct Kalshi trading fee
                fee = calculate_kalshi_fee(fill_price, size)
                realized_pnl_after_fees = realized_pnl - fee
                
                print(f"\n🏁 {ticker}")
                print(f"   Result: {result.upper()}")
                print(f"   Trade: {size} @ {fill_price:.1f}¢ ({side})")
                print(f"   P&L Before Fee: ${realized_pnl:+.2f}")
                print(f"   Kalshi Fee: -${fee:.2f}")
                print(f"   Net P&L: ${realized_pnl_after_fees:+.2f}")
                
                if not dry_run:
                    # Update database with net P&L (after fees)
                    logger.log_position_closed(
                        trade_id=trade_id,
                        realized_pnl=realized_pnl_after_fees
                    )
                    settled_count += 1
                    total_pnl += realized_pnl_after_fees
                else:
                    settled_count += 1
                    total_pnl += realized_pnl_after_fees
                    
            elif status in ['active', 'open']:
                print(f"⏳ {ticker}: Still active")
            else:
                print(f"❓ {ticker}: Unknown status '{status}'")
                
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue
    
    conn.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETTLEMENT SUMMARY")
    print("=" * 60)
    print(f"Checked: {len(open_trades)} positions")
    print(f"Settled: {settled_count} positions")
    print(f"Total P&L: ${total_pnl:+.2f}")
    
    if dry_run:
        print("\n⚠️  DRY RUN - No changes made to database")
        print("Run with --live to actually settle positions")
    else:
        print("\n✅ Database updated")
    
    return settled_count, total_pnl


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Settle open positions')
    parser.add_argument('--live', action='store_true', 
                      help='Actually update database (default is dry-run)')
    args = parser.parse_args()
    
    settle_all_positions(dry_run=not args.live)


if __name__ == "__main__":
    main()
