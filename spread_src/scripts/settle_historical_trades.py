#!/usr/bin/env python
"""
Settle Historical Trades

Uses the Kalshi settlement API to update P&L for all trades in the database.
Uses the correct P&L formula: payout = winning_count × 100, pnl = (payout - costs) / 100 - fees

Usage:
    python spread_src/scripts/settle_historical_trades.py
    python spread_src/scripts/settle_historical_trades.py --force  # Recalculate all trades
"""

import sys
import os
import sqlite3
import pytz
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient
from spread_src.execution.trade_logger import TradeLogger

# Configuration
DB_PATH = 'data/nba_data.db'
API_KEY = os.environ.get("KALSHI_KEY_ID", "a40ff1c6-12ac-4a6c-9669-ffe12f3de235")
KEY_PATH = os.environ.get("KALSHI_KEY_PATH", "key.key")


def get_settlements_pnl(kalshi):
    """
    Fetch all settlements and calculate correct P&L per ticker.
    Iterates backwards in time to cover all historical trades.
    """
    all_settlements = []
    
    # We need to cover from Dec 12, 2025 to now
    start_date = datetime(2025, 12, 11) # A bit before the first trade
    end_date = datetime.now()
    
    current_end = end_date
    while current_end > start_date:
        current_start = current_end - timedelta(days=7)
        print(f"  📥 Fetching settlements from {current_start.date()} to {current_end.date()}...")
        
        min_ts = int(current_start.timestamp() * 1000)
        max_ts = int(current_end.timestamp() * 1000)
        
        chunk = kalshi.get_settlements(min_ts=min_ts, max_ts=max_ts, limit=1000)
        if not chunk:
            # If no settlements in this week, just keep going back
            pass
        else:
            all_settlements.extend(chunk)
            
        current_end = current_start
        # Safety break - the API only returns settled things, 
        # but don't go back forever
        if current_end < datetime(2025, 11, 1):
            break

    ticker_pnl = {}
    seen_tickers = set()
    
    for s in all_settlements:
        ticker = s.get('ticker', '')
        if 'KXNBASPREAD' not in ticker:
            continue
        
        # Avoid duplicate tickers from overlapping windows
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        
        yes_count = s.get('yes_count', 0)
        no_count = s.get('no_count', 0)
        yes_cost = s.get('yes_total_cost', 0)  # in cents
        no_cost = s.get('no_total_cost', 0)    # in cents
        market_result = s.get('market_result', '')
        fees = float(s.get('fee_cost', '0'))
        
        # CORRECT P&L FORMULA
        if market_result == 'yes':
            payout_cents = yes_count * 100
        else:  # 'no'
            payout_cents = no_count * 100
        
        pnl_cents = payout_cents - yes_cost - no_cost
        pnl_dollars = pnl_cents / 100.0 - fees
        
        ticker_pnl[ticker] = {
            'pnl': pnl_dollars,
            'fees': fees,
            'market_result': market_result
        }
    
    return ticker_pnl


def settle_historical_trades(force=False):
    """
    Settle all trades using settlement API P&L.
    
    Args:
        force: If True, recalculate ALL trades (even already settled ones)
    """
    print("=" * 60)
    print("SETTLE HISTORICAL TRADES (Using Settlement API)")
    print("=" * 60)
    
    # Initialize clients
    kalshi = KalshiClient(API_KEY, KEY_PATH)
    logger = TradeLogger(DB_PATH)
    
    # Fetch settlement P&L from API
    print("\n📥 Fetching settlements from Kalshi API...")
    ticker_pnl = get_settlements_pnl(kalshi)
    print(f"  ✓ Got P&L for {len(ticker_pnl)} tickers")
    
    # Get trades to update
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if force:
        # Get ALL filled trades
        cursor.execute("""
            SELECT trade_id, ticker, side, fill_price, size, status
            FROM trades
            WHERE status IN ('filled', 'settled', 'closed')
            ORDER BY created_at
        """)
        trades = cursor.fetchall()
        print(f"\n📋 Found {len(trades)} total filled trades (force recalculating all)")
    else:
        # Get only unsettled trades
        cursor.execute("""
            SELECT trade_id, ticker, side, fill_price, size, status
            FROM trades
            WHERE status = 'filled'
            AND closed_at IS NULL
            ORDER BY created_at
        """)
        trades = cursor.fetchall()
        print(f"\n📋 Found {len(trades)} unsettled trades")
    
    conn.close()
    
    if not trades:
        print("\n✅ No trades to settle!")
        return 0
    
    print("-" * 60)
    
    # Group trades by ticker
    trades_by_ticker = {}
    for trade_id, ticker, side, fill_price, size, status in trades:
        if ticker not in trades_by_ticker:
            trades_by_ticker[ticker] = []
        trades_by_ticker[ticker].append({
            'trade_id': trade_id,
            'side': side,
            'fill_price': fill_price,
            'size': size,
            'status': status
        })
    
    # Get trade outcomes and calculate P&L per trade
    settled_count = 0
    total_pnl = 0.0
    
    from spread_src.utils.kalshi_fees import calculate_kalshi_fee

    for ticker, trade_list in trades_by_ticker.items():
        if ticker not in ticker_pnl:
            # Market hasn't settled yet
            print(f"  ⏳ {ticker[-15:]}: Not settled yet ({len(trade_list)} trades)")
            continue
        
        # Get outcome for this ticker
        pnl_info = ticker_pnl[ticker]
        market_result = pnl_info['market_result'] # 'yes' or 'no'
        is_yes_win = (market_result.lower() == 'yes')
        
        for trade in trade_list:
            trade_id = trade['trade_id']
            side = trade['side'].lower()
            fill_price = trade['fill_price']
            size = trade['size']
            
            # OutcomeValue of the YES contract (100 if it wins, 0 otherwise)
            outcome_value = 100 if is_yes_win else 0
            
            # 1. Calculate Gross P&L
            if side == 'buy':
                # Long YES: P&L = (FinalValue - PurchasePrice) * Size
                gross_pnl_cents = (outcome_value - fill_price) * size
            else:
                # Sell YES (Short): P&L = (SalePrice - FinalValue) * Size
                gross_pnl_cents = (fill_price - outcome_value) * size
                
            # 2. Calculate Fee
            fee = calculate_kalshi_fee(fill_price, size)
            
            # 3. Final Realized P&L in Dollars
            real_pnl = (gross_pnl_cents / 100.0) - fee
            
            # Update database
            logger.log_position_closed(trade_id, real_pnl)
            
            settled_count += 1
            total_pnl += real_pnl
        
        ticker_win = (pnl_info['pnl'] >= 0) # Aggregate ticker was profitable
        result_emoji = "✅" if ticker_win else "❌"
        print(f"  {result_emoji} {ticker[-15:]}: Settled {len(trade_list)} trades | Result: {market_result.upper()}")
    
    print("-" * 60)
    print(f"\n📊 SUMMARY")
    print(f"  Trades updated: {settled_count}")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    
    return settled_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Recalculate ALL trades')
    args = parser.parse_args()
    
    settle_historical_trades(force=args.force)
