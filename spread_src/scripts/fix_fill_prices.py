#!/usr/bin/env python
"""
One-time fix for the fill_price logging bug.

Root cause: order_manager.check_for_fills() stored the NO-side price for SELL YES fills
because Kalshi's API returns side='no' for SELL YES trades (counterparty perspective).
Result: fill_price was stored as (100 - actual_yes_price) for affected trades.

Identifier: side='sell' AND fill_price + order_price ≈ 100 (within 2¢)

Fix applied:
  1. Correct fill_price  → 100 - fill_price
  2. Recalculate realized_pnl using corrected fill_price
     - settled NO (old pnl ≥ 0): revenue = correct_fill * size, cost = 0
     - settled YES (old pnl < 0): revenue = correct_fill * size, cost = 100 * size
  3. Recalculate fee with corrected fill_price

Usage:
    python -m spread_src.scripts.fix_fill_prices [--dry-run]
"""

import sqlite3
import math
import argparse

DB_PATH = 'data/nba_data.db'


def calculate_kalshi_fee(price_cents: float, num_contracts: int) -> float:
    P = price_cents / 100.0
    fee = 0.0175 * num_contracts * P * (1 - P)
    return math.ceil(fee * 100) / 100.0


def find_affected_trades(conn):
    """Find sell trades where order_price + fill_price ≈ 100 (fill logged as NO price)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trade_id, ticker, side, order_price, fill_price, size, realized_pnl, status
        FROM trades
        WHERE side = 'sell'
          AND fill_price IS NOT NULL AND fill_price > 0
          AND order_price IS NOT NULL AND order_price > 0
          AND ABS(order_price + fill_price - 100) <= 2
          AND realized_pnl IS NOT NULL
        ORDER BY trade_id
    """)
    return cursor.fetchall()


def fix_trade(conn, trade, dry_run):
    trade_id, ticker, side, order_price, fill_price, size, old_pnl, status = trade

    correct_fill = 100.0 - fill_price

    # Determine settlement direction from existing pnl
    # settled NO → seller keeps premium → pnl ≥ 0
    # settled YES → seller owes $1/contract → pnl < 0
    settled_yes = (old_pnl < 0)

    revenue = (correct_fill / 100.0) * size
    cost    = 1.0 * size if settled_yes else 0.0
    fee     = calculate_kalshi_fee(correct_fill, size)
    new_pnl = revenue - cost - fee

    old_fee = calculate_kalshi_fee(fill_price, size)

    print(f"  trade {trade_id:>6}  {ticker[-12:]:<12}  "
          f"fill: {fill_price:.0f}→{correct_fill:.0f}¢  "
          f"pnl: ${old_pnl:>+7.2f}→${new_pnl:>+7.2f}  "
          f"({'YES' if settled_yes else 'NO '} settled)")

    if not dry_run:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET fill_price   = ?,
                realized_pnl = ?
            WHERE trade_id = ?
        """, (correct_fill, new_pnl, trade_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without writing to DB')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    trades = find_affected_trades(conn)

    if not trades:
        print("No affected trades found.")
        conn.close()
        return

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(trades)} affected trades\n")

    total_old_pnl = 0.0
    total_new_pnl = 0.0

    for trade in trades:
        trade_id, ticker, side, order_price, fill_price, size, old_pnl, status = trade
        correct_fill = 100.0 - fill_price
        settled_yes  = (old_pnl < 0)
        revenue  = (correct_fill / 100.0) * size
        cost     = 1.0 * size if settled_yes else 0.0
        fee      = calculate_kalshi_fee(correct_fill, size)
        new_pnl  = revenue - cost - fee

        total_old_pnl += old_pnl
        total_new_pnl += new_pnl

        fix_trade(conn, trade, args.dry_run)

    print(f"\n{'='*65}")
    print(f"  Trades affected:    {len(trades)}")
    print(f"  PnL before fix:    ${total_old_pnl:>+.2f}")
    print(f"  PnL after fix:     ${total_new_pnl:>+.2f}")
    print(f"  PnL correction:    ${total_new_pnl - total_old_pnl:>+.2f}")

    if not args.dry_run:
        conn.commit()
        print(f"\n✅ Database updated.")
    else:
        print(f"\n  (dry run — no changes written)")

    conn.close()


if __name__ == '__main__':
    main()
