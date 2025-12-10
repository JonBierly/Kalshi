#!/usr/bin/env python
"""Quick analysis of Dec 9 trades."""

import sys
import os
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.kalshi import KalshiClient

API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"

client = KalshiClient(API_KEY, KEY_PATH)

# Get settled positions
settled = client.get_positions(settlement_status="settled", limit=1000)
market_positions = settled.get("market_positions", [])

# Filter for Dec 9 games
dec9_positions = []
for pos in market_positions:
    ticker = pos.get("ticker", "")
    match = re.search(r"-25DEC09", ticker)
    if match:
        dec9_positions.append(pos)

print(f"Dec 9 positions: {len(dec9_positions)}")
print()

# Show details
total_pnl = 0
total_fees = 0
for pos in dec9_positions:
    ticker = pos.get("ticker")
    pnl = pos.get("realized_pnl", 0) / 100.0
    fees = pos.get("fees_paid", 0) / 100.0
    total_pnl += pnl
    total_fees += fees
    print(f"{ticker}: PnL=${pnl:.2f}, Fees=${fees:.2f}")

print()
print(f"Total Dec 9 PnL: ${total_pnl:.2f}")
print(f"Total Dec 9 Fees: ${total_fees:.2f}")
