#!/usr/bin/env python
"""
Calculate Kalshi trading fees using their actual formula.

Formula: fees = round_up(0.0175 × C × P × (1-P))
- P = price in dollars (50¢ = 0.50)
- C = number of contracts
- round_up = rounds to next cent
"""

import math

def calculate_kalshi_fee(price_cents: float, num_contracts: int) -> float:
    """
    Calculate Kalshi trading fee for a fill.
    
    Args:
        price_cents: Fill price in cents (e.g., 70.0 for 70¢)
        num_contracts: Number of contracts traded
        
    Returns:
        Fee in dollars (e.g., 0.02 for 2¢)
    """
    # Convert to dollars
    P = price_cents / 100.0
    C = num_contracts
    
    # Calculate fee
    fee = 0.0175 * C * P * (1 - P)
    
    # Round up to next cent
    fee_rounded = math.ceil(fee * 100) / 100.0
    
    return fee_rounded


def test_fees():
    """Test fee calculations with examples."""
    print("Kalshi Fee Calculator Test")
    print("=" * 60)
    
    test_cases = [
        (50, 1),   # 1 @ 50¢ (max volatility)
        (70, 3),   # 3 @ 70¢
        (24, 10),  # 10 @ 24¢
        (90, 5),   # 5 @ 90¢ (low volatility)
        (10, 10),  # 10 @ 10¢ (low volatility)
    ]
    
    for price, contracts in test_cases:
        fee = calculate_kalshi_fee(price, contracts)
        print(f"{contracts} contracts @ {price}¢ → Fee: ${fee:.2f}")


if __name__ == "__main__":
    test_fees()
