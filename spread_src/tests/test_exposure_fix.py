import unittest
import numpy as np
from spread_src.trading.rebalancing_logic import MultiAssetRebalancer, RebalancingAction

class TestExposureFix(unittest.TestCase):
    def setUp(self):
        # Setup rebalancer with a $15 bankroll (like the user's config)
        self.rebalancer = MultiAssetRebalancer(
            bankroll=15.0,
            kelly_fraction=1.0
        )

    def test_long_to_short_transition_low_price(self):
        """
        GIVEN a long position (e.g. +11 contracts)
        WHEN the model target flips to short (e.g. -33% weight) at a low price (e.g. 1c)
        VERIFY the size calculation is reasonable.
        
        Old bug: (abs(-0.33 - 0.10) * 15) / 0.01 = 645 contracts
        New logic: 
          Target Contracts = -int(0.33 * 15 / 0.99) = -int(4.95/0.99) = -5 contracts
          Size = abs(-5 - 11) = 16 contracts 
        """
        ticker = "T1"
        # Current: 11 contracts @ 17c -> ~1.87 exposure -> ~12% weight
        current_pos = {ticker: 11}
        current_weights = {ticker: 0.12} 
        
        # Target: -33.3% weight
        optimal_weights = {ticker: -0.333}
        
        # Prices: Market is really low (shorting is expensive, buying is cheap)
        bids = {ticker: 1}
        asks = {ticker: 2}
        
        # Even if edge is high, let's see the size
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=[ticker],
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges={ticker: 0.10},
            current_positions=current_pos
        )
        
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action.action, 'sell')
        
        # Size should be around 16 (+11 to 0, then 0 to -5)
        # Target contracts = -int(0.333 * 15 / ( (100-1)/100 )) = -int(4.995 / 0.99) = -5
        # size = abs(-5 - 11) = 16
        print(f"Calculated Size: {action.size}")
        self.assertLess(action.size, 50) # Definitely not 600+
        self.assertEqual(action.size, 16)

    def test_short_to_long_transition_low_price(self):
        """
        Inverse case: Short to Long transition.
        """
        ticker = "T1"
        # Current: -5 contracts @ 1c (exposure cost 99c) -> ~4.95 exposure -> ~-33% weight
        current_pos = {ticker: -5}
        current_weights = {ticker: -0.33} 
        
        # Target: +33% weight
        optimal_weights = {ticker: 0.33}
        
        # Price is low (1c)
        bids = {ticker: 1}
        asks = {ticker: 2}
        
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=[ticker],
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges={ticker: 0.10},
            current_positions=current_pos
        )
        
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action.action, 'buy')
        
        # Target contracts = int(0.33 * 15 / 0.02) = int(4.95 / 0.02) = 247 contracts (since price is so low)
        # This is high but CORRECT because at 2c, you can buy many YES contracts.
        # But it's still way more stable than the previous logic which might have done crazy division.
        print(f"Calculated Size (S->L): {action.size}")
        self.assertGreater(action.size, 200)

if __name__ == "__main__":
    unittest.main()
