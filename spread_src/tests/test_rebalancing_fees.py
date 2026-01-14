import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.trading.rebalancing_logic import MultiAssetRebalancer, calculate_kalshi_fee

class TestRebalancingFees(unittest.TestCase):
    def test_fee_calculation(self):
        # 1 contract at 50c
        # 0.0175 * 1 * 0.5 * 0.5 = 0.004375 dollars
        # ceil(0.4375c) = 1c
        self.assertEqual(calculate_kalshi_fee(50, 1), 0.01)
        
        # 10 contracts at 50c
        # 0.0175 * 10 * 0.5 * 0.5 = 0.04375 dollars
        # ceil(4.375c) = 5c
        self.assertEqual(calculate_kalshi_fee(50, 10), 0.05)
        
        # 1 contract at 95c
        # 0.0175 * 1 * 0.95 * 0.05 = 0.00083125 dollars
        # ceil(0.083c) = 1c (Wait, ceil(0.00083125 * 100) = ceil(0.083125) = 1)
        self.assertEqual(calculate_kalshi_fee(95, 1), 0.01)

    def test_guaranteed_harvesting(self):
        rebalancer = MultiAssetRebalancer(bankroll=15.0)
        
        tickers = ["T1"]
        current_weights = {"T1": 0.5} # Big position
        optimal_weights = {"T1": 0.5} # Model thinks we are good
        bids = {"T1": 96}
        asks = {"T1": 98}
        current_positions = {"T1": 10}
        
        # Test Case 1: 1.5% edge (should harvest)
        individual_edges = {"T1": 0.015}
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers, current_weights=current_weights, optimal_weights=optimal_weights,
            bids=bids, asks=asks, individual_edges=individual_edges, current_positions=current_positions
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].reason, "Guaranteed Harvesting")
        
        # Test Case 2: 2.5% edge (should NOT harvest)
        individual_edges = {"T1": 0.025}
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers, current_weights=current_weights, optimal_weights=optimal_weights,
            bids=bids, asks=asks, individual_edges=individual_edges, current_positions=current_positions
        )
        # Should not harvest. It might derisk if bands are hit, but raw weights are 0.5/0.5 so no diff.
        self.assertEqual(len(actions), 0)

if __name__ == '__main__':
    unittest.main()
