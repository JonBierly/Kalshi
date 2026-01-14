import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.trading.rebalancing_logic import MultiAssetRebalancer, RebalancingAction

class TestPatientRebalancer(unittest.TestCase):
    def setUp(self):
        # Patient Rebalancer setup: 10% buy, 20% sell, min_spread 8
        self.rebalancer = MultiAssetRebalancer(
            bankroll=100.0, # Easy math
            scale_up_band=0.10,
            derisk_band=0.20,
            min_trade_spread=8
        )
        self.tickers = ["T1"]
        self.individual_edges = {"T1": 0.15} # 15% mid-price edge (high conviction)

    def test_entry_blocked_by_narrow_spread(self):
        """Scale Up should be blocked if spread < 8."""
        current_weights = {"T1": 0.0}
        optimal_weights = {"T1": 0.25} # Wants 25%, diff = 25% > 10% band
        
        # Scenario: Spread is 5c (Blocked)
        bids = {"T1": 45}
        asks = {"T1": 50}
        current_positions = {"T1": 0}
        
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, current_weights, optimal_weights, bids, asks, 
            self.individual_edges, current_positions
        )
        self.assertEqual(len(actions), 0, "Should block entry when spread < 8")

        # Scenario: Spread is 10c (Allowed)
        bids = {"T1": 40}
        asks = {"T1": 50}
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, current_weights, optimal_weights, bids, asks, 
            self.individual_edges, current_positions
        )
        self.assertGreater(len(actions), 0, "Should allow entry when spread >= 8")
        self.assertEqual(actions[0].action, 'buy')

    def test_derisk_blocked_by_narrow_spread(self):
        """De-risking should be blocked if spread < 8 (Hold logic)."""
        current_weights = {"T1": 0.50} # We have a big position
        optimal_weights = {"T1": 0.25} # Wants 25%, diff = -25% which is > 20% sell band
        current_positions = {"T1": 100} # Dummy pos
        
        # Scenario: Spread is 5c (Blocked/Hold)
        # We need model_prob to be LOWER than sell price so there IS edge to sell
        # mid = 47.5, edge = -0.2 -> prob = 0.275
        # sell_price = 49 -> exec_edge = 0.49 - 0.275 = 0.215 (Positive!)
        bids = {"T1": 45}
        asks = {"T1": 50}
        
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, current_weights, optimal_weights, bids, asks, 
            {"T1": -0.20}, current_positions
        )
        self.assertEqual(len(actions), 0, "Should block de-risking when spread < 8 (Stay Strong)")

        # Scenario: Spread is 10c (Allowed)
        # mid = 45, edge = -0.2 -> prob = 0.25
        # sell = 49 -> exec_edge = 49 - 25 = 24 (Positive!)
        bids = {"T1": 40}
        asks = {"T1": 50}
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, current_weights, optimal_weights, bids, asks, 
            {"T1": -0.20}, current_positions
        )
        self.assertGreater(len(actions), 0, "Should allow de-risking when spread >= 8")
        self.assertEqual(actions[0].action, 'sell')

    def test_asymmetric_bands(self):
        """Verify 10% buy and 20% sell bands."""
        current_weights = {"T1": 0.50}
        current_positions = {"T1": 100}
        bids = {"T1": 40}
        asks = {"T1": 50} # Spread 10 (Allowed)
        
        # 1. Test Buy Band (10%)
        # Optimal 0.55 -> diff 5% < 10% band -> No action
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.50}, {"T1": 0.55}, bids, asks, self.individual_edges, current_positions
        )
        self.assertEqual(len(actions), 0)
        
        # Optimal 0.65 -> diff 15% > 10% band -> Buy
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.50}, {"T1": 0.65}, bids, asks, self.individual_edges, current_positions
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, 'buy')

        # 2. Test Sell Band (20%)
        # Optimal 0.35 -> diff -15% < 20% band -> No action
        # Set edge to negative so there is edge to sell (model prob < mid)
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.50}, {"T1": 0.35}, bids, asks, {"T1": -0.20}, current_positions
        )
        self.assertEqual(len(actions), 0)
        
        # Optimal 0.25 -> diff -25% > 20% band -> Sell
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.50}, {"T1": 0.25}, bids, asks, {"T1": -0.20}, current_positions
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, 'sell')

    def test_emergency_bypasses_filter(self):
        """Guaranteed harvesting and toxic exits should bypass the spread filter."""
        current_weights = {"T1": 0.1}
        current_positions = {"T1": 5}
        # Narrow spread (5c)
        bids = {"T1": 95}
        asks = {"T1": 100}
        
        # Optimal 0.1 -> No rebalance diff
        optimal_weights = {"T1": 0.1}
        
        # Scenario: Guaranteed Harvesting (Bid >= 95)
        # Price 95-100, mid is ~97.5, model edge is slightly positive
        # harvesting allows price up to 99, exec_edge = 99 - 98.5 = 0.5% (Positive)
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, current_weights, optimal_weights, bids, asks, 
            {"T1": 0.01}, # small edge
            current_positions
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].reason, "Guaranteed Harvesting")

    def test_hybrid_price_capping(self):
        """Scale Up should cap at 95, but Derisk/Harvest can go to 99."""
        # 1. Scale Up (Buy) at 94-104 spread -> Should be capped at 95c
        # mid = 99, edge=0.05 -> prob = 1.04. buy = 95, edge = 0.09
        bids = {"T1": 94}
        asks = {"T1": 104}
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.0}, {"T1": 0.50}, bids, asks, 
            {"T1": 0.05}, {"T1": 0}
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].price, 95, "Scale up buy should be capped at 95")

        # 2. Derisk (Sell) at 99c market -> Should be allowed at 98c (Ask-1)
        bids = {"T1": 90}
        asks = {"T1": 99}
        actions = self.rebalancer.evaluate_rebalancing(
            self.tickers, {"T1": 0.50}, {"T1": 0.0}, bids, asks, 
            {"T1": -0.05}, {"T1": 100}
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].price, 98, "Derisk sell should be Ask-1 (98)")

if __name__ == '__main__':
    unittest.main()
