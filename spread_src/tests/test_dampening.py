import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.trading.rebalancing_logic import MultiAssetRebalancer

class TestDampening(unittest.TestCase):
    def test_partial_rebalancing(self):
        rebalancer = MultiAssetRebalancer(bankroll=10.0) # $10 bankroll
        
        tickers = ["T1"]
        # Current weight is 0
        current_weights = {"T1": 0.0}
        # Optimal weight is 20% ($2.00)
        optimal_weights = {"T1": 0.2}
        bids = {"T1": 49}
        asks = {"T1": 51}
        # 50c price. $2 target = 4 contracts.
        
        individual_edges = {"T1": 0.10} # 10% edge
        current_positions = {"T1": 0}
        
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges=individual_edges,
            current_positions=current_positions,
            min_scale_up_edge=0.01
        )
        
        # 50% dampening: target weight should be 10% ($1.00)
        # $1.00 at 50c = 2 contracts
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].size, 2)
        self.assertEqual(actions[0].target_pos, 2)
        self.assertEqual(actions[0].reason, "Multi-Asset Scale Up")

    def test_full_harvest_exit(self):
        rebalancer = MultiAssetRebalancer(bankroll=10.0)
        
        tickers = ["T1"]
        current_weights = {"T1": 0.5} 
        optimal_weights = {"T1": 0.5} # Model still liked it
        bids = {"T1": 96}
        asks = {"T1": 98}
        individual_edges = {"T1": 0.01} # Edge fell to 1% (below 2% harvesting)
        current_positions = {"T1": 10}
        
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges=individual_edges,
            current_positions=current_positions
        )
        
        # Should be a FULL EXIT (10 contracts) despite dampening
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].size, 10)
        self.assertEqual(actions[0].target_pos, 0)
        self.assertEqual(actions[0].reason, "Guaranteed Harvesting")

    def test_derisk_band(self):
        rebalancer = MultiAssetRebalancer(bankroll=10.0) # $10 bankroll
        
        tickers = ["T1"]
        # Current weight is 50% ($5.00)
        current_weights = {"T1": 0.5}
        # Optimal weight is 45% ($4.50)
        # Gap is 5%, which is LESS than the 7% derisk band. Should NOT trade.
        optimal_weights = {"T1": 0.45}
        bids = {"T1": 49}
        asks = {"T1": 51}
        
        individual_edges = {"T1": 0.10}
        current_positions = {"T1": 10}
        
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges=individual_edges,
            current_positions=current_positions
        )
        
        self.assertEqual(len(actions), 0)
        
        # Now drop to 40% ($4.00)
        # Gap is 10%, which is MORE than the 7% derisk band.
        optimal_weights = {"T1": 0.40}
        actions = rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            bids=bids,
            asks=asks,
            individual_edges=individual_edges,
            current_positions=current_positions
        )
        
        # Gap is 10%, dampening 50% -> sell 5% ($0.50)
        # $0.50 at 50c = 1 contract
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].size, 1)
        self.assertEqual(actions[0].target_pos, 9)
        self.assertEqual(actions[0].reason, "Multi-Asset De-risk")

if __name__ == '__main__':
    unittest.main()
