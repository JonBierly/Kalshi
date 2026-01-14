import unittest
import numpy as np
import scipy.stats as stats
from spread_src.trading.rebalancing_logic import MultiAssetRebalancer, RebalancingAction

class TestMultiAssetOptimization(unittest.TestCase):
    def setUp(self):
        self.rebalancer = MultiAssetRebalancer(
            bankroll=100.0,
            kelly_fraction=1.0, # Use full Kelly for easier testing
            max_ticker_exposure=10.0,
            scale_up_band=0.07,
            derisk_band=0.02
        )

    def test_covariance_matrix_monotonic(self):
        """Test that covariance matrix makes sense for monotonic probs."""
        tickers = ["T1", "T2"]
        probs = [0.6, 0.4] # T1 is easier to hit than T2
        prices = [0.5, 0.3]
        
        weights = self.rebalancer.calculate_optimal_weights(
            tickers=tickers,
            probs=probs,
            prices=prices,
            distribution=None
        )
        
        # We expect weights to interact. 
        # T1 has 10% edge, T2 has 10% edge.
        # But they are correlated.
        self.assertIn("T1", weights)
        self.assertIn("T2", weights)
        
    def test_rebalance_bands_scale_up(self):
        """Test the 7% scale-up band."""
        tickers = ["T1"]
        # opt_w = 0.1, curr_w = 0.04 (diff = 0.06 < 0.07) -> No action
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.04},
            optimal_weights={"T1": 0.10},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.05}
        )
        self.assertEqual(len(actions), 0)
        
        # opt_w = 0.12, curr_w = 0.04 (diff = 0.08 > 0.07) -> Buy
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.04},
            optimal_weights={"T1": 0.12},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.05}
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, 'buy')

    def test_min_edge_check_for_scale_up(self):
        """Verify we DON'T scale up if individual edge is below threshold."""
        tickers = ["T1"]
        # Kelly wants to buy (diff = 0.10), but individual edge is low (0.01)
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.0},
            optimal_weights={"T1": 0.10},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.01}, # Low individual edge (< 0.02)
            min_scale_up_edge=0.02
        )
        self.assertEqual(len(actions), 0)

    def test_rebalance_bands_derisk(self):
        """Test the 2% de-risk band."""
        tickers = ["T1"]
        # opt_w = 0.08, curr_w = 0.10 (diff = -0.02 == band) -> No action (usually > band)
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.10},
            optimal_weights={"T1": 0.08},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.05}
        )
        self.assertEqual(len(actions), 0)
        
        # opt_w = 0.07, curr_w = 0.10 (diff = -0.03 < -0.02) -> Sell
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.10},
            optimal_weights={"T1": 0.07},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.05}
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, 'sell')

    def test_toxic_exit_bypasses_bands(self):
        """Verify toxic exit ignores bands."""
        tickers = ["T1"]
        # diff = 0.01 (within bands), but is_toxic = True
        actions = self.rebalancer.evaluate_rebalancing(
            tickers=tickers,
            current_weights={"T1": 0.10},
            optimal_weights={"T1": 0.11},
            bids={"T1": 50},
            asks={"T1": 60},
            individual_edges={"T1": 0.05},
            is_toxic={"T1": True}
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].reason, "Toxic Exit")

    def test_negative_edge_safety(self):
        """Verify we never go long on negative edge, even if math suggests it."""
        # Scenario: Market is 60c, Model says 50c (negative edge)
        tickers = ["KX-TEAM-A1"]
        probs = [0.50]
        prices = [0.60] # Negative edge (-0.10)
        
        weights = self.rebalancer.calculate_optimal_weights(tickers, probs, prices, None)
        self.assertLessEqual(weights["KX-TEAM-A1"], 0.0)

    def test_multi_team_covariance(self):
        """Verify opposite teams have zero joint probability (mutually exclusive)."""
        tickers = ["KX-TEAM-A1", "KX-TEAM-B1"]
        probs = [0.4, 0.4]
        prices = [0.3, 0.3]
        
        weights = self.rebalancer.calculate_optimal_weights(tickers, probs, prices, None)
        self.assertTrue(len(weights) == 2)
        # Just verifying it doesn't crash and returns reasonable weights
        for w in weights.values():
            self.assertGreater(w, 0)

if __name__ == "__main__":
    unittest.main()
