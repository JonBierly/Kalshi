import unittest
from spread_src.trading.rebalancing_logic import RebalancingTrader
from spread_src.execution.portfolio import Portfolio
from unittest.mock import MagicMock
from datetime import datetime

class TestRebalancingFix(unittest.TestCase):
    def setUp(self):
        self.trader = RebalancingTrader(
            bankroll=10.0,
            kelly_fraction=0.25,
            min_edge=0.05,
            min_harvest_edge=0.01,
            hysteresis_buffer=2
        )
        self.portfolio = Portfolio(max_exposure=100.0)

    def test_no_rebalance_on_negative_edge(self):
        """
        Verify that we don't rebalance if the edge is negative, 
        even if we are far from the target.
        """
        # Scenario: Current pos is 50, Target is 0 (fair value crashed).
        # Market is 5-10 cents. 
        # Fair value is 1 cent.
        # sell_price = 10 - 1 = 9 cents.
        # sell_edge = (9 - fair) / 100 = (9 - 1) / 100 = 0.08 (positive) -> Rebalance
        
        # Scenario: Current pos is 50, Target is 0.
        # Market is 50-60 cents.
        # Fair value is 65 cents (we are long, but fair is above ask!).
        # sell_price = 60 - 1 = 59 cents.
        # sell_edge = (59 - 65) / 100 = -0.06 (negative) -> SHOULD NOT REBALANCE
        
        actions = self.trader.evaluate_rebalancing(
            ticker="TEST",
            current_pos=50,
            cost_basis=50.0,
            model_prob=0.65, # Fair = 65
            ci_lower=0.60,
            ci_upper=0.70, # CI upper = 70
            bid=50,
            ask=60
        )
        
        # sell_price = 59. sell_edge = (59 - 70) / 100 = -0.11.
        # Even though target might be lower, edge is negative to sell.
        for a in actions:
            if not a.is_toxic_exit:
                self.assertGreaterEqual(a.edge, 0.01, f"Action {a} has negative/low edge")

    def test_portfolio_no_double_count(self):
        """
        Verify that update_fill does not mutate positions, relying on API sync.
        """
        self.portfolio.positions = {"TEST": 10}
        self.portfolio.cash = 100.0
        
        # Fill happens
        self.portfolio.update_fill("TEST", "buy", 50.0, 5)
        
        # Position should still be 10 (awaiting API refresh)
        self.assertEqual(self.portfolio.positions["TEST"], 10)
        # Cash should be updated
        self.assertEqual(self.portfolio.cash, 100.0 - (50.0/100.0 * 5))

    def test_toxic_exit_still_works(self):
        """
        Verify that toxic flow exits are still triggered even with negative edge.
        """
        # Long position, fair value drops way below cost.
        # Fair = 20, Cost = 50.
        # bid=15, ask=25.
        # sell_price = 24. sell_edge = (24 - 30) / 100 = -0.06 (if CI upper is 30)
        
        actions = self.trader.evaluate_rebalancing(
            ticker="TEST",
            current_pos=10,
            cost_basis=50.0,
            model_prob=0.20,
            ci_lower=0.15,
            ci_upper=0.30,
            bid=15,
            ask=25
        )
        
        toxic_exits = [a for a in actions if a.is_toxic_exit]
        self.assertTrue(len(toxic_exits) > 0, "Toxic exit should have triggered")

if __name__ == "__main__":
    unittest.main()
