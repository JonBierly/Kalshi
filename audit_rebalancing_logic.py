
import sys
import os
import unittest
from dataclasses import dataclass
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from spread_src.trading.rebalancing_logic import RebalancingTrader, RebalancingAction
from spread_src.execution.risk_manager import RiskManager

class TestRebalancingAudit(unittest.TestCase):
    def setUp(self):
        # Professional settings
        self.bankroll = 10.0  # max game exposure
        self.kelly_fraction = 0.25
        self.min_edge = 0.04
        self.max_ticker_exposure = 3.0
        self.hysteresis = 2
        
        self.trader = RebalancingTrader(
            bankroll=self.bankroll,
            kelly_fraction=self.kelly_fraction,
            min_edge=self.min_edge,
            max_ticker_exposure=self.max_ticker_exposure,
            hysteresis_buffer=self.hysteresis
        )
        
        self.risk_mgr = RiskManager(max_total_exposure=200, max_game_exposure=10)

    def test_kelly_sizing_and_ticker_limit(self):
        print("\n--- Testing Kelly Sizing and Ticker Limits ---")
        ticker = "GAME-TEST-1"
        # Fair 95c, Bid 5c -> 90% edge
        # With bankroll=10, fractional kelly (0.25) should result in high contrast count
        # but the ticker limit ($3) should constrain it.
        actions = self.trader.evaluate_rebalancing(
            ticker=ticker,
            current_pos=0,
            cost_basis=0,
            model_prob=0.95,
            ci_lower=0.94,
            ci_upper=0.96,
            bid=5,
            ask=7
        )
        
        self.assertTrue(len(actions) > 0)
        action = actions[0]
        self.assertEqual(action.action, 'buy')
        
        # Manual verify if within $3
        cost = (action.price / 100.0) * action.size
        print(f"  Ticker Limit Check: Size {action.size} @ {action.price}c = ${cost:.2f}")
        self.assertLessEqual(cost, 3.01)

    def test_hysteresis_no_trade(self):
        print("\n--- Testing Hysteresis ---")
        # Fair 60c, Bid 50c -> Target ~1 contract
        # If we have 0, target is 1, delta is 1. Buffer is 2. Should NOT trade.
        actions = self.trader.evaluate_rebalancing(
            ticker="GAME-TEST-2",
            current_pos=0,
            cost_basis=0,
            model_prob=0.60,
            ci_lower=0.55,
            ci_upper=0.65,
            bid=50,
            ask=52
        )
        print(f"  Small Delta (1 < 2): {len(actions)} actions (Expected 0)")
        self.assertEqual(len(actions), 0)

    def test_toxic_flow_exit(self):
        print("\n--- Testing Toxic Flow ---")
        # Position: LONG 10 @ 60c
        # Fair: 50c (below cost)
        # CI Lower: 45c (edge to buy is negative)
        actions = self.trader.evaluate_rebalancing(
            ticker="GAME-TEST-3",
            current_pos=10,
            cost_basis=60.0,
            model_prob=0.50, # Below cost
            ci_lower=0.45,
            ci_upper=0.55,
            bid=48,
            ask=50
        )
        
        self.assertTrue(any(a.reason == "Toxic Flow Exit" for a in actions))
        exit_action = [a for a in actions if a.reason == "Toxic Flow Exit"][0]
        self.assertEqual(exit_action.action, 'sell')
        self.assertEqual(exit_action.size, 10)
        print(f"  Toxic Exit Triggered: {exit_action.action} {exit_action.size} contracts")

    def test_exposure_delta_logic(self):
        print("\n--- Testing RiskManager Exposure Delta ---")
        # Buying to close a short
        # Position: -10 contracts @ 50c
        # Order: BUY 10 @ 60c
        # RiskManager logic for closing short at 60c: -((100-60)/100)*10 = -$4.00
        delta = self.risk_mgr.get_exposure_delta('buy', 60, 10, -10)
        print(f"  Closing Short (-10 -> 0) at 60c: Delta ${delta:.2f} (Expected -$4.00)")
        self.assertAlmostEqual(delta, -4.0)
        
        # Scaling up
        # Position: 5 @ 50c ($2.50 exposure)
        # Order: BUY 5 @ 50c
        # Delta should be +2.50
        delta = self.risk_mgr.get_exposure_delta('buy', 50, 5, 5)
        print(f"  Scaling Up (5 -> 10): Delta ${delta:.2f} (Expected +$2.50)")
        self.assertAlmostEqual(delta, 2.5)

    def test_pending_aware_rebalancing(self):
        print("\n--- Testing Pending-Aware Rebalancing ---")
        # Fair 95c, Bid 5c -> Target high (max contracts)
        # But if we have 0 pos and a pending BUY order for the max contracts,
        # we should NOT place another order.
        ticker = "GAME-TEST-PENDING"
        max_contracts = int(self.max_ticker_exposure / (6 / 100.0))
        
        # 1. No pending -> Should trade
        actions = self.trader.evaluate_rebalancing(
            ticker=ticker,
            current_pos=0,
            cost_basis=0,
            model_prob=0.95,
            ci_lower=0.94,
            ci_upper=0.96,
            bid=5,
            ask=7,
            pending_pos=0
        )
        self.assertTrue(len(actions) > 0)
        print(f"  Actions with 0 pending: {len(actions)}")
        
        # 2. Pending max contracts -> Should NOT trade (specifically not BUY)
        # It might return a SELL if it wants to reduce the pending, but let's check for BUY
        actions = self.trader.evaluate_rebalancing(
            ticker=ticker,
            current_pos=0,
            cost_basis=0,
            model_prob=0.95,
            ci_lower=0.94,
            ci_upper=0.96,
            bid=5,
            ask=7,
            pending_pos=max_contracts
        )
        buy_actions = [a for a in actions if a.action == 'buy']
        print(f"  Buy actions with max pending: {len(buy_actions)} (Expected 0)")
        self.assertEqual(len(buy_actions), 0)

if __name__ == '__main__':
    unittest.main()
