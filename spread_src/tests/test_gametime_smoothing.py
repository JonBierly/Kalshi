import sys
import os
import unittest
import numpy as np

# Mocking parts of the trader for testing
class MockTrader:
    def __init__(self):
        self.weight_history = {}

    def smooth_weights(self, game_id, tickers, total_seconds, raw_optimal_weights):
        if game_id not in self.weight_history:
            self.weight_history[game_id] = {t: [] for t in tickers}
            
        smoothed_weights = {}
        for ticker in tickers:
            hist = self.weight_history[game_id].get(ticker, [])
            hist.append((total_seconds, raw_optimal_weights.get(ticker, 0.0)))
            
            # Prune entries older than 90 game-seconds
            hist = [h for h in hist if h[0] <= total_seconds + 90]
            
            self.weight_history[game_id][ticker] = hist
            smoothed_weights[ticker] = np.mean([h[1] for h in hist])
        return smoothed_weights

class TestGametimeSmoothing(unittest.TestCase):
    def test_gametime_window(self):
        trader = MockTrader()
        game_id = "GAME1"
        tickers = ["T1"]
        
        # 1. Start: 2880s remaining. Weight 0.1
        w1 = trader.smooth_weights(game_id, tickers, 2880, {"T1": 0.1})
        self.assertEqual(w1["T1"], 0.1)
        
        # 2. 10s later: 2870s remaining. Weight 0.2
        w2 = trader.smooth_weights(game_id, tickers, 2870, {"T1": 0.2})
        # Mean of 0.1, 0.2 = 0.15
        self.assertAlmostEqual(w2["T1"], 0.15)
        
        # 3. Time out! 10 real seconds pass, but gametime stays 2870s. Weight 0.3
        w3 = trader.smooth_weights(game_id, tickers, 2870, {"T1": 0.3})
        # Mean of 0.1, 0.2, 0.3 = 0.2
        self.assertAlmostEqual(w3["T1"], 0.2)
        
        # 4. Jump forward! Halftime or big break. 2870 -> 2700 (170s jump)
        # Weight 0.5. Old entries (2880, 2870) should be pruned because 2880 > 2700 + 90
        w4 = trader.smooth_weights(game_id, tickers, 2700, {"T1": 0.5})
        # Only 0.5 should remain
        self.assertAlmostEqual(w4["T1"], 0.5)

if __name__ == '__main__':
    unittest.main()
