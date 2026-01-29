
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.models.spread_model import SpreadDistributionModel

def test_game_start_sigma():
    model = SpreadDistributionModel()
    
    # 0-0 game, 2880 seconds remaining (Start of NBA game)
    live_features = {
        'score_diff': 0,
        'seconds_remaining': 2880,
        'period': 1,
        'home_team_id': 1610612737, # Hawks
        'away_team_id': 1610612738, # Celtics
        'home_pts': 0,
        'away_pts': 0,
        'score_volatility': 0
    }
    
    params = model.predict_distribution_params(live_features)
    predicted_means = params['mean']
    predicted_stds = params['std']
    
    print(f"\nGame Start (0-0, 2880s left) Prediction:")
    print(f"  Predicted Means: {predicted_means}")
    print(f"  Predicted Sigmas (Aleatoric): {predicted_stds}")
    print(f"  Ensemble Mean Sigma: {np.mean(predicted_stds):.2f}")
    if len(predicted_means) > 1:
        print(f"  Ensemble Epistemic Sigma (Std of Means): {np.std(predicted_means):.2f}")
        total_sigma = np.sqrt(np.mean(predicted_stds**2) + np.var(predicted_means))
        print(f"  TOTAL SIGMA: {total_sigma:.2f}")
    else:
        print(f"  TOTAL SIGMA: {predicted_stds[0]:.2f}")

if __name__ == "__main__":
    test_game_start_sigma()
