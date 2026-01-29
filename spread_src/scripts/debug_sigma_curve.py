
import joblib
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.models.spread_model import SpreadDistributionModel

def plot_sigma_curve():
    model = SpreadDistributionModel()
    
    times = np.linspace(0, 2880, 50) # 0 to 48 mins
    sigmas = []
    
    for t in times:
        live_features = {
            'score_diff': 0,
            'seconds_remaining': t,
            'period': 1 if t > 2160 else 2 if t > 1440 else 3 if t > 720 else 4,
            'home_team_id': 1610612737,
            'away_team_id': 1610612738,
            'home_pts': 0,
            'away_pts': 0,
            'score_volatility': 0
        }
        params = model.predict_distribution_params(live_features)
        sigmas.append(np.mean(params['std']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(2880 - times, sigmas, marker='o') # X axis: Seconds elapsed
    plt.xlabel('Seconds Elapsed')
    plt.ylabel('Predicted Sigma (σ)')
    plt.title('Predicted Sigma vs Game Time (Score Diff = 0)')
    plt.grid(True)
    plt.savefig('reports/sigma_curve_debug.png')
    print(f"Saved sigma curve to reports/sigma_curve_debug.png")
    print(f"Sigma at start (2880 left): {sigmas[-1]}")
    print(f"Sigma at end (0 left): {sigmas[0]}")

if __name__ == "__main__":
    plot_sigma_curve()
