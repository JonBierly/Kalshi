import pandas as pd
import numpy as np
from src.models.prediction import PredictionEngine

def verify_model():
    print("Loading Prediction Engine...")
    engine = PredictionEngine()
    
    # Scenario 1: Blowout (Home up 20, 2 mins left)
    # Even if Home Team is weak (low ratings), they should win.
    print("\n--- Scenario 1: Blowout (Home +20, 2 mins left) ---")
    
    # Mock Features
    # We need to construct the feature vector manually or use engine.process_event
    # Let's use predict_with_confidence directly with a constructed dict
    
    # Base Features
    feats = {
        'score_diff': 20,
        'seconds_remaining': 120,
        'home_efg': 0.55,
        'away_efg': 0.45,
        'turnover_diff': -2,
        'home_rebound_rate': 0.6,
        'required_catchup_rate': 20 / (120 + 1) # ~0.165
    }
    
    # Advanced Features (Weak Home Team)
    feats.update({
        'home_team_season_off_rtg': 105.0,
        'home_team_season_def_rtg': 115.0,
        'home_team_season_win_pct': 0.30,
        'home_roster_season_est_off_rating': 105.0,
        
        'away_team_season_off_rtg': 115.0,
        'away_team_season_def_rtg': 105.0,
        'away_team_season_win_pct': 0.70,
        'away_roster_season_est_off_rating': 115.0
    })
    
    pred = engine.predict_with_confidence(feats)
    print(f"Prediction: {pred['probability']:.4f}")
    print(f"95% CI: [{pred['ci_95_lower']:.4f}, {pred['ci_95_upper']:.4f}]")
    
    if pred['probability'] > 0.90:
        print("PASS: Model correctly predicts high win prob for blowout.")
    else:
        print("FAIL: Model predicts low win prob despite blowout.")

    # Scenario 2: Close Game (Tie, 1 min left, Strong Home Team)
    print("\n--- Scenario 2: Close Game (Tie, 1 min left, Strong Home Team) ---")
    
    feats = {
        'score_diff': 0,
        'seconds_remaining': 60,
        'home_efg': 0.50,
        'away_efg': 0.50,
        'turnover_diff': 0,
        'home_rebound_rate': 0.5,
        'required_catchup_rate': 0.0
    }
    
    # Strong Home Team vs Weak Away
    feats.update({
        'home_team_season_off_rtg': 115.0,
        'home_team_season_def_rtg': 105.0,
        'home_team_season_win_pct': 0.70,
        
        'away_team_season_off_rtg': 105.0,
        'away_team_season_def_rtg': 115.0,
        'away_team_season_win_pct': 0.30
    })
    
    pred = engine.predict_with_confidence(feats)
    print(f"Prediction: {pred['probability']:.4f}")
    
    if pred['probability'] > 0.55:
        print("PASS: Model favors strong home team in close game.")
    else:
        print("FAIL: Model does not favor strong home team enough.")

if __name__ == "__main__":
    verify_model()
