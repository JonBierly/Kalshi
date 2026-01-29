
import joblib
import pandas as pd
import numpy as np
from spread_src.features.engineering import add_interaction_features
from spread_src.models.distributions import SafeT
from ngboost.distns import Laplace

def test_blowout_prior_drag():
    model_path = 'models/nba_spread_ngboost.pkl'
    try:
        model_data = joblib.load(model_path)
        ensemble = model_data['ensemble']
        feature_order = model_data['feature_order']
    except Exception as e:
        print(f"Failed to load: {e}")
        return
    
    def get_prediction(feats):
        enriched = add_interaction_features(feats)
        row = {col: enriched.get(col, 0.0) for col in feature_order}
        X = pd.DataFrame([row], columns=feature_order)
        means = []
        for model in ensemble:
            dist = model.pred_dist(X.values)
            means.append(feats['score_diff'] + dist.loc[0])
        return np.mean(means), np.mean([model.pred_dist(X.values).scale[0] for model in ensemble])

    # Case 1: Early game, neutral priors, tied.
    base_feats = {'score_diff': 0.0, 'seconds_remaining': 2880.0, 'period': 1, 'home_team_season_win_margin': 0.0, 'away_team_season_win_margin': 0.0}
    # Fill defaults
    for f in feature_order:
        if f not in base_feats: base_feats[f] = 0.0
        if 'off_rtg' in f or 'def_rtg' in f: base_feats[f] = 110.0
        if 'win_pct' in f: base_feats[f] = 0.5
        if 'pie' in f: base_feats[f] = 0.1
    
    m, s = get_prediction(base_feats)
    print(f"Start of Game (0-0): Mean={m:.2f}, Std={s:.2f}")

    # Case 2: 2 Minutes left, Up 17, Neutral priors.
    base_feats['score_diff'] = 17.0
    base_feats['seconds_remaining'] = 120.0
    base_feats['period'] = 4
    m, s = get_prediction(base_feats)
    print(f"2 Mins Left (+17): Mean={m:.2f}, Std={s:.2f} (Remainder: {m-17:.2f})")

    # Case 3: 2 Minutes left, Tied, Neutral priors.
    base_feats['score_diff'] = 0.0
    m, s = get_prediction(base_feats)
    print(f"2 Mins Left (0-0): Mean={m:.2f}, Std={s:.2f} (Remainder: {m:.2f})")

    # Case 4: 10 Seconds left, Up 17, Neutral priors.
    base_feats['score_diff'] = 17.0
    base_feats['seconds_remaining'] = 10.0
    m, s = get_prediction(base_feats)
    print(f"10 Secs Left (+17): Mean={m:.2f}, Std={s:.2f} (Remainder: {m-17:.2f})")

if __name__ == "__main__":
    test_blowout_prior_drag()
