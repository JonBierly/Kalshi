
import joblib
import pandas as pd
import numpy as np
from spread_src.features.engineering import add_interaction_features
from spread_src.models.distributions import SafeT
from ngboost.distns import Laplace

def test_priors_at_end():
    model_path = 'models/nba_spread_ngboost.pkl'
    model_data = joblib.load(model_path)
    ensemble = model_data['ensemble']
    feature_order = model_data['feature_order']
    
    def get_prediction(feats):
        enriched = add_interaction_features(feats)
        row = {col: enriched.get(col, 0.0) for col in feature_order}
        X = pd.DataFrame([row], columns=feature_order)
        means = []
        for model in ensemble:
            dist = model.pred_dist(X.values)
            means.append(feats['score_diff'] + dist.loc[0])
        return np.mean(means)

    # Base features (Neutral, Tied)
    base_feats = {f: 0.0 for f in feature_order}
    for f in feature_order:
        if 'off_rtg' in f or 'def_rtg' in f: base_feats[f] = 110.0
        if 'win_pct' in f: base_feats[f] = 0.5
        if 'pie' in f: base_feats[f] = 0.1
    
    times = [2880, 1440, 600, 300, 60, 10, 1]
    print(f"Home Advantage Drift Over Time (Neutral Priors, Tied Score):")
    for t in times:
        base_feats['seconds_remaining'] = float(t)
        base_feats['period'] = 1 if t > 2160 else (2 if t > 1440 else (3 if t > 720 else 4))
        m = get_prediction(base_feats)
        print(f"  T-{t:4d}s: Predicted Final Margin = {m:+.2f}")

if __name__ == "__main__":
    test_priors_at_end()
