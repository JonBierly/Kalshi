"""
Spread distribution model - predicts P(score_diff > threshold).

Key insight: Kalshi spreads are continuous (3.5, 6.5, 9.5, etc.)
Instead of predicting discrete bins, we predict:
    P(final_score_diff > threshold) for any threshold

This is the Cumulative Distribution Function (CDF) of score differentials.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import joblib


class SpreadDistributionModel:
    """
    Predicts probability that score differential exceeds various thresholds.
    
    Uses parametric approach: fit a distribution (normal, skew-normal, etc.)
    to the score differential, then compute P(diff > threshold) analytically.
    """
    
    def __init__(self, models_path='models/nba_spread_model.pkl'):
        """
        Load ensemble of regression models that predict distribution parameters.
        
        Models predict:
            - mean (mu): expected score differential
            - std (sigma): uncertainty in score differential  
            - skew (alpha): asymmetry (optional, for skew-normal)
        """
        try:
            self.models = joblib.load(models_path)
            print(f"Loaded {len(self.models)} spread models")
        except FileNotFoundError:
            print(f"No models found at {models_path}")
            self.models = []
    
    def predict_distribution_params(self, live_features: dict) -> Dict[str, np.ndarray]:
        """
        Predict distribution parameters from ensemble.
        
        Returns:
            {
                'mean': array of means from each model,
                'std': array of stds from each model,
            }
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        # Prepare features
        from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST
        feature_order = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
        row_data = {col: live_features.get(col, 0.0) for col in feature_order}
        X_live = pd.DataFrame([row_data], columns=feature_order)
        
        # Get predictions from each model
        means = []
        stds = []
        
        for model in self.models:
            # Each model is a dict with 'mean_model' and 'std_model'
            if isinstance(model, dict):
                mean_pred = model['mean_model'].predict(X_live)[0]
                std_pred = model['std_model'].predict(X_live)[0]
            else:
                # Fallback: single model predicts mean only
                mean_pred = model.predict(X_live)[0]
                std_pred = 5.0  # Default uncertainty
            
            means.append(mean_pred)
            stds.append(max(std_pred, 1.0))  # Ensure positive std
        
        return {
            'mean': np.array(means),
            'std': np.array(stds)
        }
    
    def predict_spread_probabilities(self, live_features: dict, thresholds: List[float]) -> Dict:
        """
        Predict P(score_diff > threshold) for each threshold.
        
        Args:
            live_features: Live game features
            thresholds: List of spread values (e.g., [3.5, 6.5, 9.5])
        
        Returns:
            {
                'thresholds': [3.5, 6.5, 9.5, ...],
                'probabilities': [0.45, 0.28, 0.15, ...],  # Mean from ensemble
                'ci_90_lower': [...],
                'ci_90_upper': [...],
                'mean_diff': 5.2,  # Expected score differential
                'std_diff': 8.5    # Uncertainty
            }
        """
        # Get distribution parameters from ensemble
        params = self.predict_distribution_params(live_features)
        
        # For each model in ensemble, compute P(diff > threshold) for each threshold
        all_probs = []  # Shape: (n_models, n_thresholds)
        
        for mean, std in zip(params['mean'], params['std']):
            # Assume normal distribution
            dist = stats.norm(loc=mean, scale=std)
            
            # P(diff > threshold) = 1 - CDF(threshold)
            probs_for_model = [1 - dist.cdf(t) for t in thresholds]
            all_probs.append(probs_for_model)
        
        all_probs = np.array(all_probs)  # Shape: (n_models, n_thresholds)
        
        # Aggregate across ensemble
        mean_probs = np.mean(all_probs, axis=0)
        ci_90_lower = np.percentile(all_probs, 10, axis=0)
        ci_90_upper = np.percentile(all_probs, 90, axis=0)
        
        # Overall distribution stats
        mean_diff = np.mean(params['mean'])
        std_diff = np.mean(params['std'])
        
        return {
            'thresholds': thresholds,
            'probabilities': mean_probs,
            'ci_90_lower': ci_90_lower,
            'ci_90_upper': ci_90_upper,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        }
    
    def predict_for_market(self, live_features: dict, market_spread: float, 
                          market_team: str, home_team: str) -> Dict:
        """
        Predict probability for a specific spread market.
        
        Args:
            live_features: Live game features
            market_spread: Spread value (e.g., 6.5)
            market_team: Team that must cover (e.g., 'DET')
            home_team: Home team identifier
        
        Returns:
            {
                'probability': 0.35,  # P(team covers spread)
                'ci_90_lower': 0.28,
                'ci_90_upper': 0.42
            }
        """
        # Determine if market is for home or away
        is_home_market = (market_team == home_team)
        
        # Predict spread distribution
        result = self.predict_spread_probabilities(live_features, [market_spread])
        
        if is_home_market:
            # Market: "Home wins by > X"
            # This is P(home_diff > X) where home_diff = home_score - away_score
            prob = result['probabilities'][0]
            ci_lower = result['ci_90_lower'][0]
            ci_upper = result['ci_90_upper'][0]
        else:
            # Market: "Away wins by > X"
            # This is P(away_diff > X) = P(-home_diff > X) = P(home_diff < -X)
            # Using symmetry: P(diff < -X) = CDF(-X)
            # But we computed P(diff > X), so:
            # P(diff < -X) = 1 - P(diff > -X)
            
            # Recompute for negative threshold
            neg_result = self.predict_spread_probabilities(live_features, [-market_spread])
            prob = 1 - neg_result['probabilities'][0]
            ci_lower = 1 - neg_result['ci_90_upper'][0]
            ci_upper = 1 - neg_result['ci_90_lower'][0]
        
        return {
            'probability': prob,
            'ci_90_lower': ci_lower,
            'ci_90_upper': ci_upper
        }


# Example usage
if __name__ == "__main__":
    # This would be used like:
    # model = SpreadDistributionModel()
    # 
    # live_features = {
    #     'score_diff': 5,  # Home up by 5
    #     'seconds_remaining': 600,
    #     'home_efg': 0.52,
    #     'away_efg': 0.48,
    #     ...
    # }
    # 
    # # Predict for all common spreads
    # thresholds = [3.5, 6.5, 9.5, 12.5]
    # result = model.predict_spread_probabilities(live_features, thresholds)
    # 
    # print("Probabilities:")
    # for t, p in zip(result['thresholds'], result['probabilities']):
    #     print(f"  P(diff > {t}) = {p:.1%}")
    
    print("Spread distribution model defined - ready for training!")
