"""
Spread distribution model - predicts P(score_diff > threshold).

Uses NGBoost ensemble for:
1. Fat-tailed distributions (Student-T) - each model's aleatoric uncertainty
2. Confidence intervals from ensemble variance - epistemic uncertainty

Key insight: Kalshi spreads are continuous (3.5, 6.5, 9.5, etc.)
Instead of predicting discrete bins, we predict:
    P(final_score_diff > threshold) for any threshold
"""

import joblib
import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List

from spread_src.features.engineering import add_interaction_features
from spread_src.models.distributions import SafeT  # Required for pickle loading
from ngboost.distns import Laplace # Required for pickle loading


class SpreadDistributionModel:
    """
    Predicts probability that score differential exceeds various thresholds.
    
    Model predicts SCORE REMAINDER (final_diff - current_diff).
    At inference, we add current_diff to get expected final diff.
    
    Uses NGBoost ensemble with Student-T distribution for:
    - Fat tails (aleatoric uncertainty from each model)
    - Confidence intervals (epistemic uncertainty from ensemble variance)
    """
    
    def __init__(self, models_path='models/nba_spread_ngboost.pkl'):
        """
        Load NGBoost ensemble that predicts distribution parameters.
        """
        try:
            model_data = joblib.load(models_path)
            
            # Support both single model and ensemble formats
            if 'ensemble' in model_data:
                self.ensemble = model_data['ensemble']
                self.n_models = model_data.get('n_models', len(self.ensemble))
            elif 'model' in model_data:
                # Legacy single model format
                self.ensemble = [model_data['model']]
                self.n_models = 1
            else:
                raise ValueError("Unknown model format")
            
            self.feature_order = model_data['feature_order']
            self.distribution_type = model_data.get('distribution', 'Unknown')
            print(f"Loaded NGBoost ensemble ({self.n_models} models, {self.distribution_type} distribution)")
        except FileNotFoundError:
            print(f"No model found at {models_path}")
            self.ensemble = None
            self.feature_order = []
            self.n_models = 0
            
    def predict_distribution_params(self, live_features: dict, variance_multiplier: float = 1.0) -> Dict[str, np.ndarray]:
        """Simple wrapper for single prediction."""
        return self.predict_distribution_params_batch([live_features], variance_multiplier)


    def predict_distribution_params_batch(self, live_features_list: List[dict], variance_multiplier: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Predict distribution parameters for a list of features efficiently.
        
        Returns:
            {
                'mean': (N_samples, N_models) array,
                'std': (N_samples, N_models) array
            }
        """
        if self.ensemble is None:
            raise ValueError("No model loaded")
        if not live_features_list:
            return {'mean': np.array([]), 'std': np.array([])}
        
        # Batch preparation
        enriched_list = [add_interaction_features(f) for f in live_features_list]
        rows = []
        for feat in enriched_list:
            rows.append([feat.get(col, 0.0) for col in self.feature_order])
        
        X_batch = pd.DataFrame(rows, columns=self.feature_order)
        current_diffs = np.array([f.get('score_diff', 0.0) for f in enriched_list])
        
        all_means = []
        all_stds = []
        
        for model in self.ensemble:
            dist = model.pred_dist(X_batch.values)
            predicted_remainders = dist.loc
            scales = dist.scale
            
            if self.distribution_type == 'Laplace' or isinstance(dist, Laplace):
                true_stds = scales * np.sqrt(2)
            else:
                dfs = dist.df if hasattr(dist, 'df') else np.full(len(scales), 30.0)
                # Use np.maximum to avoid division-by-zero warnings for df=2 (even though np.where masks them)
                true_stds = np.where(dfs > 2, scales * np.sqrt(dfs / np.maximum(dfs - 2, 1e-6)), scales * 10.0)

            
            reconstructed_means = current_diffs + predicted_remainders
            all_means.append(reconstructed_means)
            all_stds.append(true_stds)
            
        return {
            'mean': np.array(all_means).T, # Shape (N_samples, N_models)
            'std': np.array(all_stds).T
        }
    
    def predict_spread_probabilities(self, live_features: dict, thresholds: List[float], variance_multiplier: float = 1.0) -> Dict:
        """
        Predict P(score_diff > threshold) for each threshold.
        """
        if self.ensemble is None:
            raise ValueError("No model loaded")
        
        # Restore critical interaction features
        enriched_features = add_interaction_features(live_features)
        
        # Prepare features using stored feature order
        row_data = {col: enriched_features.get(col, 0.0) for col in self.feature_order}
        X_live = pd.DataFrame([row_data], columns=self.feature_order)
        
        # Get current score diff for reconstruction
        current_diff = enriched_features.get('score_diff', 0.0)
        
        # Collect predictions from all models
        all_probs = []
        all_remainders = []
        all_stds = []
        
        for model in self.ensemble:
            dist = model.pred_dist(X_live.values)
            
            # Reconstruct mean and calculate true std
            predicted_remainder = dist.loc[0]
            scale = dist.scale[0]
            
            # Use raw scale for distributions
            if hasattr(dist, 'dist'):
                 # It's a SafeT / Student-T wrapper
                 df = dist.df[0] if hasattr(dist, 'df') else 30.0
                 from scipy.stats import t as t_dist
                 calibrated_dist = t_dist(df=df, loc=predicted_remainder, scale=scale)
            elif isinstance(dist, Laplace):
                 from scipy.stats import laplace
                 calibrated_dist = laplace(loc=predicted_remainder, scale=scale)
            else:
                 # Fallback to normal
                 from scipy.stats import norm
                 calibrated_dist = norm(loc=predicted_remainder, scale=scale)
            
            if self.distribution_type == 'Laplace' or isinstance(dist, Laplace):
                true_std = scale * np.sqrt(2)
            else:
                df = dist.df[0] if hasattr(dist, 'df') else 30.0
                if df > 2:
                    true_std = scale * np.sqrt(df / (df - 2))
                else:
                    true_std = scale * 10.0

            
            all_remainders.append(predicted_remainder)
            all_stds.append(true_std)
            
            # Compute probabilities for each threshold
            probs_for_model = []
            for threshold in thresholds:
                adjusted_threshold = threshold - current_diff
                # Use the CALIBRATED distribution for CDF
                prob = 1 - calibrated_dist.cdf(adjusted_threshold)
                probs_for_model.append(prob)
            
            all_probs.append(probs_for_model)
        
        all_probs = np.array(all_probs)
        
        # Ensemble aggregation
        mean_probs = np.mean(all_probs, axis=0)
        ci_90_lower = np.percentile(all_probs, 10, axis=0)
        ci_90_upper = np.percentile(all_probs, 90, axis=0)
        
        # Overall distribution stats based on true std
        mean_diff = current_diff + np.mean(all_remainders)
        std_diff = np.mean(all_stds)
        
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
        """
        is_home_market = (market_team == home_team)
        
        # predict_spread_probabilities already handles add_interaction_features
        result = self.predict_spread_probabilities(live_features, [market_spread])
        
        if is_home_market:
            prob = result['probabilities'][0]
            ci_lower = result['ci_90_lower'][0]
            ci_upper = result['ci_90_upper'][0]
        else:
            neg_result = self.predict_spread_probabilities(live_features, [-market_spread])
            prob = 1 - neg_result['probabilities'][0]
            ci_lower = 1 - neg_result['ci_90_upper'][0]
            ci_upper = 1 - neg_result['ci_90_lower'][0]
        
        return {
            'probability': prob,
            'ci_90_lower': ci_lower,
            'ci_90_upper': ci_upper
        }


if __name__ == "__main__":
    print("Spread distribution model defined!")
