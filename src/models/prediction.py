import joblib
import pandas as pd
import numpy as np
import time
from src.features.engineering import FeatureEngine, TeamStatsEngine, RosterEngine, BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST

class PredictionEngine:
    def __init__(self, model_type='lr', model_path=None):
        """
        Initialize prediction engine with specified model type.
        
        Args:
            model_type: 'lr' for Logistic Regression or 'xgboost' for XGBoost ensemble
            model_path: Optional custom path to model file. If None, uses default paths.
        """
        self.model_type = model_type
        
        # Determine model path
        if model_path is None:
            if model_type == 'lr':
                model_path = 'models/nba_lr_model.pkl'
            elif model_type == 'xgboost':
                model_path = 'models/nba_xgboost_ensemble.pkl'
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Must be 'lr' or 'xgboost'")
        
        print(f"Loading {model_type.upper()} model from {model_path}...")
        try:
            self.models = joblib.load(model_path)
            
            # For LR, convert single model to list for consistent interface
            if model_type == 'lr' and not isinstance(self.models, list):
                self.models = [self.models]
                
            n_models = len(self.models) if isinstance(self.models, list) else 1
            print(f"  Loaded {n_models} model(s)")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Please train the model first.")
            self.models = []
            
        self.feature_engine = FeatureEngine()
        self.team_engine = TeamStatsEngine()
        self.roster_engine = RosterEngine()
        
        self.current_game_id = None
        self.current_game_context = {}

    def load_game_context(self, game_id, home_id, away_id):
        """Pre-loads advanced features for the game."""
        print(f"Loading context for Game {game_id}...")
        t_feats = self.team_engine.get_features(game_id, home_id, away_id)
        r_feats = self.roster_engine.get_features(game_id, home_id, away_id)
        
        self.current_game_context = {**t_feats, **r_feats}
        self.current_game_id = game_id
        
        # Fill NaNs
        for k, v in self.current_game_context.items():
            if pd.isna(v): self.current_game_context[k] = 0.0

    def predict_with_confidence(self, live_features: dict):
        """Predicts win probability and confidence interval given live features."""
        start_time = time.time()
        
        feature_order = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
        
        # Ensure all columns are present, filling missing with 0
        row_data = {col: live_features.get(col, 0.0) for col in feature_order}
        X_live = pd.DataFrame([row_data], columns=feature_order)
        
        # Get predictions from all models in ensemble
        preds = []
        for model in self.models:
            try:
                # predict_proba returns [prob_0, prob_1]
                p = model.predict_proba(X_live)[0][1]
                preds.append(p)
            except Exception as e:
                print(f"Prediction error: {e}")
                preds.append(0.5)
            
        preds = np.array(preds)
        
        # Calculate stats
        mean_prob = np.mean(preds)
        std_dev = np.std(preds)
        
        # Confidence Intervals
        if len(preds) > 1:
            # Ensemble: Use percentiles from bootstrap predictions
            ci_95_lower = np.percentile(preds, 2.5)
            ci_95_upper = np.percentile(preds, 97.5)
            ci_90_lower = np.percentile(preds, 10)
            ci_90_upper = np.percentile(preds, 90)
            ci_60_lower = np.percentile(preds, 20)
            ci_60_upper = np.percentile(preds, 80)
            ci_50_lower = np.percentile(preds, 25)
            ci_50_upper = np.percentile(preds, 75)
        else:
            # Fallback for single model (shouldn't happen with new LR ensemble)
            # Simple ±5% confidence interval
            print("Warning: Single model detected, using fallback CI")
            ci_95_lower = max(0, mean_prob - 0.05)
            ci_95_upper = min(1, mean_prob + 0.05)
            ci_90_lower = max(0, mean_prob - 0.04)
            ci_90_upper = min(1, mean_prob + 0.04)
            ci_60_lower = max(0, mean_prob - 0.03)
            ci_60_upper = min(1, mean_prob + 0.03)
            ci_50_lower = max(0, mean_prob - 0.02)
            ci_50_upper = min(1, mean_prob + 0.02)
            std_dev = 0.05 / 2
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'probability': mean_prob,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'ci_90_lower': ci_90_lower,
            'ci_90_upper': ci_90_upper,
            'ci_60_lower': ci_60_lower,
            'ci_60_upper': ci_60_upper,
            'ci_50_lower': ci_50_lower,
            'ci_50_upper': ci_50_upper,
            'std_dev': std_dev,
            'latency_ms': latency_ms
        }

    def process_event(self, event_row: pd.Series):
        """
        Updates state with new event and returns prediction.
        """
        # Check if we need to load context
        game_id = event_row.get('game_id')
        if game_id and game_id != self.current_game_id:
            # Try to get team IDs from row
            home_id = event_row.get('home_team_id')
            away_id = event_row.get('away_team_id')
            if home_id and away_id:
                self.load_game_context(game_id, home_id, away_id)
        
        live_feats = self.feature_engine.update(event_row)
        
        # Merge with context
        full_feats = {**live_feats, **self.current_game_context}
        
        return self.predict_with_confidence(full_feats)

