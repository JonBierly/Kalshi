import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import log_loss, accuracy_score
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Analyzes feature importance using multiple methods."""
    
    def __init__(self, models: List[Any], feature_names: List[str]):
        """
        Args:
            models: List of trained XGBoost models
            feature_names: List of feature names in order
        """
        self.models = models
        self.feature_names = feature_names
        
    def get_native_importance(self) -> pd.DataFrame:
        """Get XGBoost native feature importance (averaged across ensemble)."""
        importance_dfs = []
        
        for i, model in enumerate(self.models):
            # Get importance scores
            importance = model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names,
                f'model_{i}': importance
            })
            importance_dfs.append(df)
        
        # Merge all models
        result = importance_dfs[0]
        for i in range(1, len(importance_dfs)):
            result = result.merge(importance_dfs[i], on='feature')
        
        # Calculate mean and std
        model_cols = [c for c in result.columns if c.startswith('model_')]
        result['mean_importance'] = result[model_cols].mean(axis=1)
        result['std_importance'] = result[model_cols].std(axis=1)
        
        return result.sort_values('mean_importance', ascending=False)
    
    def get_permutation_importance(self, X: pd.DataFrame, y: np.ndarray, 
                                   n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation importance for ensemble."""
        print("Calculating permutation importance...")
        
        all_importances = []
        
        for i, model in enumerate(self.models):
            print(f"  Model {i+1}/{len(self.models)}...")
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='neg_log_loss'
            )
            all_importances.append(perm_importance.importances_mean)
        
        # Average across models
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'permutation_importance': mean_importance,
            'std': std_importance
        })
        
        return df.sort_values('permutation_importance', ascending=False)
    
    def plot_importance(self, importance_df: pd.DataFrame, 
                       top_n: int = 20, 
                       importance_col: str = 'mean_importance',
                       title: str = 'Feature Importance') -> plt.Figure:
        """Plot feature importance with error bars."""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        importance_values = top_features[importance_col].values
        
        # Add error bars if std column exists
        std_col = importance_col.replace('mean', 'std')
        if std_col in top_features.columns:
            errors = top_features[std_col].values
        else:
            errors = None
        
        ax.barh(y_pos, importance_values, xerr=errors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig


class SHAPAnalyzer:
    """Analyzes predictions using SHAP values."""
    
    def __init__(self, models: List[Any], X_background: pd.DataFrame, 
                 feature_names: List[str]):
        """
        Args:
            models: List of trained models
            X_background: Background dataset for SHAP (sample of training data)
            feature_names: List of feature names
        """
        self.models = models
        self.feature_names = feature_names
        
        # Create SHAP explainers for each model
        print("Creating SHAP explainers (this may take a moment)...")
        self.explainers = []
        
        # Use a sample for faster computation
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        
        for i, model in enumerate(models):
            print(f"  Explainer {i+1}/{len(models)}...")
            explainer = shap.TreeExplainer(model, background_sample)
            self.explainers.append(explainer)
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values averaged across ensemble."""
        all_shap_values = []
        
        for explainer in self.explainers:
            shap_values = explainer.shap_values(X)
            all_shap_values.append(shap_values)
        
        # Average SHAP values across models
        mean_shap = np.mean(all_shap_values, axis=0)
        return mean_shap
    
    def plot_summary(self, X: pd.DataFrame, max_display: int = 20) -> plt.Figure:
        """Create SHAP summary plot showing feature importance."""
        shap_values = self.get_shap_values(X)
        
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.tight_layout()
        return fig
    
    def plot_waterfall(self, X_instance: pd.DataFrame, instance_idx: int = 0) -> plt.Figure:
        """Create waterfall plot for a single prediction."""
        shap_values = self.get_shap_values(X_instance.iloc[[instance_idx]])
        
        # Get base value (average across explainers)
        base_values = [exp.expected_value for exp in self.explainers]
        base_value = np.mean(base_values)
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_instance.iloc[instance_idx].values,
            feature_names=self.feature_names
        )
        
        fig = plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        return fig
    
    def plot_force(self, X_instance: pd.DataFrame, instance_idx: int = 0):
        """Create force plot for a single prediction (returns HTML)."""
        shap_values = self.get_shap_values(X_instance.iloc[[instance_idx]])
        
        base_values = [exp.expected_value for exp in self.explainers]
        base_value = np.mean(base_values)
        
        return shap.force_plot(
            base_value,
            shap_values[0],
            X_instance.iloc[instance_idx],
            feature_names=self.feature_names
        )
    
    def get_feature_contributions(self, X_instance: pd.DataFrame, 
                                  instance_idx: int = 0) -> pd.DataFrame:
        """Get feature contributions for a single prediction."""
        shap_values = self.get_shap_values(X_instance.iloc[[instance_idx]])
        
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_instance.iloc[instance_idx].values,
            'shap_value': shap_values[0],
            'abs_shap': np.abs(shap_values[0])
        })
        
        return contributions.sort_values('abs_shap', ascending=False)


class PartialDependenceAnalyzer:
    """Analyzes partial dependence of features."""
    
    def __init__(self, models: List[Any], feature_names: List[str]):
        self.models = models
        self.feature_names = feature_names
    
    def plot_partial_dependence(self, X: pd.DataFrame, features: List[str],
                                grid_resolution: int = 50) -> plt.Figure:
        """Plot partial dependence for specified features."""
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        # Calculate PD for each feature using first model (representative)
        model = self.models[0]
        
        for idx, feature in enumerate(features):
            if feature not in self.feature_names:
                print(f"Warning: {feature} not in feature list")
                continue
            
            feature_idx = self.feature_names.index(feature)
            
            # Calculate partial dependence
            pd_result = partial_dependence(
                model, X, [feature_idx],
                grid_resolution=grid_resolution
            )
            
            ax = axes[idx]
            ax.plot(pd_result['grid_values'][0], pd_result['average'][0])
            ax.set_xlabel(feature)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'PD Plot: {feature}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig


class FeatureCorrelationAnalyzer:
    """Analyzes feature correlations and redundancy."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def get_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix."""
        return X[self.feature_names].corr()
    
    def plot_correlation_matrix(self, X: pd.DataFrame, 
                               threshold: float = 0.8) -> plt.Figure:
        """Plot correlation heatmap with highly correlated pairs highlighted."""
        corr = self.get_correlation_matrix(X)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Print highly correlated pairs
        high_corr = []
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_corr.append((
                        corr.index[i], 
                        corr.columns[j], 
                        corr.iloc[i, j]
                    ))
        
        if high_corr:
            print(f"\nHighly correlated features (|r| > {threshold}):")
            for feat1, feat2, r in high_corr:
                print(f"  {feat1} <-> {feat2}: {r:.3f}")
        
        return fig
    
    def find_redundant_features(self, X: pd.DataFrame, 
                               threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find pairs of highly correlated features."""
        corr = self.get_correlation_matrix(X)
        redundant = []
        
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if abs(corr.iloc[i, j]) > threshold:
                    redundant.append((
                        corr.index[i],
                        corr.columns[j],
                        corr.iloc[i, j]
                    ))
        
        return redundant


class PredictionExplainer:
    """Explains individual predictions."""
    
    def __init__(self, models: List[Any], feature_names: List[str]):
        self.models = models
        self.feature_names = feature_names
    
    def explain_prediction(self, X_instance: pd.DataFrame, 
                          shap_analyzer: Optional[SHAPAnalyzer] = None) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for a single prediction.
        
        Returns:
            Dictionary with prediction, probabilities, and feature contributions
        """
        # Get predictions from all models
        probs = []
        for model in self.models:
            prob = model.predict_proba(X_instance)[0][1]
            probs.append(prob)
        
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        
        result = {
            'probability': mean_prob,
            'std': std_prob,
            'min_prob': np.min(probs),
            'max_prob': np.max(probs),
            'individual_predictions': probs
        }
        
        # Add SHAP values if analyzer provided
        if shap_analyzer:
            contributions = shap_analyzer.get_feature_contributions(X_instance)
            result['feature_contributions'] = contributions
        
        return result
    
    def compare_predictions(self, X_instances: pd.DataFrame, 
                           labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare predictions for multiple instances."""
        results = []
        
        for idx in range(len(X_instances)):
            instance = X_instances.iloc[[idx]]
            
            # Get ensemble prediction
            probs = [m.predict_proba(instance)[0][1] for m in self.models]
            mean_prob = np.mean(probs)
            
            label = labels[idx] if labels else f"Instance {idx}"
            
            results.append({
                'label': label,
                'probability': mean_prob,
                'std': np.std(probs)
            })
        
        return pd.DataFrame(results)
