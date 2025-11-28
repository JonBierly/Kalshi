# Model Analysis & Interpretability Tools

Comprehensive toolkit for analyzing NBA prediction model behavior and understanding which features drive predictions.

## Overview

This toolkit provides:
- **Feature Importance Analysis** - Identify which features matter most
- **SHAP Values** - Understand individual prediction explanations
- **Partial Dependence Plots** - Visualize feature effects on predictions
- **Feature Correlations** - Detect redundant features
- **Prediction Explanations** - Break down specific game predictions

## Setup

Install required dependencies:
```bash
pip install shap matplotlib seaborn
```

## Usage

### 1. Comprehensive Model Analysis

Generate a full analysis report including feature importance, SHAP visualizations, and correlation analysis:

```bash
python scripts/analyze_model.py
```

**Outputs** (saved to `reports/model_analysis/`):
- `feature_importance_native.csv` - XGBoost native feature importance scores
- `feature_importance_permutation.csv` - Permutation importance scores
- `importance_native.png` - Visualization of native importance
- `importance_permutation.png` - Visualization of permutation importance
- `shap_summary.png` - SHAP summary plot showing global feature importance
- `shap_waterfall_example_*.png` - Example SHAP waterfall plots for individual predictions
- `partial_dependence.png` - Partial dependence plots for key features
- `correlation_matrix.png` - Feature correlation heatmap
- `feature_correlations.csv` - Full correlation matrix
- `analysis_summary.txt` - Text summary of key findings

**Time**: ~5-10 minutes depending on dataset size

### 2. Explain Specific Predictions

Analyze and explain predictions for specific games:

```bash
# Analyze a specific game
python scripts/explain_prediction.py 0022100001

# Analyze random sample games
python scripts/explain_prediction.py
```

**Outputs** (saved to `reports/predictions/`):
- `shap_waterfall_{game_id}.png` - SHAP waterfall showing feature contributions
- `contributions_{game_id}.png` - Bar chart of top feature contributions
- `trajectory_{game_id}.png` - Prediction probability over time during the game

**Features**:
- Shows final prediction vs actual outcome
- Displays top contributing features with SHAP values
- Visualizes how prediction evolved throughout the game
- Highlights model confidence (ensemble std dev)

## Python API

You can also use the analysis tools programmatically:

```python
from src.models.model_analysis import (
    FeatureImportanceAnalyzer,
    SHAPAnalyzer,
    PredictionExplainer
)
import joblib

# Load your trained models
models = joblib.load('models/nba_live_model_ensemble.pkl')
feature_names = ['score_diff', 'seconds_remaining', ...]  # Your features

# Feature Importance
importance_analyzer = FeatureImportanceAnalyzer(models, feature_names)
native_importance = importance_analyzer.get_native_importance()
perm_importance = importance_analyzer.get_permutation_importance(X_test, y_test)

# SHAP Analysis
shap_analyzer = SHAPAnalyzer(models, X_background, feature_names)
shap_values = shap_analyzer.get_shap_values(X_test)
fig = shap_analyzer.plot_waterfall(X_test, instance_idx=0)

# Explain Prediction
explainer = PredictionExplainer(models, feature_names)
result = explainer.explain_prediction(X_instance, shap_analyzer)
print(f"Probability: {result['probability']:.2%}")
print(result['feature_contributions'])
```

## Analysis Methods

### Feature Importance

**Native Importance (XGBoost)**:
- Based on how features are used in tree splits
- Fast to compute
- Shows average importance across ensemble
- Good for overall feature ranking

**Permutation Importance**:
- Measures performance drop when feature is shuffled
- More robust to correlated features
- Slower to compute
- Better reflects real predictive power

### SHAP (SHapley Additive exPlanations)

- Provides local (instance-level) explanations
- Shows how each feature contributes to specific predictions
- Waterfall plots show cumulative effect of features
- Summary plots aggregate SHAP values across dataset
- Theory: Based on game theory (Shapley values)

### Partial Dependence

- Shows marginal effect of features on predictions
- Helps understand non-linear relationships
- Useful for identifying optimal feature ranges
- Can reveal interaction effects

### Feature Correlation

- Identifies redundant features
- Helps with feature selection
- Can reveal multicollinearity issues

## Interpreting Results

### Feature Importance

High importance features are critical for model decisions. Consider:
- Are important features domain-appropriate?
- Are any surprising features highly ranked? (potential data leakage)
- Are game-state features (score_diff, seconds_remaining) dominating?

### SHAP Values

For individual predictions:
- Positive SHAP = pushes toward Home Win
- Negative SHAP = pushes toward Away Win
- Magnitude = strength of contribution
- Base value = average model output

### Prediction Confidence

- Low ensemble std dev = models agree (high confidence)
- High ensemble std dev = models disagree (uncertainty)
- Wide prediction range = ambiguous situation

## Tips

1. **Start with `analyze_model.py`** for overall understanding
2. **Use `explain_prediction.py`** to debug surprising predictions
3. **Focus on permutation importance** for feature selection
4. **Check correlation matrix** before removing features
5. **Compare SHAP with native importance** - discrepancies can reveal issues

## Examples

### Finding Key Features
```bash
python scripts/analyze_model.py
# Check: reports/model_analysis/analysis_summary.txt
# Look at: importance_native.png and importance_permutation.png
```

### Debugging Wrong Predictions
```bash
python scripts/explain_prediction.py 0022100123
# Check: reports/predictions/shap_waterfall_0022100123.png
# Look for: Which features contributed incorrectly?
```

### Understanding Model Behavior
```bash
# Run full analysis
python scripts/analyze_model.py

# Examine:
# - shap_summary.png: Global feature effects
# - partial_dependence.png: How features impact probability
# - correlation_matrix.png: Feature redundancy
```

## Troubleshooting

**"Model not found"**: Train model first with `python src/models/training.py`

**SHAP computation slow**: Reduce background sample size in the code

**Memory issues**: Reduce sample sizes in analysis scripts

**Missing visualizations**: Ensure matplotlib and seaborn are installed

## References

- SHAP: https://github.com/slundberg/shap
- XGBoost Feature Importance: https://xgboost.readthedocs.io/
- Permutation Importance: Breiman (2001), "Random Forests"
