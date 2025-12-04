# Model Comparison: 3 Approaches to Spread Prediction

## Models

### 1. Ridge Mean+Std (Baseline) ✓ Trained
**Approach:** Parametric (Normal distribution assumed)

**How it works:**
- Train Ridge regression to predict **mean** (expected score diff)
- Train Ridge regression to predict **std** (uncertainty)
- Assume score diff ~ Normal(μ, σ)
- Compute P(diff > threshold) = 1 - Φ((threshold - μ) / σ)

**Pros:**
- ⚡ Very fast inference (<1ms)
- Simple, interpretable
- Linear feature relationships

**Cons:**
- Assumes normal distribution (NBA scores aren't perfectly normal)
- Limited modeling capacity
- May underfit complex patterns

**Best for:** Speed-critical applications, baseline comparison

---

### 2. XGBoost Mean+Std
**Approach:** Parametric (Normal distribution assumed)

**How it works:**
- Same as Ridge, but use XGBoost instead
- Captures non-linear feature interactions
- Early stopping to prevent overfitting

**Pros:**
- 🎯 Better accuracy (typically 10-20% improvement)
- Captures complex patterns (e.g., "late game + close score → tighter spread")
- Feature importance available

**Cons:**
- Slower inference (~5ms vs <1ms)
- Still assumes normality
- More prone to overfitting

**Best for:** When accuracy matters more than speed

---

### 3. Quantile Regression
**Approach:** Non-parametric (no distribution assumed)

**How it works:**
- Train 5 separate models, each predicting a different quantile:
  - 10th percentile
  - 25th percentile (Q1)
  - 50th percentile (median)
  - 75th percentile (Q3)
  - 90th percentile
- Interpolate between quantiles to get P(diff > threshold)

**Pros:**
- 📊 Best calibration (no parametric assumptions)
- Handles skewness naturally
- Direct confidence interval estimation

**Cons:**
- Need to train 5x models (slower training)
- Interpolation required for arbitrary thresholds
- Slightly slower inference

**Best for:** When calibration is critical (it is for trading!)

---

## Quick Start

### Train All Models & Compare

```bash
cd /Users/jonathanbierly/Desktop/Classes/Projects/Kalshi
python spread_src/scripts/train_and_compare.py
```

This will:
1. Use existing Ridge model ✓
2. Train XGBoost model (~5 mins)
3. Train Quantile model (~8 mins)
4. Compare all 3 on test set

**Total time:** ~15 minutes

---

### Train Individual Models

```bash
# Already done
python -m spread_src.scripts.train_spread_model  # Ridge

# New models
python -m spread_src.scripts.train_xgboost       # XGBoost
python -m spread_src.scripts.train_quantile      # Quantile
```

---

### Compare Existing Models

```bash
python -m spread_src.scripts.compare_models
```

---

## Expected Results

### Point Prediction Accuracy

| Model | MAE | RMSE | Speed (ms/row) |
|-------|-----|------|----------------|
| **Ridge** | 7.79 | 10.31 | 0.001 |
| **XGBoost** | ~7.0 | ~9.5 | 0.005 |
| **Quantile** | ~7.2 | ~9.8 | 0.003 |

**Winner:** XGBoost (best accuracy)

---

### Probabilistic Calibration

Example: P(diff > +5)

| Model | Predicted | Actual | Error |
|-------|-----------|--------|-------|
| **Ridge** | 38.8% | 43.2% | 4.4% |
| **XGBoost** | ~41.0% | 43.2% | ~2% |
| **Quantile** | ~42.5% | 43.2% | ~1% |

**Winner:** Quantile (best calibration)

---

### Overall Recommendation

**For live trading:** Use **Quantile Regression**

**Why:**
- Calibration is more important than point accuracy
- Bad probabilities → bad Kelly sizing → losses
- Quantile is only slightly slower than Ridge
- No parametric assumptions = robust to outliers

**Fallback:** If quantile is too slow in production, use **XGBoost** (good middle ground)

---

## Interpreting Results

### MAE (Mean Absolute Error)
- Average miss distance
- **Lower is better**
- Example: MAE = 7.5 means off by 7.5 points on average

### RMSE (Root Mean Squared Error)  
- Penalizes big misses more
- **Lower is better**
- RMSE > MAE indicates some large errors

### Bias
- Systematic over/under prediction
- **Close to 0 is best**
- Positive = predicts too high, Negative = predicts too low

### Std Calibration
- Predicted uncertainty vs actual uncertainty
- **Ratio of 1.0 is perfect**
- < 1.0 = overconfident, > 1.0 = underconfident

### Probabilistic Calibration
- P(diff > threshold): Predicted vs Actual
- **Smaller error is better**
- This is THE most important metric for trading!

---

## Files

```
spread_src/scripts/
├── train_spread_model.py    ✓ Ridge Mean+Std
├── train_xgboost.py          ✓ XGBoost Mean+Std
├── train_quantile.py         ✓ Quantile Regression
├── compare_models.py         ✓ Benchmarking script
└── train_and_compare.py      ✓ Master script
```

---

## Next Steps After Comparison

1. **Choose best model** based on calibration
2. **Integrate into spread_model.py** (update prediction logic)
3. **Build spread tracker** for live trading
4. **Paper trade** for 1 week to validate

---

Happy training! 🚀
