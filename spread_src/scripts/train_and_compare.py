#!/usr/bin/env python
"""
Train and compare all 3 spread models.

Models:
1. Ridge Mean+Std (already trained)
2. XGBoost Mean+Std
3. Quantile Regression

Usage:
    python spread_src/scripts/train_and_compare.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.scripts.train_xgboost import train_xgboost_spread_models
from spread_src.scripts.train_quantile import train_quantile_regression_models
from spread_src.scripts.compare_models import compare_models


def main():
    print("=" * 80)
    print("TRAINING ALL SPREAD MODELS")
    print("=" * 80)
    
    # Model 1: Ridge (already trained)
    print("\n[1/3] Ridge Mean+Std: Already trained ✓")
    
    # Model 2: XGBoost
    print("\n[2/3] Training XGBoost Mean+Std...")
    print("=" * 80)
    try:
        train_xgboost_spread_models(n_models=5)
        print("\n✓ XGBoost training complete!")
    except Exception as e:
        print(f"\n✗ XGBoost training failed: {e}")
        return
    
    # Model 3: Quantile
    print("\n[3/3] Training Quantile Regression...")
    print("=" * 80)
    try:
        train_quantile_regression_models(n_models=5)
        print("\n✓ Quantile training complete!")
    except Exception as e:
        print(f"\n✗ Quantile training failed: {e}")
        return
    
    # Compare all models
    print("\n\n")
    print("=" * 80)
    print("COMPARING ALL MODELS")
    print("=" * 80)
    
    try:
        compare_models()
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
