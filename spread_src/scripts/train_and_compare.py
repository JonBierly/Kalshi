#!/usr/bin/env python
"""
Train and compare Ridge vs XGBoost spread models.

Models:
1. Ridge Mean+Std
2. XGBoost Mean+Std

Usage:
    python spread_src/scripts/train_and_compare.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.scripts.train_spread_model import train_spread_models
from spread_src.scripts.train_xgboost import train_xgboost_spread_models
from spread_src.scripts.compare_models import compare_models


def main():
    print("=" * 80)
    print("TRAINING RIDGE VS XGBOOST SPREAD MODELS")
    print("=" * 80)
    
    # Model 1: Ridge
    print("\n[1/2] Training Ridge Mean+Std...")
    print("=" * 80)
    try:
        train_spread_models(n_models=10)
        print("\n✓ Ridge training complete!")
    except Exception as e:
        print(f"\n✗ Ridge training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Model 2: XGBoost
    print("\n[2/2] Training XGBoost Mean+Std...")
    print("=" * 80)
    try:
        train_xgboost_spread_models(n_models=10)
        print("\n✓ XGBoost training complete!")
    except Exception as e:
        print(f"\n✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare models
    print("\n\n")
    print("=" * 80)
    print("COMPARING RIDGE VS XGBOOST")
    print("=" * 80)
    
    try:
        compare_models()
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

