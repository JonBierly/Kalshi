import joblib
import numpy as np
import pandas as pd
from spread_src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST

def check_feature_importance():
    model_path = 'models/nba_spread_model.pkl'
    try:
        ensemble = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    # Ridge models have .coef_
    feature_names = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    
    all_coefs = []
    for model_pair in ensemble:
        all_coefs.append(model_pair['mean_model'].coef_)
    
    mean_coefs = np.mean(all_coefs, axis=0)
    std_coefs = np.std(all_coefs, axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_coef': mean_coefs,
        'std_coef': std_coefs,
        'abs_coef': np.abs(mean_coefs)
    })
    
    importance_df = importance_df.sort_values('abs_coef', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Ridge Coefficients)")
    print("="*80)
    print(importance_df[['feature', 'mean_coef', 'std_coef']].to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    check_feature_importance()
