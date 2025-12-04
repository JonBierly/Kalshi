import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from src.data.database import DatabaseManager
from src.features.engineering import create_live_features, add_advanced_features, BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def prepare_training_data(num_games=None):
    """Generates a dataset using the Database and labels it."""
    db_manager = DatabaseManager()
    limit_str = f"Limit {num_games}" if num_games else "All"
    print(f"Loading training data from Database ({limit_str} games)...")
    
    session = db_manager.get_session()
    from src.data.database import Game
    # Filter for valid games only (non-zero team IDs)
    query = session.query(Game.game_id, Game.home_team_id, Game.away_team_id)\
        .filter(Game.home_team_id != 0, Game.away_team_id != 0)
        
    if num_games:
        query = query.limit(num_games)
        
    games_meta = query.all()
    session.close()
    
    print(f"Found {len(games_meta)} games in DB.")
    
    all_live_features = []
    all_targets = []
    
    for game_id, home_id, away_id in games_meta:
        query = f"SELECT * FROM pbp_events WHERE game_id = '{game_id}' ORDER BY period, remaining_time DESC"
        pbp_df = pd.read_sql(query, db_manager.engine)
        
        if pbp_df.empty: continue
        
        pbp_df['home_team_id'] = home_id
        pbp_df['away_team_id'] = away_id
        
        features_df = create_live_features(pbp_df)
        
        # Determine winner (Target)
        final_row = pbp_df.iloc[-1]
        home_win = 1 if final_row['home_score'] > final_row['away_score'] else 0
        
        all_live_features.append(features_df)
        all_targets.extend([home_win] * len(features_df))
        
    if not all_live_features:
        return pd.DataFrame(), np.array([])
        
    full_features_df = pd.concat(all_live_features, ignore_index=True)
    y = np.array(all_targets)
    
    print("Adding Advanced Features...")
    X = add_advanced_features(full_features_df)
    X = X.fillna(0)
    
    return X, y

from src.models.experiment import ExperimentLogger

def train_models(models_to_train=['lr', 'xgboost']):
    """
    Train NBA prediction models.
    
    Args:
        models_to_train: List of models to train. Options: 'lr', 'xgboost'
                        Default: ['lr', 'xgboost'] (train both)
    
    Returns:
        dict: Trained models {'lr': model, 'xgboost': ensemble_models}
    """
    # 1. Prepare Data
    X, y = prepare_training_data()
    
    # Split by Game ID to prevent leakage
    game_ids = X['game_id'].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    
    train_mask = X['game_id'].isin(train_ids)
    test_mask = X['game_id'].isin(test_ids)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    
    logger = ExperimentLogger()
    
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    
    # Preserve game_id for bootstrap sampling (before filtering to features)
    X_train_game_ids = X_train['game_id'].copy()
    X_test_game_ids = X_test['game_id'].copy()
    
    # Filter X to only include feature cols
    available_cols = [c for c in feature_cols if c in X_train.columns]
    X_train_features = X_train[available_cols]
    X_test_features = X_test[available_cols]
    
    print(f"Using {len(available_cols)} features.")
    print(f"Models to train: {', '.join(models_to_train)}\n")
    
    trained_models = {}
    
    # =========================================================================
    # 2. Logistic Regression Ensemble (Bootstrap)
    # =========================================================================
    if 'lr' in models_to_train:
        print("=" * 80)
        print("Training Logistic Regression Ensemble with Bootstrap...")
        print("=" * 80)
        
        ensemble_models = []
        n_models = 10
        
        # Get unique game IDs for game-level bootstrap
        train_game_ids = X_train_game_ids.unique()
        
        for i in range(n_models):
            print(f"  Training model {i+1}/{n_models}...", end=' ')
            
            # Game-level bootstrap sampling
            # Sample game_ids with replacement to ensure different models see different games
            boot_game_ids = np.random.choice(train_game_ids, len(train_game_ids), replace=True)
            boot_mask = X_train_game_ids.isin(boot_game_ids)
            X_boot = X_train_features[boot_mask]
            y_boot = y_train[boot_mask]
            
            # Train model on bootstrap sample
            lr_model = LogisticRegression(
                max_iter=3000,
                solver='saga',      # Better for large datasets
                n_jobs=-1,          # Parallel processing
                random_state=i
            )
            
            lr_model.fit(X_boot, y_boot)
            ensemble_models.append(lr_model)
            print(f"Trained on {len(boot_game_ids)} games, {len(X_boot)} events")
        
        print(f"\nTrained {len(ensemble_models)} models in ensemble")
        
        # Evaluate ensemble
        import time
        start_time = time.time()
        ensemble_preds = np.array([m.predict_proba(X_test_features)[:, 1] for m in ensemble_models])
        mean_preds = np.mean(ensemble_preds, axis=0)
        latency_ms = (time.time() - start_time) * 1000 / len(X_test_features)
        
        lr_loss = log_loss(y_test, mean_preds)
        lr_acc = accuracy_score(y_test, mean_preds > 0.5)
        lr_auc = roc_auc_score(y_test, mean_preds)
        
        print(f"\nEnsemble Performance:")
        print(f"Log Loss: {lr_loss:.4f}")
        print(f"Accuracy: {lr_acc:.4f}")
        print(f"AUC: {lr_auc:.4f}")
        print(f"Latency: {latency_ms:.4f} ms/row")
        
        logger.log_experiment(
            model_type='LogisticRegressionEnsemble',
            features=available_cols,
            hyperparams={
                'max_iter': 3000, 
                'solver': 'saga', 
                'n_models': n_models,
                'sampling': 'game-level bootstrap'
            },
            metrics={'accuracy': lr_acc, 'log_loss': lr_loss, 'auc': lr_auc, 'latency_ms': latency_ms},
            notes='Bootstrap ensemble with game-level sampling for proper CIs'
        )
        
        # Save LR ensemble
        import joblib
        lr_path = 'models/nba_lr_model.pkl'
        joblib.dump(ensemble_models, lr_path)
        print(f"✓ Saved to '{lr_path}'\n")
        
        trained_models['lr'] = ensemble_models
        trained_models['lr_probs'] = mean_preds
    
    # =========================================================================
    # 3. XGBoost Ensemble
    # =========================================================================
    if 'xgboost' in models_to_train:
        print("=" * 80)
        print("Training XGBoost Ensemble with Early Stopping...")
        print("=" * 80)
        
        ensemble_models = []
        n_models = 5
        
        for i in range(n_models):
            print(f"  Training model {i+1}/{n_models}...", end=' ')
            
            # Bootstrap sampling
            indices = np.random.choice(len(X_train_features), len(X_train_features), replace=True)
            X_boot = X_train_features.iloc[indices]
            y_boot = y_train[indices]
            
            # Split bootstrap sample into train/validation for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_boot, y_boot, test_size=0.2, random_state=i
            )
            
            model = xgb.XGBClassifier(
                n_estimators=1000,         # More trees, early stopping will cut off
                learning_rate=0.03,        # Much lower
                max_depth=3,               # Slightly shallower
                min_child_weight=3,        # More conservative splits
                subsample=0.8,             # Row sampling (regularization)
                colsample_bytree=0.8,      # Column sampling (regularization)
                gamma=0.1,                 # Min loss reduction for split
                reg_alpha=0.1,             # L1 regularization
                reg_lambda=1.0,            # L2 regularization
                objective='binary:logistic',
                eval_metric='logloss',
                early_stopping_rounds=50,  # Early stopping in constructor (new XGBoost)
                random_state=i
            )
            
            # Fit with early stopping
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            ensemble_models.append(model)
            print(f"Best iteration: {model.best_iteration}")
        
        print(f"\nTrained {len(ensemble_models)} models in ensemble")
        
        # Evaluate
        import time
        start_time = time.time()
        ensemble_preds = np.array([m.predict_proba(X_test_features)[:, 1] for m in ensemble_models])
        mean_preds = np.mean(ensemble_preds, axis=0)
        latency_ms = (time.time() - start_time) * 1000 / len(X_test_features)
        
        ensemble_loss = log_loss(y_test, mean_preds)
        ensemble_acc = accuracy_score(y_test, mean_preds > 0.5)
        xgb_auc = roc_auc_score(y_test, mean_preds)
        
        print(f"\nEnsemble Performance:")
        print(f"Log Loss: {ensemble_loss:.4f}")
        print(f"Accuracy: {ensemble_acc:.4f}")
        print(f"AUC: {xgb_auc:.4f}")
        print(f"Latency: {latency_ms:.4f} ms/row")
        
        logger.log_experiment(
            model_type='XGBoostEnsemble',
            features=available_cols,
            hyperparams={
                'n_estimators': 1000, 
                'learning_rate': 0.03, 
                'max_depth': 3, 
                'n_models': 5,
                'early_stopping_rounds': 50,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            metrics={'accuracy': ensemble_acc, 'log_loss': ensemble_loss, 'auc': xgb_auc, 'latency_ms': latency_ms},
            notes='Early stopping with validation, reduced features (no season stats)'
        )
        
        # Save XGBoost ensemble
        import joblib
        xgb_path = 'models/nba_xgboost_ensemble.pkl'
        joblib.dump(ensemble_models, xgb_path)
        print(f"✓ Saved to '{xgb_path}'\n")
        
        trained_models['xgboost'] = ensemble_models
        trained_models['xgb_probs'] = mean_preds
    
    # =========================================================================
    # Comparison (if both models trained)
    # =========================================================================
    if 'lr' in trained_models and 'xgboost' in trained_models:
        print("=" * 80)
        print("Model Comparison")
        print("=" * 80)
        
        lr_probs = trained_models['lr_probs']
        xgb_probs = trained_models['xgb_probs']
        
        lr_auc = roc_auc_score(y_test, lr_probs)
        xgb_auc = roc_auc_score(y_test, xgb_probs)
        
        print(f"\nLR AUC: {lr_auc:.4f}")
        print(f"XGBoost AUC: {xgb_auc:.4f}")
        
        # Check if predictions are similar
        correlation = np.corrcoef(lr_probs, xgb_probs)[0, 1]
        print(f"Prediction correlation: {correlation:.4f}")
        
        print("\n" + "=" * 80)
    
    return {k: v for k, v in trained_models.items() if k in ['lr', 'xgboost']}

if __name__ == "__main__":
    train_models(models_to_train=['lr', 'xgboost'])
