import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from src.data.database import DatabaseManager
from src.features.engineering import create_live_features, add_advanced_features, BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST

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

def train_models():
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
    
    # Filter X to only include feature cols
    available_cols = [c for c in feature_cols if c in X_train.columns]
    X_train = X_train[available_cols]
    X_test = X_test[available_cols]
    
    print(f"Using {len(available_cols)} features.")
    
    # 2. Logistic Regression Baseline
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=5000)
    lr_model.fit(X_train, y_train)
    
    import time
    start_time = time.time()
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    latency_ms = (time.time() - start_time) * 1000 / len(X_test)
    
    lr_loss = log_loss(y_test, lr_probs)
    lr_acc = accuracy_score(y_test, lr_probs > 0.5)
    
    print(f"Logistic Regression - Log Loss: {lr_loss:.4f}, Accuracy: {lr_acc:.4f}, Latency: {latency_ms:.4f} ms/row")
    
    logger.log_experiment(
        model_type='LogisticRegression',
        features=available_cols,
        hyperparams={'max_iter': 5000},
        metrics={'accuracy': lr_acc, 'log_loss': lr_loss, 'latency_ms': latency_ms},
        notes='Advanced features-all games'
    )
    
    # 3. XGBoost Ensemble
    print("\nTraining XGBoost Ensemble...")
    ensemble_models = []
    n_models = 5
    
    for i in range(n_models):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train[indices]
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=i
        )
        model.fit(X_boot, y_boot)
        ensemble_models.append(model)
    
    # Evaluate
    start_time = time.time()
    ensemble_preds = np.array([m.predict_proba(X_test)[:, 1] for m in ensemble_models])
    mean_preds = np.mean(ensemble_preds, axis=0)
    latency_ms = (time.time() - start_time) * 1000 / len(X_test)
    
    ensemble_loss = log_loss(y_test, mean_preds)
    ensemble_acc = accuracy_score(y_test, mean_preds > 0.5)
    
    print(f"Ensemble (5 models) - Log Loss: {ensemble_loss:.4f}, Accuracy: {ensemble_acc:.4f}, Latency: {latency_ms:.4f} ms/row")
    
    logger.log_experiment(
        model_type='XGBoostEnsemble',
        features=available_cols,
        hyperparams={'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 5, 'n_models': 5},
        metrics={'accuracy': ensemble_acc, 'log_loss': ensemble_loss, 'latency_ms': latency_ms},
        notes='Advanced features-all games'
    )
    
    # Save models
    import joblib
    joblib.dump(ensemble_models, 'models/nba_live_model_ensemble.pkl')
    print("Ensemble saved to 'models/nba_live_model_ensemble.pkl'")
    
    return lr_model, ensemble_models

if __name__ == "__main__":
    train_models()
