"""
Historical backtest: Find spread betting opportunities on test set.

Simulates what would have happened if you traded spreads in past games.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import sys
import os
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from src.data.database import DatabaseManager
from src.features.engineering import BASE_FEATURES_LIST, ADVANCED_FEATURES_LIST


def get_final_score_diffs(X: pd.DataFrame) -> np.ndarray:
    """Get final score differential for each game."""
    db = DatabaseManager()
    final_diffs = []
    
    for game_id in X['game_id'].unique():
        query = f"""
            SELECT home_score, away_score 
            FROM pbp_events 
            WHERE game_id = '{game_id}' 
            ORDER BY period DESC, remaining_time ASC 
            LIMIT 1
        """
        result = pd.read_sql(query, db.engine)
        
        if not result.empty:
            final_diff = result['home_score'].iloc[0] - result['away_score'].iloc[0]
        else:
            final_diff = 0
        
        game_rows = len(X[X['game_id'] == game_id])
        final_diffs.extend([final_diff] * game_rows)
    
    return np.array(final_diffs)


def simulate_market_price(actual_diff, threshold, noise=0.05):
    """
    Simulate what market price might have been.
    
    Assumes market is somewhat efficient but noisy.
    Actual edge = model_prob - market_prob
    """
    # "True" probability (what market should be if perfect)
    if actual_diff > threshold:
        true_prob = 0.70  # Home actually wins by more
    else:
        true_prob = 0.30  # Home doesn't win by that much
    
    # Add noise to simulate market inefficiency
    market_prob = true_prob + np.random.normal(0, noise)
    market_prob = np.clip(market_prob, 0.05, 0.95)
    
    return market_prob


def backtest_spread_bets():
    """
    Backtest spread betting strategy.
    
    For each test event:
    1. Get model prediction for P(diff > threshold)
    2. Simulate what market price would be
    3. If edge > 5%, count as betting opportunity
    4. Track if bet would have won
    """
    print("=" * 80)
    print("SPREAD BETTING BACKTEST")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X, _ = prepare_training_data()
    final_diffs = get_final_score_diffs(X)
    
    # Split
    game_ids = X['game_id'].unique()
    _, test_ids = train_test_split(game_ids, test_size=0.2, random_state=42)
    test_mask = X['game_id'].isin(test_ids)
    
    feature_cols = BASE_FEATURES_LIST + ADVANCED_FEATURES_LIST
    available_cols = [c for c in feature_cols if c in X.columns]
    
    X_test = X[test_mask][available_cols]
    y_test = final_diffs[test_mask]
    
    print(f"Test set: {len(X_test)} events from {len(test_ids)} games")
    
    # Load model
    print("\nLoading Ridge model...")
    models = joblib.load('models/nba_spread_model.pkl')
    
    # Predict
    print("Generating predictions...")
    mean_preds = []
    std_preds = []
    
    for model_pair in models:
        mean_pred = model_pair['mean_model'].predict(X_test)
        std_pred = model_pair['std_model'].predict(X_test)
        mean_preds.append(mean_pred)
        std_preds.append(std_pred)
    
    mean_pred = np.mean(mean_preds, axis=0)
    std_pred = np.mean(std_preds, axis=0)
    
    # Test spread thresholds
    thresholds = [3.5, 6.5, 9.5]
    
    print("\n" + "=" * 80)
    print("BETTING OPPORTUNITIES")
    print("=" * 80)
    
    total_opportunities = 0
    total_wins = 0
    total_profit = 0
    
    for threshold in thresholds:
        print(f"\n### Spread: >{threshold} Points ###")
        
        opportunities = 0
        wins = 0
        profit = 0
        
        for i in range(len(X_test)):
            # Model prediction
            dist = stats.norm(loc=mean_pred[i], scale=max(std_pred[i], 1.0))
            model_prob = 1 - dist.cdf(threshold)
            
            # Simulate market (assumes some efficiency + noise)
            # In reality, you'd get this from Kalshi API
            market_prob = simulate_market_price(y_test.iloc[i], threshold, noise=0.08)
            
            # Edge
            edge = model_prob - market_prob
            
            # Would we bet?
            if edge > 0.05:  # 5% minimum edge
                opportunities += 1
                
                # Did we win?
                actual_diff = y_test.iloc[i]
                if actual_diff > threshold:
                    wins += 1
                    profit += (1 - market_prob)  # Profit = (1 - price) if win
                else:
                    profit -= market_prob  # Loss = -price if lose
        
        if opportunities > 0:
            win_rate = wins / opportunities
            avg_profit = profit / opportunities
            
            print(f"Opportunities: {opportunities}")
            print(f"Bets Won:     {wins} ({win_rate:.1%})")
            print(f"Avg Profit:   ${avg_profit:.2f} per $1 bet")
            print(f"Total Profit: ${profit:.2f} (on {opportunities} $1 bets)")
            
            total_opportunities += opportunities
            total_wins += wins
            total_profit += profit
        else:
            print("No opportunities found (edge < 5%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Opportunities: {total_opportunities}")
    print(f"Win Rate:            {total_wins/total_opportunities:.1%}")
    print(f"Total Profit:        ${total_profit:.2f}")
    print(f"Per Event:           ${total_profit/len(X_test):.4f}")
    print(f"ROI:                 {total_profit/total_opportunities:.1%}")
    
    # Events per game
    events_per_game = len(X_test) / len(test_ids)
    opps_per_game = total_opportunities / len(test_ids)
    profit_per_game = total_profit / len(test_ids)
    
    print(f"\nPer Game:")
    print(f"  Events:        {events_per_game:.0f}")
    print(f"  Opportunities: {opps_per_game:.1f}")
    print(f"  Profit:        ${profit_per_game:.2f}")
    
    # Extrapolate to full season
    print(f"\nExtrapolated to 82 games:")
    print(f"  Opportunities: {opps_per_game * 82:.0f}")
    print(f"  Profit:        ${profit_per_game * 82:.2f}")
    
    print("\n" + "=" * 80)
    print("NOTE: This uses SIMULATED market prices.")
    print("Real markets may be more/less efficient.")
    print("Run 'live_market_check.py' to see actual Kalshi spreads.")
    print("=" * 80)


if __name__ == "__main__":
    # Set seed for reproducible market simulation
    np.random.seed(42)
    backtest_spread_bets()
