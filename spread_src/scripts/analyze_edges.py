"""
Analyze edge distribution on test set.

Shows:
- How often edges appear
- How big edges are
- Edge by game state (close vs blowout, early vs late)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import sys
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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


def analyze_edge_distribution():
    """Analyze when and how often edges appear."""
    print("=" * 80)
    print("EDGE DISTRIBUTION ANALYSIS")
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
    
    X_test_full = X[test_mask]
    X_test = X_test_full[available_cols]
    y_test = final_diffs[test_mask]
    
    # Load model
    print("Loading model...")
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
    
    # Analyze edges at different thresholds
    threshold = 6.5  # Example spread
    
    print(f"\n" + "=" * 80)
    print(f"EDGE ANALYSIS: Spread >{threshold} points")
    print("=" * 80)
    
    # Model probabilities
    model_probs = []
    for mean, std in zip(mean_pred, std_pred):
        dist = stats.norm(loc=mean, scale=max(std, 1.0))
        prob = 1 - dist.cdf(threshold)
        model_probs.append(prob)
    
    model_probs = np.array(model_probs)
    
    # "True" probability (actual outcome)
    true_outcomes = (y_test > threshold).values.astype(float)
    
    # Simulated market (efficient but noisy)
    np.random.seed(42)
    market_probs = []
    for outcome in true_outcomes:
        if outcome == 1:
            market_prob = 0.65 + np.random.normal(0, 0.08)
        else:
            market_prob = 0.35 + np.random.normal(0, 0.08)
        market_probs.append(np.clip(market_prob, 0.05, 0.95))
    
    market_probs = np.array(market_probs)
    
    # Edges
    edges = model_probs - market_probs
    
    # Analysis
    print(f"\nEdge Distribution:")
    print(f"  Mean edge:    {np.mean(edges):+.1%}")
    print(f"  Std edge:     {np.std(edges):.1%}")
    print(f"  Min edge:     {np.min(edges):+.1%}")
    print(f"  Max edge:     {np.max(edges):+.1%}")
    
    # Edge buckets
    print(f"\nEdge Frequency:")
    print(f"  Edge > +10%:  {np.mean(edges > 0.10):.1%} of events")
    print(f"  Edge > +5%:   {np.mean(edges > 0.05):.1%} of events")
    print(f"  Edge > +3%:   {np.mean(edges > 0.03):.1%} of events")
    print(f"  |Edge| < 3%:  {np.mean(np.abs(edges) < 0.03):.1%} of events (no trade)")
    print(f"  Edge < -3%:   {np.mean(edges < -0.03):.1%} of events")
    print(f"  Edge < -5%:   {np.mean(edges < -0.05):.1%} of events")
    
    # By game state
    score_diff = X_test_full['score_diff'].values
    seconds_remaining = X_test_full['seconds_remaining'].values
    
    print(f"\n" + "=" * 80)
    print("EDGE BY GAME STATE")
    print("=" * 80)
    
    # Close games vs blowouts
    close_mask = np.abs(score_diff) < 10
    blowout_mask = np.abs(score_diff) >= 10
    
    print(f"\nClose Games (|diff| < 10):")
    print(f"  Events:        {np.sum(close_mask)}")
    print(f"  Avg |edge|:    {np.mean(np.abs(edges[close_mask])):.1%}")
    print(f"  Edge > 5%:     {np.mean(edges[close_mask] > 0.05):.1%}")
    
    print(f"\nBlowouts (|diff| >= 10):")
    print(f"  Events:        {np.sum(blowout_mask)}")
    print(f"  Avg |edge|:    {np.mean(np.abs(edges[blowout_mask])):.1%}")
    print(f"  Edge > 5%:     {np.mean(edges[blowout_mask] > 0.05):.1%}")
    
    # Early vs late game
    early_mask = seconds_remaining > 1440  # > 24 mins (Q1-Q2)
    late_mask = seconds_remaining <= 720   # <= 12 mins (Q4)
    
    print(f"\nEarly Game (>24 mins left):")
    print(f"  Events:        {np.sum(early_mask)}")
    print(f"  Avg |edge|:    {np.mean(np.abs(edges[early_mask])):.1%}")
    print(f"  Edge > 5%:     {np.mean(edges[early_mask] > 0.05):.1%}")
    
    print(f"\nLate Game (<=12 mins left):")
    print(f"  Events:        {np.sum(late_mask)}")
    print(f"  Avg |edge|:    {np.mean(np.abs(edges[late_mask])):.1%}")
    print(f"  Edge > 5%:     {np.mean(edges[late_mask] > 0.05):.1%}")
    
    # ROI analysis
    print(f"\n" + "=" * 80)
    print("EXPECTED ROI (if edges are real)")
    print("=" * 80)
    
    # Only trade when edge > 5%
    trade_mask = edges > 0.05
    
    if np.sum(trade_mask) > 0:
        trade_model_probs = model_probs[trade_mask]
        trade_market_probs = market_probs[trade_mask]
        trade_outcomes = true_outcomes[trade_mask]
        
        # P&L
        profits = []
        for model_prob, market_prob, outcome in zip(trade_model_probs, trade_market_probs, trade_outcomes):
            if outcome == 1:  # Win
                profit = (1 - market_prob)
            else:  # Lose
                profit = -market_prob
            profits.append(profit)
        
        profits = np.array(profits)
        
        print(f"\nTrades:        {len(profits)}")
        print(f"Win rate:      {np.mean(trade_outcomes):.1%}")
        print(f"Avg profit:    ${np.mean(profits):.3f} per $1 bet")
        print(f"Total profit:  ${np.sum(profits):.2f}")
        print(f"Sharpe ratio:  {np.mean(profits) / np.std(profits):.2f}")
        
        # Per game
        events_per_game = len(X_test) / len(test_ids)
        trades_per_game = len(profits) / len(test_ids)
        profit_per_game = np.sum(profits) / len(test_ids)
        
        print(f"\nPer Game:")
        print(f"  Trades:   {trades_per_game:.1f}")
        print(f"  Profit:   ${profit_per_game:.2f}")
    else:
        print("\nNo trades with edge > 5%")
    
    print("\n" + "=" * 80)
    print("NOTE: Uses simulated 'efficient' markets")
    print("Real edges depend on actual Kalshi pricing efficiency")
    print("=" * 80)


if __name__ == "__main__":
    analyze_edge_distribution()
