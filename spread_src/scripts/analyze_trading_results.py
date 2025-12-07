#!/usr/bin/env python
"""
Analyze live trading results from SQLite database.

Generates comprehensive performance reports including:
- P&L over time
- Win rate analysis
- Model prediction accuracy
- Trade distribution
- Risk metrics
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

DB_PATH = 'data/nba_data.db'

def load_data():
    """Load all data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    
    # Load trades
    trades = pd.read_sql_query("""
        SELECT *,
               datetime(timestamp) as dt
        FROM trades
        ORDER BY timestamp
    """, conn)
    
    # Load predictions
    predictions = pd.read_sql_query("""
        SELECT *,
               datetime(timestamp) as dt
        FROM model_predictions
        ORDER BY timestamp
    """, conn)
    
    # Load performance metrics
    metrics = pd.read_sql_query("""
        SELECT *
        FROM performance_metrics
        ORDER BY date
    """, conn)
    
    conn.close()
    
    print(f"✓ Loaded {len(trades)} trades")
    print(f"✓ Loaded {len(predictions)} predictions")
    print(f"✓ Loaded {len(metrics)} metric rows")
    
    return trades, predictions, metrics


def analyze_pnl(trades):
    """Analyze P&L performance."""
    print("\n" + "=" * 60)
    print("📊 P&L ANALYSIS")
    print("=" * 60)
    
    # Filter to closed trades
    closed = trades[trades['status'] == 'closed'].copy()
    
    if len(closed) == 0:
        print("⚠️  No closed trades yet!")
        return
    
    # Summary stats
    total_pnl = closed['realized_pnl'].sum()
    num_trades = len(closed)
    wins = len(closed[closed['realized_pnl'] > 0])
    losses = len(closed[closed['realized_pnl'] < 0])
    win_rate = wins / num_trades if num_trades > 0 else 0
    
    avg_win = closed[closed['realized_pnl'] > 0]['realized_pnl'].mean() if wins > 0 else 0
    avg_loss = closed[closed['realized_pnl'] < 0]['realized_pnl'].mean() if losses > 0 else 0
    
    print(f"\n📈 Overall Performance")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Total Trades: {num_trades}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    if avg_loss != 0:
        print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x")
    
    # Plot cumulative P&L
    closed['cumulative_pnl'] = closed['realized_pnl'].cumsum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative P&L
    ax1.plot(range(len(closed)), closed['cumulative_pnl'], 
             linewidth=2, color='green' if total_pnl > 0 else 'red')
    ax1.fill_between(range(len(closed)), 0, closed['cumulative_pnl'], 
                      alpha=0.3, color='green' if total_pnl > 0 else 'red')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Cumulative P&L Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative P&L ($)')
    ax1.grid(True, alpha=0.3)
    
    # Individual trade P&L
    colors = ['green' if x > 0 else 'red' for x in closed['realized_pnl']]
    ax2.bar(range(len(closed)), closed['realized_pnl'], color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/analysis_pnl.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/analysis_pnl.png")
    

def analyze_model_accuracy(predictions):
    """Analyze model prediction accuracy."""
    print("\n" + "=" * 60)
    print("🎯 MODEL ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Filter to settled predictions
    settled = predictions[predictions['actual_outcome'].notna()].copy()
    
    if len(settled) == 0:
        print("⚠️  No settled predictions yet!")
        return
    
    # Calculate accuracy
    settled['correct'] = (settled['predicted_prob'] > 0.5) == (settled['actual_outcome'] == 1)
    accuracy = settled['correct'].mean()
    
    # Calibration analysis
    settled['prob_bucket'] = pd.cut(settled['predicted_prob'], 
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    calibration = settled.groupby('prob_bucket').agg({
        'actual_outcome': 'mean',
        'predicted_prob': ['mean', 'count']
    }).round(3)
    
    print(f"\n📊 Overall Accuracy: {accuracy:.1%}")
    print(f"  Total Predictions: {len(settled)}")
    print(f"  Correct: {settled['correct'].sum()}")
    print(f"  Incorrect: {(~settled['correct']).sum()}")
    
    print(f"\n📊 Calibration by Confidence:")
    print(calibration)
    
    # Plot calibration curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration curve
    cal_data = settled.groupby('prob_bucket').agg({
        'actual_outcome': 'mean',
        'predicted_prob': 'mean'
    })
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    ax1.scatter(cal_data['predicted_prob'], cal_data['actual_outcome'], 
                s=200, alpha=0.7, color='blue')
    for idx, row in cal_data.iterrows():
        ax1.annotate(idx, (row['predicted_prob'], row['actual_outcome']),
                     xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Actual Outcome Rate')
    ax1.set_title('Model Calibration Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Prediction error distribution
    ax2.hist(settled['prediction_error'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/analysis_model.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/analysis_model.png")


def analyze_trade_distribution(trades):
    """Analyze trade patterns."""
    print("\n" + "=" * 60)
    print("📦 TRADE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Filter trades with fills
    filled = trades[trades['fill_price'].notna()].copy()
    
    if len(filled) == 0:
        print("⚠️  No filled trades yet!")
        return
    
    # Extract game from ticker
    filled['game'] = filled['ticker'].str.extract(r'25DEC\d{2}([A-Z]+)')
    
    # Side analysis
    side_counts = filled['side'].value_counts()
    print(f"\n📊 Trade Sides:")
    for side, count in side_counts.items():
        print(f"  {side.upper()}: {count} ({count/len(filled):.1%})")
    
    # Game analysis
    game_counts = filled['game'].value_counts().head(10)
    print(f"\n📊 Top Games (by trade count):")
    for game, count in game_counts.items():
        print(f"  {game}: {count} trades")
    
    # Price distribution
    print(f"\n📊 Fill Price Distribution:")
    print(f"  Mean: {filled['fill_price'].mean():.1f}¢")
    print(f"  Median: {filled['fill_price'].median():.1f}¢")
    print(f"  Std: {filled['fill_price'].std():.1f}¢")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Side distribution
    side_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Buy vs Sell Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Side')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Game distribution
    game_counts.plot(kind='barh', ax=axes[0, 1], color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Top 10 Games by Trade Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Count')
    
    # Fill price distribution
    axes[1, 0].hist(filled['fill_price'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=filled['fill_price'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {filled["fill_price"].mean():.1f}¢')
    axes[1, 0].set_title('Fill Price Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Price (cents)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Size distribution
    axes[1, 1].hist(filled['size'], bins=range(1, filled['size'].max()+2), 
                    alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Position Size Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Size (contracts)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data/analysis_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/analysis_distribution.png")


def analyze_risk_metrics(trades):
    """Analyze risk and exposure metrics."""
    print("\n" + "=" * 60)
    print("⚠️  RISK METRICS ANALYSIS")
    print("=" * 60)
    
    closed = trades[trades['status'] == 'closed'].copy()
    
    if len(closed) == 0:
        print("⚠️  No closed trades yet!")
        return
    
    # Calculate drawdown
    closed['cumulative_pnl'] = closed['realized_pnl'].cumsum()
    closed['running_max'] = closed['cumulative_pnl'].cummax()
    closed['drawdown'] = closed['cumulative_pnl'] - closed['running_max']
    
    max_drawdown = closed['drawdown'].min()
    
    # Consecutive wins/losses
    closed['win'] = closed['realized_pnl'] > 0
    closed['streak'] = (closed['win'] != closed['win'].shift()).cumsum()
    streaks = closed.groupby('streak')['win'].agg(['first', 'count'])
    
    max_win_streak = streaks[streaks['first']]['count'].max() if len(streaks[streaks['first']]) > 0 else 0
    max_loss_streak = streaks[~streaks['first']]['count'].max() if len(streaks[~streaks['first']]) > 0 else 0
    
    print(f"\n📉 Risk Metrics:")
    print(f"  Max Drawdown: ${max_drawdown:.2f}")
    print(f"  Max Win Streak: {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")
    
    # Sharpe-like ratio (simplified)
    returns = closed['realized_pnl']
    if len(returns) > 1:
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        print(f"  Sharpe-like Ratio: {sharpe:.2f}")
    
    # Plot drawdown
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(range(len(closed)), 0, closed['drawdown'], 
                     color='red', alpha=0.3, label='Drawdown')
    ax.plot(range(len(closed)), closed['drawdown'], color='darkred', linewidth=2)
    ax.axhline(y=max_drawdown, color='red', linestyle='--', 
               linewidth=2, label=f'Max Drawdown: ${max_drawdown:.2f}')
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Drawdown ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/analysis_risk.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: data/analysis_risk.png")


def generate_summary_report(trades, predictions, metrics):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 60)
    print("📋 SUMMARY REPORT")
    print("=" * 60)
    
    # Trading period
    if len(trades) > 0:
        start_time = trades['timestamp'].min()
        end_time = trades['timestamp'].max()
        print(f"\n🕐 Trading Period:")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
    
    # Quick stats
    closed = trades[trades['status'] == 'closed']
    filled = trades[trades['fill_price'].notna()]
    
    print(f"\n📊 Trade Summary:")
    print(f"  Total Orders: {len(trades)}")
    print(f"  Filled: {len(filled)}")
    print(f"  Closed: {len(closed)}")
    print(f"  Fill Rate: {len(filled)/len(trades):.1%}" if len(trades) > 0 else "  N/A")
    
    if len(closed) > 0:
        print(f"\n💰 P&L Summary:")
        print(f"  Total P&L: ${closed['realized_pnl'].sum():+.2f}")
        print(f"  Best Trade: ${closed['realized_pnl'].max():.2f}")
        print(f"  Worst Trade: ${closed['realized_pnl'].min():.2f}")
    
    # Prediction summary
    settled_preds = predictions[predictions['actual_outcome'].notna()]
    if len(settled_preds) > 0:
        settled_preds['correct'] = (settled_preds['predicted_prob'] > 0.5) == (settled_preds['actual_outcome'] == 1)
        print(f"\n🎯 Model Performance:")
        print(f"  Predictions: {len(settled_preds)}")
        print(f"  Accuracy: {settled_preds['correct'].mean():.1%}")
        print(f"  Avg Error: {abs(settled_preds['prediction_error']).mean():.3f}")


def main():
    """Main analysis function."""
    print("\n" + "=" * 60)
    print("🔬 LIVE TRADING ANALYSIS")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    trades, predictions, metrics = load_data()
    
    # Run analyses
    if len(trades) > 0:
        analyze_pnl(trades)
        analyze_trade_distribution(trades)
        analyze_risk_metrics(trades)
    
    if len(predictions) > 0:
        analyze_model_accuracy(predictions)
    
    # Summary
    generate_summary_report(trades, predictions, metrics)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/analysis_pnl.png")
    print("  - data/analysis_model.png")
    print("  - data/analysis_distribution.png")
    print("  - data/analysis_risk.png")
    print("\n")


if __name__ == "__main__":
    main()
