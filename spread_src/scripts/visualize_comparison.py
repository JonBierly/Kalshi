import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results():
    try:
        df = pd.read_csv("data/model_vs_market_comparison.csv")
    except FileNotFoundError:
        print("Comparison results not found. Run compare_model_vs_market.py first.")
        return

    sns.set_theme(style="whitegrid")
    
    # 1. Brier Score by Game Time
    plt.figure(figsize=(12, 6))
    
    # Bin seconds remaining into 5-minute intervals
    df['minutes_remaining'] = df['seconds_remaining'] / 60
    df['time_bin'] = pd.cut(df['minutes_remaining'], bins=np.arange(0, 50, 4)) # 4 min intervals
    
    brier_by_time = df.groupby('time_bin', observed=True)[['model_brier', 'market_brier']].mean().sort_index(ascending=False)
    
    brier_by_time.plot(kind='line', marker='o', figsize=(12, 6))
    plt.title("Brier Score vs. Game Time (Lower is Better)")
    plt.xlabel("Minutes Remaining")
    plt.ylabel("Mean Brier Score")
    plt.gca().invert_xaxis()
    plt.legend(["NGBoost Model", "Kalshi Market mid-price"])
    plt.savefig("reports/model_vs_market_brier_over_time.png")
    print("Saved reports/model_vs_market_brier_over_time.png")

    # 2. Calibration Plot
    plt.figure(figsize=(8, 8))
    
    # Bin probabilities
    df['model_prob_bin'] = pd.cut(df['model_prob'], bins=np.arange(0, 1.1, 0.1), labels=np.arange(0.05, 1.05, 0.1))
    df['market_prob_bin'] = pd.cut(df['market_prob'], bins=np.arange(0, 1.1, 0.1), labels=np.arange(0.05, 1.05, 0.1))
    
    model_cal = df.groupby('model_prob_bin', observed=True)['outcome'].mean()
    market_cal = df.groupby('market_prob_bin', observed=True)['outcome'].mean()
    
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(model_cal.index.astype(float), model_cal.values, 'o-', label="NGBoost Model")
    plt.plot(market_cal.index.astype(float), market_cal.values, 's-', label="Kalshi Market")
    
    plt.title("Calibration Curve (Model vs Market)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/model_vs_market_calibration.png")
    print("Saved reports/model_vs_market_calibration.png")

    # 3. Trade Performance & Edge Analysis
    trades_df = df[df['is_trade'] == 1].copy()
    
    if not trades_df.empty:
        plt.figure(figsize=(10, 6))
        # Bin Captured Edge
        trades_df['edge_bin'] = pd.cut(trades_df['captured_edge'], bins=np.arange(0, 0.51, 0.05), labels=np.arange(0.025, 0.5, 0.05))
        edge_perf = trades_df.groupby('edge_bin', observed=True)['model_brier'].mean()
        
        edge_perf.plot(marker='s', color='green')
        plt.title("Model Brier Score vs. Captured Edge (Signals Only)")
        plt.xlabel("Captured Edge (0.05 bins)")
        plt.ylabel("Mean Brier Score")
        plt.grid(True)
        plt.savefig("reports/model_edge_performance.png")
        print("Saved reports/model_edge_performance.png")

    # 4. Summary Stats (Updated)
    trade_count = df['is_trade'].sum()
    trade_pct = (trade_count / len(df)) * 100
    avg_edge = df[df['is_trade']==1]['captured_edge'].mean() if trade_count > 0 else 0
    
    summary = f"""
Model vs Market Comparison Summary (Precise Anchors)
==================================================
Observations: {len(df)}
Unique Markets: {df['ticker'].nunique()}
Unique Games: {df['game_id'].nunique()}

Mean Brier Scores (All):
- Model:  {df['model_brier'].mean():.4f}
- Market: {df['market_brier'].mean():.4f}

Trade Analysis:
- Signals Found: {trade_count} ({trade_pct:.1f}% of observations)
- Avg Captured Edge: {avg_edge:.4f}
- Model Brier (on Signals): {df[df['is_trade']==1]['model_brier'].mean():.4f} if trade_count > 0 else "N/A"
- Market Brier (on Signals): {df[df['is_trade']==1]['market_brier'].mean():.4f} if trade_count > 0 else "N/A"
"""
    print(summary)
    with open("reports/model_vs_market_summary.txt", "w") as f:
        f.write(summary)
    print("Saved reports/model_vs_market_summary.txt")

if __name__ == "__main__":
    visualize_results()
