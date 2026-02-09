import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

def analyze_calibration(db_path='data/nba_data.db', game_id='0022500702'):
    conn = sqlite3.connect(db_path)
    
    # Get final scores for derived outcome
    game_meta = pd.read_sql_query(f"SELECT home_score, away_score FROM games WHERE game_id = '{game_id}'", conn)
    if game_meta.empty:
        # Try finding it in pbp_events
        game_meta = pd.read_sql_query(f"SELECT home_score, away_score FROM pbp_events WHERE game_id = '{game_id}' ORDER BY timestamp DESC LIMIT 1", conn)
    
    if game_meta.empty:
        print(f"Could not find final score for {game_id}")
        conn.close()
        return
        
    final_home = game_meta.iloc[0]['home_score']
    final_away = game_meta.iloc[0]['away_score']
    final_diff = final_home - final_away
    print(f"Game {game_id} Final Score: Home {final_home}, Away {final_away} (Diff: {final_diff})")

    # Query predictions
    query = f"""
    SELECT predicted_prob, ticker, seconds_remaining
    FROM model_predictions
    WHERE game_id = '{game_id}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No predictions found for game {game_id}")
        return

    # Derive actual outcome from final score and ticker
    def derive_outcome(row):
        ticker = row['ticker']
        parts = ticker.split('-')
        market_part = parts[-1] 
        
        target_team = "".join([c for c in market_part if c.isalpha()])
        try:
            # Extract threshold (digits and dots)
            threshold_str = "".join([c for c in market_part if c.isdigit() or c == '.'])
            threshold = float(threshold_str) if threshold_str else 0
        except:
            threshold = 0
            
        if "BOS" in market_part:
            return 1.0 if final_diff > threshold else 0.0
        elif "MIL" in market_part:
            return 1.0 if (-final_diff) > threshold else 0.0
        elif "DET" in market_part:
            return 1.0 if final_diff > threshold else 0.0
        elif "BKN" in market_part:
            return 1.0 if (-final_diff) > threshold else 0.0
        return 0.0

    df['actual_outcome'] = df.apply(derive_outcome, axis=1)

    # 1. Overall Brier Score
    df['squared_error'] = (df['predicted_prob'] - df['actual_outcome'].astype(float))**2
    brier_score = df['squared_error'].mean()
    print(f"Brier Score for {game_id}: {brier_score:.5f}")

    # 2. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(df['actual_outcome'], df['predicted_prob'], n_bins=10)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(prob_pred, prob_true, marker='o', label=game_id)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Reliability Diagram: {game_id}')
    plt.legend()
    plt.grid(True)

    # 3. Predicted Prob over Time
    plt.subplot(1, 2, 2)
    # Focus on a few representative tickers if there are many
    tickers = df['ticker'].unique()[:5]
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].sort_values('seconds_remaining', ascending=False)
        plt.plot(ticker_df['seconds_remaining'], ticker_df['predicted_prob'], label=ticker)
    
    plt.gca().invert_xaxis()
    plt.xlabel('Seconds Remaining')
    plt.ylabel('Predicted Probability')
    plt.title('Prediction Confidence over Time')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.tight_layout()
    output_path = f'/Users/jonathanbierly/.gemini/antigravity/brain/a0602cb7-fac4-4bce-944b-83af03b8c532/calibration_{game_id}.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    analyze_calibration(game_id='0022500702') # MIL@BOS
    analyze_calibration(game_id='0022500704') # BKN@DET
