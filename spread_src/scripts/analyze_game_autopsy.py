import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def game_autopsy(db_path='data/nba_data.db', game_slug='26FEB01MIL', nba_id='0022500702'):
    conn = sqlite3.connect(db_path)
    
    # 1. Prediction Trend: Model Confidence over time
    pred_query = f"""
    SELECT timestamp_local, ticker, predicted_prob, bid_price, ask_price, seconds_remaining
    FROM model_predictions
    WHERE game_id = '{nba_id}'
    """
    df_pred = pd.read_sql_query(pred_query, conn)
    
    # 2. Score Trend: Score Diff over time
    pbp_query = f"""
    SELECT period, remaining_time, score_diff, (period-1)*720 + (720-remaining_time) as elapsed_seconds
    FROM pbp_events
    WHERE game_id = '{nba_id}'
    ORDER BY elapsed_seconds ASC
    """
    df_score = pd.read_sql_query(pbp_query, conn)
    
    # 3. Trade History
    trade_query = f"""
    SELECT timestamp, ticker, side, fill_price, realized_pnl, seconds_remaining
    FROM trades
    WHERE game_id = '{game_slug}'
    ORDER BY timestamp ASC
    """
    df_trades = pd.read_sql_query(trade_query, conn)
    conn.close()

    if df_pred.empty or df_score.empty:
        print("Missing data for autopsy.")
        return

    # Visualization
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Score Diff (Primary Axis)
    ax1.plot(df_score['elapsed_seconds'], df_score['score_diff'], color='gray', alpha=0.3, label='Score Diff (Home-Away)')
    ax1.set_xlabel('Elapsed Seconds')
    ax1.set_ylabel('Score Differential', color='gray')

    # Plot Model Prob for the main tickers (Secondary Axis)
    ax2 = ax1.twinx()
    tickers = df_pred['ticker'].unique()[:3] # Select top 3 tickers for clarity
    for ticker in tickers:
        t_df = df_pred[df_pred['ticker'] == ticker].copy()
        t_df['elapsed'] = 2880 - t_df['seconds_remaining']
        t_df = t_df.sort_values('elapsed')
        ax2.plot(t_df['elapsed'], t_df['predicted_prob'], label=f'Model Prob: {ticker.split("-")[-1]}')

    # Mark Trades
    for _, trade in df_trades.iterrows():
        elapsed = 2880 - trade['seconds_remaining']
        marker = '^' if trade['side'] == 'buy' else 'v'
        color = 'green' if trade['side'] == 'buy' else 'red'
        ax2.scatter(elapsed, trade['fill_price']/100.0, marker=marker, color=color, s=100, zorder=5)

    ax2.set_ylabel('Predicted Prob / Price (0-1)')
    ax2.set_ylim(-0.1, 1.1)
    
    plt.title(f'Game Autopsy: {game_slug} ({nba_id})')
    fig.tight_layout()
    
    output_path = f'/Users/jonathanbierly/.gemini/antigravity/brain/a0602cb7-fac4-4bce-944b-83af03b8c532/autopsy_{game_slug}.png'
    plt.savefig(output_path)
    print(f"Saved autopsy plot to {output_path}")

if __name__ == "__main__":
    game_autopsy(game_slug='26FEB01MIL', nba_id='0022500702')
    game_autopsy(game_slug='26FEB01BKN', nba_id='0022500704')
