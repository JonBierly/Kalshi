import pandas as pd
import sqlite3
import os

def analyze_worst_games(results_path, db_path):
    df = pd.read_csv(results_path)
    
    # Calculate avg brier per game
    avg_brier = df.groupby('game_id')['brier'].mean().sort_values(ascending=False)
    worst_game_ids = avg_brier.head(5).index.tolist()
    
    print("--- Detailed Analysis of Worst Games ---")
    conn = sqlite3.connect(db_path)
    
    for gid in worst_game_ids:
        print(f"\nGame ID: {gid} (Avg Brier: {avg_brier[gid]:.4f})")
        
        # Get game info
        game_info = pd.read_sql_query(f"SELECT * FROM games WHERE game_id = '{gid}'", conn)
        if not game_info.empty:
            print(f"  Matchup: {game_info.iloc[0]['home_team_id']} vs {game_info.iloc[0]['away_team_id']}")
            print(f"  Final Score: Home {game_info.iloc[0]['home_score']} - Away {game_info.iloc[0]['away_score']}")
            print(f"  Final Margin: {game_info.iloc[0]['home_score'] - game_info.iloc[0]['away_score']}")
        
        # Look at predictions over time for this game
        game_df = df[df['game_id'] == gid]
        # Group by time to see where it failed most
        time_brier = game_df.groupby('seconds_remaining')['brier'].mean().sort_index(ascending=False)
        print("  Brier over time:")
        # Print every few time points
        for t in list(time_brier.index)[::4]:
            print(f"    {int(t/60)}m left: {time_brier[t]:.4f}")
            
        # Check for "huge flips"
        # Find timepoints where model was confident (prob > 0.8 or < 0.2) but outcome was the opposite
        flops = game_df[((game_df['prob'] > 0.8) & (game_df['outcome'] == 0)) | 
                        ((game_df['prob'] < 0.2) & (game_df['outcome'] == 1))]
        if not flops.empty:
            print(f"  Confidence Flops detected: {len(flops)} points")
            # Sample a few flops
            for _, flop in flops.head(3).iterrows():
                print(f"    At {int(flop['seconds_remaining']/60)}m left: Prob={flop['prob']:.2f}, Threshold={flop['threshold']}, Outcome={flop['outcome']}")
        else:
            print("  No major confidence flops (model was uncertain but wrong).")
            
    conn.close()

if __name__ == "__main__":
    results_path = 'reports/model_evaluation_100.csv'
    db_path = '/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/nba_data.db'
    analyze_worst_games(results_path, db_path)
