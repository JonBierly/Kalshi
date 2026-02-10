import sqlite3
import pandas as pd
import os

def sample_db(db_path):
    print(f"Working directory: {os.getcwd()}")
    if not os.path.exists(db_path):
        print(f"Error: {db_path} does not exist")
        return

    try:
        conn = sqlite3.connect(db_path)
        
        print("\n--- Games Table Sample ---")
        games = pd.read_sql_query("SELECT * FROM games ORDER BY date DESC LIMIT 5", conn)
        print(games)
        
        print("\n--- Model Predictions Sample ---")
        try:
            preds = pd.read_sql_query("SELECT * FROM model_predictions LIMIT 5", conn)
            print(preds)
        except Exception as e:
            print(f"Error reading model_predictions: {e}")
            
        print("\n--- Games Count ---")
        count = pd.read_sql_query("SELECT count(*) as count FROM games", conn)
        print(count)
        
        conn.close()
    except Exception as e:
        print(f"General error: {e}")

if __name__ == "__main__":
    db_path = '/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/nba_data.db'
    sample_db(db_path)
