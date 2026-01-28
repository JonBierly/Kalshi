"""
Standalone script to backfill actual_outcome and prediction_error in model_predictions.
Uses the 'games' and 'team_basic_stats' tables to determine final scores and tricodes.
"""

import sqlite3
import json
import os
import sys

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.data.spread_markets import parse_spread_ticker

def backfill():
    db_path = 'data/nba_data.db'
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Build team_id to tricode mapping
    print("Building team mapping...")
    cursor.execute("SELECT team_id, matchup FROM team_basic_stats")
    team_mapping = {}
    for team_id, matchup in cursor.fetchall():
        # Matchup format: "ATL vs. CHA" or "ATL @ NOP"
        # We can extract the first tricode
        tricode = matchup.split(' ')[0]
        team_mapping[team_id] = tricode

    # 2. Fetch all unsettled predictions
    print("Fetching unsettled predictions...")
    cursor.execute("""
        SELECT prediction_id, game_id, ticker, predicted_prob 
        FROM model_predictions 
        WHERE actual_outcome IS NULL
    """)
    predictions = cursor.fetchall()
    print(f"Found {len(predictions)} predictions to settle.")

    # 3. Fetch all game scores
    cursor.execute("SELECT game_id, home_team_id, away_team_id, home_score, away_score FROM games")
    game_scores = {row[0]: row[1:] for row in cursor.fetchall()}

    updates = []
    skipped = 0
    
    for pred_id, game_id, ticker, pred_prob in predictions:
        if game_id not in game_scores:
            skipped += 1
            continue
            
        home_id, away_id, home_score, away_score = game_scores[game_id]
        
        try:
            parsed = parse_spread_ticker(ticker)
        except Exception as e:
            print(f"Error parsing ticker {ticker}: {e}")
            continue
            
        spread_team_tri = parsed['spread_team']
        spread_value = parsed['spread_value']
        
        # Get tricodes for home/away teams in this game
        home_tri = team_mapping.get(home_id)
        away_tri = team_mapping.get(away_id)
        
        if not home_tri or not away_tri:
            skipped += 1
            continue
            
        # Actual outcome logic:
        # If spread_team is HOME: winner if home_score - away_score > spread_value
        # If spread_team is AWAY: winner if away_score - home_score > spread_value
        
        actual_outcome = False
        if spread_team_tri == home_tri:
            actual_outcome = (home_score - away_score) > spread_value
        elif spread_team_tri == away_tri:
            actual_outcome = (away_score - home_score) > spread_value
        else:
            # Should not happen if data is consistent, but handle anyway
            print(f"Warning: Spread team {spread_team_tri} not in game {game_id} ({home_tri} vs {away_tri})")
            skipped += 1
            continue
            
        outcome_int = 1 if actual_outcome else 0
        error = abs(pred_prob - outcome_int)
        
        updates.append((outcome_int, error, pred_id))

    # 4. Perform batch update
    print(f"Updating {len(updates)} predictions (skipped {skipped})...")
    cursor.executemany("""
        UPDATE model_predictions 
        SET actual_outcome = ?, prediction_error = ?, settled_at = CURRENT_TIMESTAMP
        WHERE prediction_id = ?
    """, updates)

    conn.commit()
    print("Done!")
    conn.close()

if __name__ == "__main__":
    backfill()
