from data.database import DatabaseManager, PlayerAdvancedStats, Game
from data.acquisition import HistoricalDataClient
import time

def backfill_advanced_stats(season="2025-26", limit=None):
    """
    Iterates through games in DB and fetches advanced stats if missing.
    If season is provided, only checks games from that season.
    """
    db_manager = DatabaseManager()
    api_client = HistoricalDataClient()
    
    session = db_manager.get_session()
    
    print("Finding games missing advanced stats...")
    
    # Get IDs of games that have stats
    existing_ids = session.query(PlayerAdvancedStats.game_id).distinct().all()
    existing_ids = set([i[0] for i in existing_ids])
    
    # Get game IDs (optionally filtered by season)
    query = session.query(Game.game_id)
    if season:
        query = query.filter(Game.season == season)
        print(f"Filtering for season {season}...")
        
    all_games = query.all()
    all_ids = [g.game_id for g in all_games]
    
    missing_ids = [gid for gid in all_ids if gid not in existing_ids]
    
    print(f"Found {len(missing_ids)} games missing stats (out of {len(all_ids)} total).")
    
    if limit:
        missing_ids = missing_ids[:limit]
        print(f"Limiting to {limit} games.")
    
    session.close()
    
    for i, game_id in enumerate(missing_ids):
        print(f"[{i+1}/{len(missing_ids)}] Processing {game_id}...")
        
        try:
            stats_df = api_client.get_advanced_boxscore(game_id)
            
            if not stats_df.empty:
                db_manager.save_advanced_stats(game_id, stats_df)
            else:
                print(f"No stats found for {game_id}")
                
            
        except Exception as e:
            print(f"Error processing {game_id}: {e}")
            time.sleep(5) # Backoff on error
            continue

if __name__ == "__main__":
    # Run for 2024-25 season
    backfill_advanced_stats(season='2023-24', limit=None)
