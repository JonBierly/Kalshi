
from data.acquisition import HistoricalDataClient
from data.database import DatabaseManager, Game
import time

def run_backfill(seasons=['2023-24'], limit=None):
    """
    Fetches games from API and saves to DB for specified seasons.
    """
    api_client = HistoricalDataClient()
    db_manager = DatabaseManager()
    
    for season in seasons:
        print(f"Starting backfill for season {season}...")
        
        # Get Game IDs
        try:
            game_ids = api_client.get_season_games(season=season)
            print(f"Found {len(game_ids)} games for {season}.")
        except Exception as e:
            print(f"Error fetching games for {season}: {e}")
            continue
        
        if limit:
            game_ids = game_ids[:limit]
            print(f"Limiting to {limit} games.")
            
        # Iterate and Save
        for i, game_id in enumerate(game_ids):
            print(f"[{i+1}/{len(game_ids)}] Processing game {game_id} ({season})...")
            try:
                # Check if game exists in DB first (optimization)
                # We need to expose a check method or just rely on save_game_data's check
                # But save_game_data checks AFTER fetching PBP in the current logic?
                # Wait, save_game_data receives the DF. So we fetch PBP first.
                # User wants to avoid sleep if duplicate.
                # So we should check DB *before* fetching PBP.
                
                # Let's check DB first
                session = db_manager.get_session()
                exists = session.query(Game).filter_by(game_id=game_id).first()
                session.close()
                
                if exists:
                    print(f"Game {game_id} already exists in DB. Skipping.")
                    continue

                # Fetch PBP
                pbp_df = api_client.get_game_pbp(game_id)
                
                # Save to DB
                db_manager.save_game_data(game_id, pbp_df, season=season)
                
            except Exception as e:
                print(f"Failed to process game {game_id}: {e}")
                # Continue to next game
                continue

if __name__ == "__main__":
    # Run for both seasons. 
    # WARNING: This will take a long time (hours) for full seasons.
    # For demonstration, we'll limit to 10 games per season.
    # User can remove limit=10 to run full backfill.
    run_backfill(limit=None)
