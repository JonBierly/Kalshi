import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.database import DatabaseManager, PlayerAdvancedStats, Game
from data.acquisition import HistoricalDataClient
import time

def backfill_advanced_stats(seasons=["2025-26"], limit=None):
    """
    Iterates through games in DB and fetches advanced stats if missing.
    Processes multiple seasons.
    
    Args:
        seasons: List of season strings (e.g., ['2024-25', '2025-26'])
        limit: Optional limit on total number of games to process
    """
    db_manager = DatabaseManager()
    api_client = HistoricalDataClient()
    
    session = db_manager.get_session()
    
    print("Finding games missing advanced stats...")
    
    # Get IDs of games that have stats
    existing_ids = session.query(PlayerAdvancedStats.game_id).distinct().all()
    existing_ids = set([i[0] for i in existing_ids])
    
    # Get game IDs filtered by seasons
    all_ids = []
    for season in seasons:
        query = session.query(Game.game_id).filter(Game.season == season)
        season_games = query.all()
        season_ids = [g.game_id for g in season_games]
        print(f"Found {len(season_ids)} games for season {season}")
        all_ids.extend(season_ids)
    
    missing_ids = [gid for gid in all_ids if gid not in existing_ids]
    
    print(f"Found {len(missing_ids)} games missing stats (out of {len(all_ids)} total across {len(seasons)} seasons).")
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill player advanced stats for specified seasons.')
    parser.add_argument('--seasons', nargs='+', default=['2024-25', '2025-26'],
                        help='List of seasons to backfill (e.g., --seasons 2024-25 2025-26)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit total number of games to process (for testing)')
    
    args = parser.parse_args()
    
    print(f"Running advanced stats backfill for seasons: {args.seasons}")
    backfill_advanced_stats(seasons=args.seasons, limit=args.limit)
