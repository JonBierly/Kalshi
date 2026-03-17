import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.acquisition import HistoricalDataClient
from data.s3_client import S3DataClient
from data.database import DatabaseManager, Game
import time


def _get_client(use_s3: bool):
    """Return S3DataClient or HistoricalDataClient based on flag."""
    if use_s3:
        print("Using S3 data source (bypasses stats.nba.com block)")
        return S3DataClient()
    # Try stats.nba.com first, auto-fallback to S3 on failure
    try:
        import requests
        r = requests.get('https://stats.nba.com/stats/scoreboardv3?GameDate=2026-01-01&LeagueID=00',
                         timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            return HistoricalDataClient()
    except Exception:
        pass
    print("stats.nba.com unreachable, auto-falling back to S3")
    return S3DataClient()

def run_backfill(seasons=['2025-26'], limit=None, use_s3=False):
    """
    Fetches games from API and saves to DB for specified seasons.
    """
    api_client = _get_client(use_s3)
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill game play-by-play data for specified seasons.')
    parser.add_argument('--seasons', nargs='+', default=['2025-26'],
                        help='List of seasons to backfill (e.g., --seasons 2024-25 2025-26)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of games per season (for testing)')
    parser.add_argument('--use-s3', action='store_true',
                        help='Use S3 data source instead of stats.nba.com (for VPS)')
    
    args = parser.parse_args()
    
    print(f"Running backfill for seasons: {args.seasons}")
    run_backfill(seasons=args.seasons, limit=args.limit, use_s3=args.use_s3)
