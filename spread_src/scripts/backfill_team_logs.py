import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.database import DatabaseManager, TeamBasicStats, Game
from data.acquisition import HistoricalDataClient
from data.s3_client import S3DataClient
import time
from nba_api.stats.static import teams

def backfill_team_logs(seasons=['2025-26'], use_s3=False):
    """
    Iterates through all 30 NBA teams and fetches their game logs for multiple seasons.
    Saves stats to TeamBasicStats and updates Game metadata (Home/Away IDs).
    
    Args:
        seasons: List of season strings (e.g., ['2024-25', '2025-26'])
        use_s3: Use S3 data source instead of stats.nba.com
    """
    db_manager = DatabaseManager()

    # Determine data source
    if not use_s3:
        try:
            import requests
            r = requests.get('https://stats.nba.com/stats/scoreboardv3?GameDate=2026-01-01&LeagueID=00',
                             timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                raise ConnectionError("Blocked")
        except Exception:
            print("stats.nba.com unreachable, auto-falling back to S3")
            use_s3 = True

    if use_s3:
        # S3 path: efficient single-scan for all teams
        s3_client = S3DataClient()
        for season in seasons:
            print(f"\n{'='*50}")
            print(f"Processing season: {season} (via S3)")
            print(f"{'='*50}")

            # Get existing game IDs to skip
            session = db_manager.get_session()
            existing = session.query(TeamBasicStats.game_id).distinct().all()
            existing_ids = set(g[0] for g in existing)
            session.close()
            print(f"  {len(existing_ids)} games already in DB, scanning for new ones...")

            logs_df = s3_client.get_all_team_logs(season, existing_game_ids=existing_ids)
            if not logs_df.empty:
                db_manager.save_team_basic_stats(logs_df)
            else:
                print("  No new team logs found.")
        return

    # Original stats.nba.com path
    api_client = HistoricalDataClient()
    nba_teams = teams.get_teams()
    print(f"Found {len(nba_teams)} NBA teams.")
    
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Processing season: {season}")
        print(f"{'='*50}")
        
        for i, team in enumerate(nba_teams):
            team_id = team['id']
            team_name = team['full_name']
            print(f"[{i+1}/{len(nba_teams)}] Processing {team_name} ({team_id}) for {season}...")
            
            try:
                # Fetch logs for Regular Season AND Playoffs
                for season_type in ['Regular Season', 'Playoffs']:
                    logs_df = api_client.get_team_game_log(team_id, season=season, season_type=season_type)
                    
                    if not logs_df.empty:
                        # Save to DB (and update metadata)
                        db_manager.save_team_basic_stats(logs_df)
                    else:
                        print(f"No {season_type} logs found for {team_name}")
                    
                    time.sleep(0.5) # Be nice
                
            except Exception as e:
                print(f"Error processing {team_name}: {e}")
                time.sleep(5)
                continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill team game logs for specified seasons.')
    parser.add_argument('--seasons', nargs='+', default=['2025-26'],
                        help='List of seasons to backfill (e.g., --seasons 2024-25 2025-26)')
    parser.add_argument('--use-s3', action='store_true',
                        help='Use S3 data source instead of stats.nba.com (for VPS)')
    
    args = parser.parse_args()
    
    print(f"Running team logs backfill for seasons: {args.seasons}")
    backfill_team_logs(seasons=args.seasons, use_s3=args.use_s3)
