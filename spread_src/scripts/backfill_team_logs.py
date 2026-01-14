import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.database import DatabaseManager, TeamBasicStats, Game
from data.acquisition import HistoricalDataClient
import time
from nba_api.stats.static import teams

def backfill_team_logs(seasons=['2025-26']):
    """
    Iterates through all 30 NBA teams and fetches their game logs for multiple seasons.
    Saves stats to TeamBasicStats and updates Game metadata (Home/Away IDs).
    
    Args:
        seasons: List of season strings (e.g., ['2024-25', '2025-26'])
    """
    db_manager = DatabaseManager()
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
    
    args = parser.parse_args()
    
    print(f"Running team logs backfill for seasons: {args.seasons}")
    backfill_team_logs(seasons=args.seasons)
