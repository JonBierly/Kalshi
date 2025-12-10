

from data.database import DatabaseManager, TeamBasicStats, Game
from data.acquisition import HistoricalDataClient
import time
from nba_api.stats.static import teams

def backfill_team_logs(season='2025-26'):
    """
    Iterates through all 30 NBA teams and fetches their game logs.
    Saves stats to TeamBasicStats and updates Game metadata (Home/Away IDs).
    """
    db_manager = DatabaseManager()
    api_client = HistoricalDataClient()
    
    nba_teams = teams.get_teams()
    print(f"Found {len(nba_teams)} NBA teams.")
    
    for i, team in enumerate(nba_teams):
        team_id = team['id']
        team_name = team['full_name']
        print(f"[{i+1}/{len(nba_teams)}] Processing {team_name} ({team_id})...")
        
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
    # You can change the season here
    backfill_team_logs(season='2023-24')
