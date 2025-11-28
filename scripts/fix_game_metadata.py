from src.data.database import DatabaseManager
from sqlalchemy import text

def fix_metadata():
    db = DatabaseManager()
    session = db.get_session()
    
    print("Fixing Game Metadata (Home/Away IDs)...")
    
    # Update Home Team IDs
    query_home = text("""
    UPDATE games 
    SET home_team_id = (
        SELECT team_id 
        FROM team_basic_stats 
        WHERE team_basic_stats.game_id = games.game_id 
        AND team_basic_stats.side = 'Home'
    )
    WHERE home_team_id = 0 OR home_team_id IS NULL
    """)
    
    # Update Away Team IDs
    query_away = text("""
    UPDATE games 
    SET away_team_id = (
        SELECT team_id 
        FROM team_basic_stats 
        WHERE team_basic_stats.game_id = games.game_id 
        AND team_basic_stats.side = 'Away'
    )
    WHERE away_team_id = 0 OR away_team_id IS NULL
    """)
    
    try:
        result_home = session.execute(query_home)
        print(f"Updated Home IDs for {result_home.rowcount} games.")
        
        result_away = session.execute(query_away)
        print(f"Updated Away IDs for {result_away.rowcount} games.")
        
        session.commit()
        print("Commit successful.")
        
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    fix_metadata()
