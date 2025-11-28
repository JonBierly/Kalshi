from src.data.database import DatabaseManager
from sqlalchemy import text

def fix_pbp_team_ids():
    db = DatabaseManager()
    session = db.get_session()
    
    print("Fixing PBP Team IDs...")
    
    # Update player_team_id using player_advanced_stats
    # We join on game_id and player_id
    query = text("""
    UPDATE pbp_events
    SET player_team_id = (
        SELECT team_id
        FROM player_advanced_stats
        WHERE player_advanced_stats.game_id = pbp_events.game_id
        AND player_advanced_stats.player_id = pbp_events.player_id
    )
    WHERE (player_team_id = 0 OR player_team_id IS NULL)
    AND player_id != 0
    """)
    
    try:
        result = session.execute(query)
        print(f"Updated {result.rowcount} PBP rows.")
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    fix_pbp_team_ids()
