from src.data.database import DatabaseManager
from sqlalchemy import text

db = DatabaseManager()

# Delete games where home_team_id or away_team_id is 0
delete_query = text("""
DELETE FROM games 
WHERE home_team_id = 0 OR away_team_id = 0
""")

# Also delete associated PBP events for those games to keep DB clean
# (Optional but good practice, though we don't have foreign keys enforced strictly)
# We'll just clean the games table first as requested.

with db.engine.connect() as conn:
    result = conn.execute(delete_query)
    conn.commit()
    print(f"Deleted {result.rowcount} invalid games.")
