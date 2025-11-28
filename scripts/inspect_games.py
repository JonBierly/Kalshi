import pandas as pd
from src.data.database import DatabaseManager

db = DatabaseManager()
query = """
SELECT 
    COUNT(*) as total_games,
    SUM(CASE WHEN home_team_id != 0 AND away_team_id != 0 THEN 1 ELSE 0 END) as valid_games,
    SUM(CASE WHEN home_team_id = 0 OR away_team_id = 0 THEN 1 ELSE 0 END) as invalid_games
FROM games
"""
print("Game Counts:")
print(pd.read_sql(query, db.engine))

print("\nSample of Invalid Games:")
query_invalid = """
SELECT * FROM games WHERE home_team_id = 0 OR away_team_id = 0 LIMIT 10
"""
print(pd.read_sql(query_invalid, db.engine))
