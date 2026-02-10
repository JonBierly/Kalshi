import sqlite3
import pandas as pd

def inspect_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in {db_path}:")
    for table in tables:
        t_name = table[0]
        cursor.execute(f"PRAGMA table_info({t_name});")
        columns = cursor.fetchall()
        print(f"\nTable: {t_name}")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
            
    conn.close()

if __name__ == "__main__":
    inspect_db('data/nba_data.db')
