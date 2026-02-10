import os

db_path = '/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/nba_data.db'
print(f"Checking {db_path}...")
print(f"Exists: {os.path.exists(db_path)}")
try:
    with open(db_path, 'rb') as f:
        chunk = f.read(100)
        print(f"Read success! First 100 bytes: {chunk[:10]}")
except Exception as e:
    print(f"Read error: {e}")

# Check permissions of parent folders
path = db_path
while path != '/':
    path = os.path.dirname(path)
    print(f"Perms for {path}: {oct(os.stat(path).st_mode)[-3:]}")
