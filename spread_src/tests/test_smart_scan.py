import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.s3_client import S3DataClient
from datetime import datetime
from zoneinfo import ZoneInfo

s3 = S3DataClient()
today_et = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
print(f"Testing Smart Scan for {today_et}...")

game_ids = s3.get_todays_game_ids()
print(f"Found {len(game_ids)} games: {game_ids}")

for gid in game_ids:
    info = s3.get_game_info(gid)
    print(f"  Game {gid}: {info['away_tri']} @ {info['home_tri']} (Status: {info['status']}, Date: {info['game_date_utc']})")
