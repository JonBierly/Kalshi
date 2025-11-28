import pandas as pd
from src.data.acquisition import HistoricalDataClient, ReplayDataClient, DataSchema

def test_historical_fetch():
    print("Testing HistoricalDataClient (MOCK)...")
    client = HistoricalDataClient()
    
    # Fetch games for a recent season (using a small window if possible, but get_season_games gets all)
    # We'll just check if it returns a non-empty list
    games = client.get_season_games(season='2023-24')
    print(f"Found {len(games)} games for 2023-24 season.")
    
    if not games:
        print("ERROR: No games found.")
        return

    # Pick one game to fetch PBP
    game_id = games[0]
    print(f"Fetching PBP for game {game_id}...")
    pbp_df = client.get_game_pbp(game_id)
    
    print("PBP DataFrame Shape:", pbp_df.shape)
    print("Columns:", pbp_df.columns.tolist())
    
    # Verify Schema
    missing_cols = [col for col in DataSchema.COLUMNS if col not in pbp_df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
    else:
        print("Schema verification PASSED.")
        
    # Show first few rows
    print(pbp_df.head())
    
    return game_id

def test_replay(game_id):
    print("\nTesting ReplayDataClient...")
    hist_client = HistoricalDataClient()
    replay_client = ReplayDataClient(hist_client)
    
    print(f"Streaming game {game_id}...")
    stream = replay_client.stream_game(game_id)
    
    # Fetch first 5 events
    for i, event_df in enumerate(stream):
        if i >= 5: break
        print(f"Event {i+1}: {event_df.iloc[0]['description']} (Time: {event_df.iloc[0]['remaining_time']})")

if __name__ == "__main__":
    game_id = test_historical_fetch()
    if game_id:
        test_replay(game_id)
