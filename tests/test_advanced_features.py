from src.features.engineering import TeamStatsEngine, RosterEngine, add_advanced_features
import pandas as pd

def test_engines():
    print("Testing TeamStatsEngine...")
    team_engine = TeamStatsEngine()
    print(f"Team Features Shape: {team_engine.features_df.shape}")
    print("Sample Team Features:")
    print(team_engine.features_df.head())
    
    print("\nTesting RosterEngine...")
    roster_engine = RosterEngine()
    print(f"Player Features Shape: {roster_engine.player_features.shape}")
    print("Sample Player Features:")
    print(roster_engine.player_features.head())
    
    # Test get_features for a specific game
    # We need a valid game_id. Let's pick one from the loaded data.
    if not team_engine.features_df.empty:
        sample_game = team_engine.features_df.iloc[-1]
        game_id = sample_game['game_id']
        team_id = sample_game['team_id']
        # We need another team ID for the opponent
        opp_id = 0 # Placeholder
        
        print(f"\nFetching features for Game {game_id}...")
        team_feats = team_engine.get_features(game_id, team_id, opp_id)
        print("Team Feats:", team_feats.keys())
        
        roster_feats = roster_engine.get_features(game_id, team_id, opp_id)
        print("Roster Feats:", roster_feats.keys())
        
if __name__ == "__main__":
    test_engines()
