import sys
import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting test...")
try:
    from spread_src.patches import apply_patch
    apply_patch()
    print("Patch applied.")
except Exception as e:
    print(f"Error applying patch: {e}")

try:
    from nba_api.live.nba.endpoints.scoreboard import ScoreBoard
    print("Initializing ScoreBoard...")
    board = ScoreBoard()
    print("Fetching games...")
    games = board.games.get_dict()
    print("Success! Got", len(games), "games.")
except Exception as e:
    print(f"Error fetching scoreboard: {e}")
