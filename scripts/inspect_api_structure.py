from nba_api.live.nba.endpoints import scoreboard, boxscore
import json

def inspect():
    print("Fetching Scoreboard...")
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()
    
    if not games:
        print("No games found.")
        return
        
    print(f"Found {len(games)} games.")
    target_game = games[0]
    print(f"Game: {target_game['gameCode']}")
    print(f"Home Team: {target_game['homeTeam']['teamName']} (ID: {target_game['homeTeam']['teamId']})")
    print(f"Away Team: {target_game['awayTeam']['teamName']} (ID: {target_game['awayTeam']['teamId']})")
    
    game_id = target_game['gameId']
    print(f"\nFetching BoxScore for {game_id}...")
    box = boxscore.BoxScore(game_id=game_id)
    data = box.game.get_dict()
    
    h = data['homeTeam']
    a = data['awayTeam']
    
    print(f"BoxScore Home: {h['teamName']} - Score: {h['score']}")
    print(f"BoxScore Away: {a['teamName']} - Score: {a['score']}")
    
    score_diff = h['score'] - a['score']
    print(f"Calculated Score Diff (Home - Away): {score_diff}")

if __name__ == "__main__":
    inspect()
