from src.inference.orchestrator import LiveGameOrchestrator

if __name__ == "__main__":
    orch = LiveGameOrchestrator()
    games = orch.get_todays_games()
    
    if not games:
        print("No games found for today.")
    else:
        print(f"Found {len(games)} games.")
        for i, g in enumerate(games):
            print(f"{i+1}. {g['gameCode']} ({g['gameStatusText']})")
            
        # Select first game for demo
        # Prompt user for selection
        try:
            choice = input("\nSelect a game number to track (or 'q' to quit): ")
            if choice.lower() == 'q':
                exit()
            idx = int(choice) - 1
            if 0 <= idx < len(games):
                target = games[idx]
                orch.run_live_loop(target['gameId'], target['homeTeam']['teamId'], target['awayTeam']['teamId'])
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
