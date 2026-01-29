import pandas as pd
import numpy as np
from spread_src.features.engineering import TeamStatsEngine, RosterEngine
from src.data.database import DatabaseManager
from datetime import datetime

def verify_live_stats():
    print("="*80)
    print("📊 DATABASE STATS VERIFICATION")
    print("="*80)
    
    # 1. Check database freshness
    db = DatabaseManager()
    latest_game = pd.read_sql("SELECT date FROM games ORDER BY date DESC LIMIT 1", db.engine)
    print(f"Latest game in DB: {latest_game['date'].iloc[0]}")
    
    # 2. Initialize Engines
    print("\nInitializing TeamStatsEngine...")
    team_engine = TeamStatsEngine()
    print("Initializing RosterEngine...")
    roster_engine = RosterEngine()
    
    # 3. Pick some relevant teams (e.g. today's teams or top teams)
    # Today is Jan 29. Let's look at NYK (Knicks) and TOR (Raptors) as they were in the previous blowout discussion.
    # We need their IDs. NYK=1610612752, TOR=1610612761 (usually)
    teams_to_check = {
        'NYK (Knicks)': 1610612752,
        'TOR (Raptors)': 1610612761,
        'LAL (Lakers)': 1610612747,
        'BOS (Celtics)': 1610612738
    }
    
    print("\n🔍 ENTERING STATS (What the model sees before tip-off):")
    print(f"{'Team':<15} {'Season Margin':>12} {'Recent Margin':>12} {'Recent SOS':>12} {'Consistency':>12}")
    print("-" * 75)
    
    for name, tid in teams_to_check.items():
        stats = team_engine.get_latest_features(tid)
        if stats:
            print(f"{name:<15} {stats['team_season_win_margin']:>12.2f} {stats['team_recent_win_margin']:>12.2f} {stats['team_recent_SOS']:>12.3f} {stats['team_recent_scoring_consistency']:>12.1f}")
        else:
            print(f"{name:<15} {'NO DATA':>12}")

    print("\n🛡️ ROSTER PIE VERIFICATION:")
    print(f"{'Team':<15} {'Base Roster PIE':>15}")
    print("-" * 35)
    for name, tid in teams_to_check.items():
        # Get projected roster (Top 10 players)
        roster_stats = roster_engine.get_projected_roster_features(tid)
        if roster_stats:
            print(f"{name:<15} {roster_stats['roster_recent_pie']:>15.3f}")
        else:
            print(f"{name:<15} {'NO DATA':>15}")

if __name__ == "__main__":
    verify_live_stats()
