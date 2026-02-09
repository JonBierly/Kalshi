
import pandas as pd
import numpy as np
from spread_src.features.engineering import RosterEngine

def test_roster_sensitivity():
    print("Initializing RosterEngine...")
    engine = RosterEngine()
    
    # Milwaukee Bucks Team ID
    mil_id = 1610612749
    giannis_id = 203507
    
    print("\n--- TEST: Milwaukee Bucks (MIL) ---")
    
    # 1. Get stats for 'Expected' roster (Top 10 players by recent minutes)
    stats_full = engine.get_projected_roster_features(mil_id)
    
    # 2. Identify the active players from the stats_full logic (Top 10)
    entire_team = engine.latest_player_stats[engine.latest_player_stats['team_id'] == mil_id]
    top10 = entire_team.sort_values('player_recent_min_float', ascending=False).head(10)
    top10_ids = top10['player_id'].tolist()
    
    print("\nTop 5 Players by Minutes:")
    print(top10[['player_id', 'player_recent_min_float', 'player_recent_pie']].head(5))
    
    print(f"\nTop 10 players include Giannis ({giannis_id}): {giannis_id in top10_ids}")
    
    # 3. Simulate roster WITHOUT Giannis
    no_giannis_ids = [pid for pid in top10_ids if pid != giannis_id]
    stats_no_giannis = engine.get_projected_roster_features(mil_id, player_ids=no_giannis_ids)
    
    # Comparison
    print("\nComparison: Full Roster vs. No Giannis")
    print(f"{'Metric':<30} | {'Full':>10} | {'No Giannis':>10} | {'Delta':>10}")
    print("-" * 70)
    
    metrics = [
        'roster_recent_pie', 
        'roster_recent_top3_pie', 
        'roster_missing_starter_count'
    ]
    
    for m in metrics:
        v_full = stats_full.get(m, 0)
        v_no = stats_no_giannis.get(m, 0)
        delta = v_no - v_full
        print(f"{m:<30} | {v_full:>10.4f} | {v_no:>10.4f} | {delta:>+10.4f}")

    # Check for net diff feature too
    print(f"\nNet Star Difference Calculation (Mocking Away=None):")
    # Note: net_star_pie_diff is added in add_interaction_features, so we'd need to mock a dict
    fake_full = {'home_roster_recent_top3_pie': stats_full.get('roster_recent_top3_pie', 0), 'away_roster_recent_top3_pie': 0.10}
    fake_no = {'home_roster_recent_top3_pie': stats_no_giannis.get('roster_recent_top3_pie', 0), 'away_roster_recent_top3_pie': 0.10}
    
    from spread_src.features.engineering import add_interaction_features
    res_full = add_interaction_features(fake_full)
    res_no = add_interaction_features(fake_no)
    
    # --- TEST 2: Brooklyn Nets (BKN) ---
    bkn_id = 1610612751
    mpj_id = 1629008 # Michael Porter Jr (ID in this DB)
    
    print("\n--- TEST: Brooklyn Nets (BKN) ---")
    stats_full = engine.get_projected_roster_features(bkn_id)
    entire_team = engine.latest_player_stats[engine.latest_player_stats['team_id'] == bkn_id]
    top10 = entire_team.sort_values('player_recent_min_float', ascending=False).head(10)
    top10_ids = top10['player_id'].tolist()
    
    print(f"Top 10 players include MPJ ({mpj_id}): {mpj_id in top10_ids}")
    
    # Force inclusion of MPJ in baseline if he's not in Top 10 by minutes (might be coming off injury)
    if mpj_id not in top10_ids:
        print(f"Manually adding MPJ to baseline for sensitivity test...")
        top10_ids = top10_ids[:9] + [mpj_id]
        stats_full = engine.get_projected_roster_features(bkn_id, player_ids=top10_ids)
    
    no_mpj_ids = [pid for pid in top10_ids if pid != mpj_id]
    stats_no_mpj = engine.get_projected_roster_features(bkn_id, player_ids=no_mpj_ids)
    
    # Comparison
    print("\nComparison: Full Roster vs. No MPJ")
    print(f"{'Metric':<30} | {'Full':>10} | {'No MPJ':>10} | {'Delta':>10}")
    print("-" * 70)
    
    for m in metrics:
        v_full = stats_full.get(m, 0)
        v_no = stats_no_mpj.get(m, 0)
        delta = v_no - v_full
        print(f"{m:<30} | {v_full:>10.4f} | {v_no:>10.4f} | {delta:>+10.4f}")

if __name__ == "__main__":
    test_roster_sensitivity()
