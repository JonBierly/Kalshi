import pandas as pd
import os
import time
from nba_api.stats.endpoints import playbyplayv3
from tqdm import tqdm

def re_fetch_pbp_with_wallclock(game_ids):
    output_dir = "data/enriched_pbp"
    os.makedirs(output_dir, exist_ok=True)
    
    for gid in tqdm(game_ids, desc="Re-fetching PBP"):
        # Ensure 10-digit format for NBA API
        gid_str = str(gid).zfill(10)
        output_path = f"{output_dir}/{gid_str}.csv"
        
        if os.path.exists(output_path):
            continue
            
        try:
            print(f"  Fetching {gid_str}...")
            pbp = playbyplayv3.PlayByPlayV3(game_id=gid_str, timeout=60)
            df = pbp.get_data_frames()[0]
            
            # The field is 'timeActual' in JSON, but nba_api might rename it to 'time_actual' or similar.
            # Let's inspect columns or just save it all.
            # Actually, per nba_api docs it is 'timeActual'
            df.to_csv(output_path, index=False)
            time.sleep(0.8) # Respect NBA API rate limits
        except Exception as e:
            print(f"Error fetching {gid_str}: {e}")

if __name__ == "__main__":
    # The 47 games identified from the previous comparison
    game_ids = [
        22500455, 22500454, 22500456, 22500447, 22500449, 22500450, 
        22500451, 22500452, 22500448, 22500453, 22500457, 22500475, 
        22500471, 22500472, 22500473, 22500474, 22500470, 22500466, 
        22500464, 22500463, 22500469, 22500467, 22500465, 22500468, 
        22500462, 22500461, 22500458, 22500460, 22500459, 22500476, 
        22500484, 22500478, 22500477, 22500480, 22500479, 22500482, 
        22500481, 22500485, 22500483, 22500486, 22500488, 22500489, 
        22500493, 22500490, 22500492, 22500487, 22500491
    ]
    re_fetch_pbp_with_wallclock(game_ids)
