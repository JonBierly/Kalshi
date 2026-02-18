import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import traceback
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.features.engineering import TeamStatsEngine, RosterEngine, add_interaction_features

def get_team_map():
    """Builds a map of Team Abbr (e.g., 'DEN') -> Team ID."""
    print("Building Team Map from DB...")
    
    # Static map fallback in case DB is messy
    static_map = {
        'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
        'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
        'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
        'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
        'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
        'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
        'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
        'DET': 1610612765, 'CHA': 1610612766
    }
    
    return static_map

def analyze_model_spreads():
    # 1. Initialize Components
    print("Initializing Engines...")
    try:
        kalshi = KalshiClient("3048039d-2104-4e20-801b-c7eb07519142", "key.key")
        model = SpreadDistributionModel('models/nba_spread_ngboost_v3_final.pkl')
        team_engine = TeamStatsEngine()
        roster_engine = RosterEngine()
        team_map = get_team_map()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Fetch Active Events
    print("\nFetching Active Spread Events...")
    endpoint = "/events"
    params = {"series_ticker": "KXNBASPREAD", "status": "open", "limit": 100}
    path = "/trade-api/v2/events"
    headers = kalshi._get_headers("GET", path)
    
    import requests
    try:
        resp = requests.get(f"{kalshi.base_url}{endpoint}", headers=headers, params=params)
        resp.raise_for_status()
        events = resp.json().get('events', [])
    except Exception as e:
        print(f"Error fetching events: {e}")
        return

    print(f"Found {len(events)} events.")
    
    results = []
    
    print("\nAnalyzing Markets vs Model...")
    print(f"{'Game':<20} {'Mkt Width':<10} {'Model CI Width':<15} {'Model Pred':<20} {'Rec. Spread':<15}")
    print("-" * 80)
    
    for event in events:
        time.sleep(0.5) # Rate limit
        
        ticker = event['event_ticker']
        # Parse teams from ticker: KXNBASPREAD-26FEB19DENLAC
        parts = ticker.split('-')
        if len(parts) < 2: continue
        
        # Last part is DATE + AWAY + HOME. e.g. 26FEB19DENLAC
        if len(parts[1]) <= 7:
            continue
            
        rest = parts[1][7:] # DENLAC
        
        if len(rest) == 6:
            away = rest[:3]
            home = rest[3:]
        else:
            # Maybe 2 letter teams or 26FEB03...
            # Let's try to just match known teams
            # Find which 3-letter combos match
            found = []
            for t in team_map.keys():
                if t in rest:
                    found.append(t)
            if len(found) == 2:
                # Sort by position in string
                found.sort(key=lambda x: rest.find(x))
                away = found[0]
                home = found[1]
            else:
                # Fallback parsing - assume first 3 are away if possible
                if len(rest) >= 6:
                    away = rest[:3]
                    home = rest[3:6]
                else:
                    print(f"Skipping {ticker} - parse fail")
                    continue
                
        away_id = team_map.get(away)
        home_id = team_map.get(home)
        
        if not away_id or not home_id:
            # print(f"Skipping {ticker} - Could not map teams {away}/{home}")
            continue
            
        # Get Markets
        try:
            markets = kalshi.get_event_markets(ticker)
        except: continue
        if not markets: continue
        
        # Find widest/tightest market spread
        spreads = []
        for m in markets:
            bid = m.get('yes_bid', 0)
            ask = m.get('yes_ask', 100) # Default to 100 if empty
            if ask == 0: ask = 100
            
            width = ask - bid
            spreads.append(width)
            
        if not spreads: continue
        
        market_width_avg = np.mean(spreads)
        market_width_min = np.min(spreads)
        
        # 3. Generate Features for Prediction
        # Use latest stats for both teams
        try:
            h_stats = team_engine.get_latest_features(home_id)
            a_stats = team_engine.get_latest_features(away_id)
            
            h_roster = roster_engine.get_projected_roster_features(home_id)
            a_roster = roster_engine.get_projected_roster_features(away_id)
        except Exception as e:
            print(f"Feature error for {away}@{home}: {e}")
            continue
        
        # Combine
        features = {}
        for k, v in h_stats.items(): features[f'home_{k}'] = v
        for k, v in a_stats.items(): features[f'away_{k}'] = v
        for k, v in h_roster.items(): features[f'home_{k}'] = v
        for k, v in a_roster.items(): features[f'away_{k}'] = v
        
        # Add Context Features (Pre-Game)
        features['score_diff'] = 0
        features['seconds_remaining'] = 2880 # 48 mins
        features['period'] = 1
        features['lead_changes'] = 0
        features['score_volatility'] = 0
        # Derived
        features['required_catchup_rate'] = 0
        features['live_pace'] = 98.0 # League average start
        features['score_momentum'] = 0
        for w in ['momentum_2min', 'momentum_4min', 'momentum_6min', 'momentum_8min', 'momentum_10min']:
            features[w] = 0
            
        # EFG and others - zero for start
        for k in ['home_efg', 'away_efg', 'turnover_diff', 'home_3p_reliance', 'away_3p_reliance', 
                  'home_steal_rate', 'away_steal_rate', 'home_block_rate', 'away_block_rate']:
            features[k] = 0
        features['home_rebound_rate'] = 0.5
        
        features['game_id'] = 'SIM'
        features['home_team_id'] = home_id
        features['away_team_id'] = away_id
        features['home_rest_days'] = h_stats.get('rest_days', 3)
        features['away_rest_days'] = a_stats.get('rest_days', 3)
        
        # 4. Predict
        try:
            # ... prediction logic ...
            # ... prediction logic ...
            dist_params = model.predict_distribution_params(features)
            
            # Debug: check type
            # print(f"DEBUG: dist_params type: {type(dist_params)}")
            # print(f"DEBUG: dist_params keys: {dist_params.keys()}")
            # if 'mean' in dist_params:
            #    print(f"DEBUG: dist_params['mean'] type: {type(dist_params['mean'])}")
            
            # Calculate ensemble stats
            ensemble_means = dist_params['mean'][0]
            ensemble_stds = dist_params['std'][0]
            
            final_mean = np.mean(ensemble_means)
            final_std = np.mean(ensemble_stds) # Average aleatoric uncertainty
            
            # Epistemic Uncertainty (Variance of means)
            epistemic_std = np.std(ensemble_means)
            
            # Total Uncertainty (approximation)
            total_std = np.sqrt(final_std**2 + epistemic_std**2)
            
            # 90% CI Width implies range where 90% of outcomes fall
            ci_width_points = 2 * 1.645 * total_std
            
        except Exception as e:
            print(f"Prediction error for {away}@{home}: {e}")
            traceback.print_exc()
            continue
            
        # ...
            
        # 5. Recommendation
        # If Model CI Width is e.g., 30 points (huge), spread represents wide range of outcomes.
        # But we want "Spread in Cents" (Probability Width).
        # We need to map "Points Uncertainty" to "Probability Slope".
        # A simple heuristic:
        # If we are market making, we want to capture the "middle" of the distribution.
        # 
        # Let's convert Point Width to Spread Width.
        # This is tricky without simulating the specific line.
        # But generally:
        # If sigma is high -> Probability curve is flatter -> 50% to 90% takes more points -> wide spread in points?
        # No, "Spread Width" in Market is ASK_PRICE - BID_PRICE.
        # If I quote Bid 40c, Ask 60c (20c width).
        # 
        # Strategy:
        # We should quote a spread that covers our model's uncertainty about the probability.
        # Epistemic Uncertainty (std of probs from ensemble) is the best metric for "Model Confidence".
        #
        # Let's calculate the average Epistemic Uncertainty across a range of thresholds.
        
        thresholds = [final_mean - 5, final_mean, final_mean + 5]
        uncertainties = []
        for t in thresholds:
             res = model.predict_spread_probabilities(features, [t])
             # How to get epistemic uncertainty from this?
             # predict_spread_probabilities returns CI 90 Lower/Upper of probabilities
             # This computed from percentile(all_probs).
             # width = upper - lower
             width_prob = res['ci_90_upper'][0] - res['ci_90_lower'][0]
             uncertainties.append(width_prob)
             
        avg_model_uncertainty_prob = np.mean(uncertainties) * 100 # In cents
        
        # Recommended Spread = Model Uncertainty + Profit Margin (e.g. 5c)
        rec_spread = avg_model_uncertainty_prob + 5.0
        
        result_row = {
            'Game': f"{away} @ {home}",
            'Market Width (Avg)': f"{market_width_avg:.1f}c",
            'Model Conf Width': f"{avg_model_uncertainty_prob:.1f}c",
            'Model Pred': f"{final_mean:+.1f} +/- {total_std:.1f}",
            'Rec. Spread': f"{rec_spread:.1f}c",
            'Raw_Market': market_width_avg,
            'Raw_Rec': rec_spread
        }
        results.append(result_row)
        print(f"{result_row['Game']:<20} {result_row['Market Width (Avg)']:<10} {result_row['Model Conf Width']:<15} {result_row['Model Pred']:<20} {result_row['Rec. Spread']:<15}")

    # Volume Granularity Check
    print("\nVolume Granularity Check:")
    print("Checking if 'volume' field in market details has timestamp info...")
    # Just checking the last fetched market
    if events:
        try:
             m = markets[0]
             print(f"Sample Market Keys: {list(m.keys())}")
             if 'volume_history' in m:
                 print("FOUND volume_history!")
             else:
                 print("No 'volume_history' found in standard object.")
        except: pass

if __name__ == "__main__":
    analyze_model_spreads()
