import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from tqdm import tqdm
import re
import pytz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.database import DatabaseManager, Game, PBPEvent
from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.features.engineering import FeatureEngine, TeamStatsEngine, RosterEngine
from spread_src.data.spread_markets import parse_spread_ticker

TRICODE_TO_ID = {
    'MIA': 1610612748, 'PHI': 1610612755, 'SAS': 1610612759, 'PHX': 1610612756, 
    'BKN': 1610612751, 'TOR': 1610612761, 'LAL': 1610612747, 'UTA': 1610612762, 
    'POR': 1610612757, 'OKC': 1610612760, 'CHA': 1610612766, 'ATL': 1610612737, 
    'ORL': 1610612753, 'BOS': 1610612738, 'LAC': 1610612746, 'CLE': 1610612739, 
    'WAS': 1610612764, 'CHI': 1610612741, 'NOP': 1610612740, 'MEM': 1610612763, 
    'DAL': 1610612742, 'DET': 1610612765, 'MIL': 1610612749, 'NYK': 1610612752, 
    'SAC': 1610612758, 'DEN': 1610612743, 'GSW': 1610612744, 'IND': 1610612754, 
    'MIN': 1610612750, 'HOU': 1610612745
}
ID_TO_TRICODE = {v: k for k, v in TRICODE_TO_ID.items()}

def parse_anchor(desc, game_date):
    # Example: "Start of 1st Period (8:11 PM EST)"
    m = re.search(r"\((\d+):(\d+) (AM|PM) (EST|EDT)\)", desc)
    if not m: return None
    hour, minute, ampm, tz_str = m.groups()
    hour = int(hour)
    if ampm == "PM" and hour < 12: hour += 12
    elif ampm == "AM" and hour == 12: hour = 0
    
    dt = datetime(game_date.year, game_date.month, game_date.day, hour, int(minute))
    tz = pytz.timezone("US/Eastern")
    dt = tz.localize(dt).astimezone(pytz.UTC)
    
    # If the game starts late (past midnight UTC), and we are in EST, 
    # the date might have rolled over. But the NBA game date usually 
    # refers to the starting date.
    # We can check if dt is way before game_date context.
    return int(dt.timestamp())

def get_period_anchors(pbp_df, game_date):
    anchors = {} # (period, type) -> ts
    # type: 1 for start, 2 for end
    for _, row in pbp_df.iterrows():
        if row.get('actionType') == 'period':
            ts = parse_anchor(row.get('description', ''), game_date)
            if ts:
                # nba_api V3 says subType is 'start' or 'end'
                st = 1 if row.get('subType') == 'start' else 2
                anchors[(row['period'], st)] = ts
    return anchors

def compare_model_vs_market():
    db = DatabaseManager()
    model = SpreadDistributionModel('models/nba_spread_ngboost.pkl')
    team_engine, roster_engine = TeamStatsEngine(), RosterEngine()
    
    start_date, end_date = datetime(2025, 12, 29), datetime(2026, 1, 4)
    session = db.get_session()
    games = session.query(Game).filter(Game.date >= start_date, Game.date <= end_date).all()
    session.close()

    candlestick_dir = "data/candlesticks"
    available_tickers = [f.replace('.json', '') for f in os.listdir(candlestick_dir) if f.endswith('.json')]
    game_markets = {}
    for ticker in available_tickers:
        try:
            p = parse_spread_ticker(ticker)
            key = f"{p['date']}_{p['away_team']}_{p['home_team']}"
            if key not in game_markets: game_markets[key] = []
            game_markets[key].append({'ticker': ticker, 'spread': p['spread_value'], 'team': p['spread_team'], 'home_tri': p['home_team']})
        except: continue

    comparison_rows = []

    for game_obj in tqdm(games, desc="Analyzing Games"):
        home_tri, away_tri = ID_TO_TRICODE.get(game_obj.home_team_id), ID_TO_TRICODE.get(game_obj.away_team_id)
        if not home_tri or not away_tri: continue
        
        game_key = f"{game_obj.date.strftime('%y%b%d').upper()}_{away_tri}_{home_tri}"
        markets = game_markets.get(game_key, [])
        if not markets:
            game_key = f"{(game_obj.date - timedelta(days=1)).strftime('%y%b%d').upper()}_{away_tri}_{home_tri}"
            markets = game_markets.get(game_key, [])
        if not markets: continue

        # Load Enriched PBP
        gid_str = str(game_obj.game_id).zfill(10)
        pbp_path = f"data/enriched_pbp/{gid_str}.csv"
        if not os.path.exists(pbp_path): continue
        
        pbp_raw = pd.read_csv(pbp_path)
        anchors = get_period_anchors(pbp_raw, game_obj.date)
        if not anchors: continue

        # Filter and prepare PBP for enrichment
        session = db.get_session()
        db_events = session.query(PBPEvent).filter_by(game_id=game_obj.game_id).all()
        session.close()
        if not db_events: continue
        
        pbp_df = pd.DataFrame([e.__dict__ for e in db_events]).sort_values('remaining_time', ascending=False)
        pbp_df['game_id'], pbp_df['game_date'] = game_obj.game_id, game_obj.date
        pbp_df['home_team_id'], pbp_df['away_team_id'] = game_obj.home_team_id, game_obj.away_team_id
        pbp_enriched = pbp_df # We'll add features in the loop

        market_data = {}
        for m in markets:
            with open(f"data/candlesticks/{m['ticker']}.json", "r") as f:
                market_data[m['ticker']] = json.load(f)

        engine = FeatureEngine()
        for i, (_, event) in enumerate(pbp_enriched.iterrows()):
            feat_dict = engine.update(event)
            if i % 20 != 0: continue
            
            # ANCHOR-BASED INTERPOLATION
            p = event['period']
            rem = event['remaining_time'] % 720
            p_start, p_end = anchors.get((p, 1)), anchors.get((p, 2))
            if p_start and p_end:
                weight = (720 - rem) / 720.0
                event_ts = p_start + weight * (p_end - p_start)
            else: continue # Skip if no anchors for this period

            home_thresholds = [m['spread'] if m['team'] == m['home_tri'] else -m['spread'] for m in markets]
            preds = model.predict_spread_probabilities(feat_dict, home_thresholds)
            
            final_diff = game_obj.home_score - game_obj.away_score

            for idx, m in enumerate(markets):
                candles = market_data.get(m['ticker'], [])
                best_c = None
                for c in candles:
                    if c['end_period_ts'] - 60 <= event_ts <= c['end_period_ts']:
                        best_c = c; break
                if not best_c: continue
                
                try:
                    bid = best_c['yes_bid']['close'] / 100.0 if best_c['yes_bid']['close'] is not None else None
                    ask = best_c['yes_ask']['close'] / 100.0 if best_c['yes_ask']['close'] is not None else None
                    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else best_c['price']['close'] / 100.0
                except: continue
                
                m_prob = mid
                if m_prob <= 0 or m_prob >= 1: continue
                
                model_prob = preds['probabilities'][idx]
                if m['team'] != m['home_tri']: model_prob = 1.0 - model_prob
                
                outcome = 1.0 if (final_diff > m['spread'] if m['team'] == m['home_tri'] else -final_diff > m['spread']) else 0.0
                
                # EDGE CALCULATION
                edge = 0
                trade_type = None
                exec_prob = None
                
                if bid is not None and ask is not None:
                    # Passive strategy: Bid + 1, Ask - 1
                    passive_bid = bid + 0.01
                    passive_ask = ask - 0.01
                    
                    if model_prob > ask: # Bullish: Want to buy
                        edge = model_prob - passive_bid
                        trade_type = 'buy'
                        exec_prob = passive_bid
                    elif model_prob < bid: # Bearish: Want to sell
                        edge = passive_ask - model_prob
                        trade_type = 'sell'
                        exec_prob = passive_ask
                
                cum_rem = event['remaining_time'] + ((4 - event['period']) * 720 if event['period'] <= 4 else 0)
                
                comparison_rows.append({
                    'game_id': game_obj.game_id, 'ticker': m['ticker'],
                    'seconds_remaining': cum_rem,
                    'model_prob': model_prob, 'market_prob': mid,
                    'bid': bid, 'ask': ask,
                    'captured_edge': edge if edge > 0 else 0,
                    'is_trade': 1 if edge > 0 else 0,
                    'outcome': outcome
                })

    res_df = pd.DataFrame(comparison_rows)
    if not res_df.empty:
        res_df['model_brier'] = (res_df['model_prob'] - res_df['outcome'])**2
        res_df['market_brier'] = (res_df['market_prob'] - res_df['outcome'])**2
        print(f"\nObs: {len(res_df)} | Model: {res_df['model_brier'].mean():.4f} | Market: {res_df['market_brier'].mean():.4f}")
        res_df.to_csv("data/model_vs_market_comparison.csv", index=False)
    else: print("No data.")

if __name__ == "__main__":
    compare_model_vs_market()
