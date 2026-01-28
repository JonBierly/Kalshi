import sqlite3
import pandas as pd
import numpy as np
import os

def analyze_historical_trades():
    db_path = 'data/nba_data.db'
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # We need to link trades to game outcomes.
    # Note: 'trades' has game_id and ticker.
    # We also need to determine the binary outcome (1 if YES wins, 0 if NO wins).
    # YES wins if score_diff > spread_threshold.
    
    # 1. Load games results with team tricodes/matchups to build a mapping
    # team_basic_stats has game_id, game_date, and matchup (e.g. 'LAL @ BOS')
    mapping_query = """
    SELECT DISTINCT game_id, matchup, game_date 
    FROM team_basic_stats
    """
    mapping_df = pd.read_sql_query(mapping_query, conn)
    
    # helper to extract tricode from matchup
    def get_tris(matchup):
        return [matchup[:3], matchup[-3:]]
    
    mapping_df['tris'] = mapping_df['matchup'].apply(get_tris)
    mapping_df['date_str'] = pd.to_datetime(mapping_df['game_date']).dt.strftime('%y%b%d').str.upper()
    
    # 2. Load trades
    query = """
    SELECT 
        trade_id, ticker, game_id as kalshi_game_id, side, fill_price, model_fair_value, 
        seconds_remaining, realized_pnl, status, created_at
    FROM trades 
    WHERE model_fair_value IS NOT NULL 
    AND (status = 'closed' OR status = 'filled')
    """
    trades_df = pd.read_sql_query(query, conn)
    
    # 3. Load games results for score diff
    games_query = "SELECT game_id, home_score, away_score FROM games WHERE home_score IS NOT NULL"
    games_df = pd.read_sql_query(games_query, conn)
    
    conn.close()

    if trades_df.empty:
        print("No historical trades with model predictions found.")
        return

    # 4. Map Kalshi game_id to NBA game_id
    # Kalshi game_id format: 26JAN20LAL
    def map_kalshi_to_nba(row):
        kalshi_id = row['kalshi_game_id']
        if not kalshi_id or len(kalshi_id) < 7: return None
        
        # Date part is usually first 7 chars: 26JAN20
        date_part = kalshi_id[:7].upper()
        team_part = kalshi_id[7:].upper()
        
        # Find matches in mapping_df
        matches = mapping_df[mapping_df['date_str'] == date_part]
        if matches.empty: return None
        
        # Filter by team tricode
        for _, m in matches.iterrows():
            if team_part in m['tris']:
                return m['game_id']
        return None

    trades_df['nba_game_id'] = trades_df.apply(map_kalshi_to_nba, axis=1)
    
    # Merge with games
    df = pd.merge(trades_df, games_df, left_on='nba_game_id', right_on='game_id', how='inner')
    
    if df.empty:
        print("Could not link any trades to completed games after mapping.")
        return

    # 5. Determine Actual Outcome
    # We need the threshold and which team it applies to.
    def parse_market_threshold(ticker):
        parts = ticker.split('-')
        if len(parts) < 3: return None, None
        thresh_str = parts[-1] 
        for i, char in enumerate(thresh_str):
            if char.isdigit():
                return float(thresh_str[i:]), thresh_str[:i]
        return None, None

    # Get team_id mapping (we need to know if the tricode in the ticker is home or away)
    # Re-fetch tricode mapping
    # ... Simplified: use the tricode in the ticker to identify the winning condition.
    
    pbp_map_query = "SELECT DISTINCT game_id, team_id, side FROM team_basic_stats"
    # We'll just assume for now if team_tri wins the spread...
    # Actually, the realized_pnl is the GOLD standard for closed trades.
    def calculate_actual_outcome(row):
        if row['status'] == 'closed':
            if row['side'] == 'buy':
                return 1.0 if row['realized_pnl'] > 0 else 0.0
            else: # sold/shorted
                return 0.0 if row['realized_pnl'] > 0 else 1.0
        return None

    df['actual_outcome'] = df.apply(calculate_actual_outcome, axis=1)
    df = df.dropna(subset=['actual_outcome'])

    if df.empty:
        print("Could not determine outcomes (closed trades only).")
        return

    # 6. Normalize Fair Value and Calculate Edge
    df['model_prob'] = df['model_fair_value']
    if df['model_prob'].max() > 1.1:
        df['model_prob'] /= 100.0

    df['exec_price'] = df['fill_price'] / 100.0
    
    # Calculate predicted edge on the trade (Model vs Execution Price)
    # If buy: edge = model - price
    # If sell: edge = price - model
    df['predicted_edge'] = np.where(
        df['side'] == 'buy',
        df['model_prob'] - df['exec_price'],
        df['exec_price'] - df['model_prob']
    )

    # Brier calculation
    def brier_score(prob, outcome):
        return np.mean((prob - outcome)**2)

    model_brier = brier_score(df['model_prob'], df['actual_outcome'])
    price_brier = brier_score(df['exec_price'], df['actual_outcome'])

    print("=" * 100)
    print("HISTORICAL TRADE PERFORMANCE (FROM 'TRADES' TABLE - MAPPED)")
    print("=" * 100)
    print(f"Total Mapped Trades: {len(df):,}")
    print(f"Unique Games: {df['nba_game_id'].nunique()}")
    print(f"Date Range: {df['created_at'].min()} to {df['created_at'].max()}")
    print("-" * 100)
    print(f"Model Brier (at execution): {model_brier:.5f}")
    print(f"Price Brier (fill price):   {price_brier:.5f}")
    
    cal_edge = (price_brier - model_brier) / price_brier
    cal_msg = "MORE calibrated" if cal_edge > 0 else "LESS calibrated"
    print(f"Calibration verdict:        Model is {abs(cal_edge):.1%} {cal_msg} than entry price.")

    # Directional Accuracy
    df['correct_pick'] = np.where(
        df['side'] == 'buy',
        df['actual_outcome'] == 1,
        df['actual_outcome'] == 0
    )
    
    acc = df['correct_pick'].mean()
    print(f"Historical Win Rate:        {acc:.1%}")
    print("-" * 100)
    
    # Stage breakdown
    bins = [179, 360, 720, 1080, 1440, 1800, 2160, 2520, 2881]
    labels = [
        'Q4 6-3m', 'Q4 12-6m', 
        'Q3 6-0m', 'Q3 12-6m', 
        'Q2 6-0m', 'Q2 12-6m', 
        'Q1 6-0m', 'Q1 12-6m'
    ]
    df['stage'] = pd.cut(df['seconds_remaining'], bins=bins, labels=labels)
    
    stage_stats = df.groupby('stage', observed=False).apply(
        lambda x: pd.Series({
            'Count': len(x),
            'WinRate': x['correct_pick'].mean(),
            'AvgProfit': x['realized_pnl'].mean(),
            'AvgEdge': x['predicted_edge'].mean(),
            'ModelBrier': brier_score(x['model_prob'], x['actual_outcome']),
            'PriceBrier': brier_score(x['exec_price'], x['actual_outcome'])
        })
    ).round(4).iloc[::-1]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\nPerformance by Game Stage:")
    print(stage_stats)
    print("-" * 100)
    print("Note: 'AvgEdge' is (ModelProb - EntryPrice).")
    print("If PriceBrier < ModelBrier, the market's entry price was a better predictor than the model fair value.")
    print("=" * 100)

if __name__ == "__main__":
    analyze_historical_trades()
