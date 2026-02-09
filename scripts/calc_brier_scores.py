import sqlite3
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Database configuration
DB_PATH = 'data/nba_data.db'

# Game Results for 2026-02-04
# Format: {game_prefix: home_margin} where home_margin = home_score - away_score
# Tickers typically follow KXNBASPREAD-26FEB04{HOME}{AWAY}-{TEAM}{LINE}
# We need to be careful with the prefix mapping.
# Based on my research:
# DENNYK: NYK 134, DEN 127 -> NYK +7
# MINTOR: TOR 126, MIN 128 -> MIN +2 (Wait, prefix is MINTOR, so Home=TOR, Away=MIN. TOR-MIN = -2)
# NOPMIL: MIL 141, NOP 137 -> MIL +4 (Home=MIL, Away=NOP. MIL-NOP = 4)
# BOSHOU: HOU 93, BOS 114 -> BOS +21 (Home=HOU, Away=BOS. HOU-BOS = -21)
# OKCSAS: SAS 116, OKC 106 -> SAS +10 (Home=SAS, Away=OKC. SAS-OKC = 10)
# MEMSAC: SAC 125, MEM 129 -> MEM +4 (Home=SAC, Away=MEM. SAC-MEM = -4)
# CLELAC: LAC 91, CLE 124 -> CLE +33 (Home=LAC, Away=CLE. LAC-CLE = -33)

GAME_RESULTS = {
    '26FEB04DENNYK': 7,    # NYK - DEN = 7
    '26FEB04MINTOR': -2,   # TOR - MIN = -2
    '26FEB04NOPMIL': 4,    # MIL - NOP = 4
    '26FEB04BOSHOU': -21,  # HOU - BOS = -21
    '26FEB04OKCSAS': 10,   # SAS - OKC = 10
    '26FEB04MEMSAC': -4,   # SAC - MEM = -4
    '26FEB04CLELAC': -33,  # LAC - CLE = -33
}

def get_outcome(ticker):
    """
    Determine if a 'yes' bet on the ticker won.
    Ticker format: KXNBASPREAD-26FEB04{HOME}{AWAY}-{TEAM}{LINE}
    """
    match = re.search(r'-(26FEB04[A-Z]{6})-([A-Z]+)(\d+)', ticker)
    if not match:
        return None
    
    game_prefix = match.group(1)
    bet_team = match.group(2)
    line = int(match.group(3))
    
    if game_prefix not in GAME_RESULTS:
        return None
    
    home_team = game_prefix[-6:-3]
    away_team = game_prefix[-9:-6] # Wait, prefix is 26FEB04 + AWAY + HOME?
    # Let's re-verify prefix logic from tickers found earlier:
    # KXNBASPREAD-26FEB03UTAIND-UTA2
    # 26FEB03 + UTA (Away) + IND (Home)
    
    home_margin = GAME_RESULTS[game_prefix]
    
    # Extract home/away from ticker teams
    # Ticker: KXNBASPREAD-26FEB04DENNYK-NYK7
    # Here AWAY=DEN, HOME=NYK
    # If bet_team == NYK (Home), they win if home_margin > line
    # If bet_team == DEN (Away), they win if -home_margin > line
    
    home_team_code = game_prefix[-3:]
    away_team_code = game_prefix[-6:-3]
    
    if bet_team == home_team_code:
        return 1.0 if home_margin > line else 0.0
    elif bet_team == away_team_code:
        return 1.0 if -home_margin > line else 0.0
    else:
        return None

def main():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT ticker, predicted_prob, bid_price, ask_price, seconds_remaining
    FROM model_predictions
    WHERE ticker LIKE '%26FEB04%'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No predictions found for February 4th.")
        return
    
    print(f"Found {len(df)} predictions.")
    
    # Calculate outcomes
    df['outcome'] = df['ticker'].apply(get_outcome)
    
    # Filter out any where outcome couldn't be determined
    df = df.dropna(subset=['outcome'])
    print(f"Processing {len(df)} predictions with known outcomes.")
    
    # Market probability calculation (Mid-price)
    # bid/ask are in cents (0-100)
    df['market_prob'] = (df['bid_price'] + df['ask_price']) / 200.0
    
    # Brier Score = (1/N) * sum((prob - outcome)^2)
    df['model_sq_error'] = (df['predicted_prob'] - df['outcome'])**2
    df['market_sq_error'] = (df['market_prob'] - df['outcome'])**2
    
    model_brier = df['model_sq_error'].mean()
    market_brier = df['market_sq_error'].mean()
    
    # Calibration
    avg_model_prob = df['predicted_prob'].mean()
    avg_market_prob = df['market_prob'].mean()
    avg_outcome = df['outcome'].mean()
    
    # Skill Score = 1 - (Brier_model / Brier_market)
    skill_score = 1 - (model_brier / market_brier)
    
    print("\n" + "="*40)
    print("ANALYSIS RESULTS: 2026-02-04")
    print("="*40)
    print(f"Total Predictions: {len(df)}")
    print(f"Actual Outcome Rate: {avg_outcome:.2%}")
    print("-" * 40)
    print(f"Model Brier Score:  {model_brier:.4f}")
    print(f"Market Brier Score: {market_brier:.4f}")
    print(f"Skill Score:        {skill_score:+.2%}")
    print("-" * 40)
    print(f"Model Avg Prob:     {avg_model_prob:.2%}")
    print(f"Market Avg Prob:    {avg_market_prob:.2%}")
    print(f"Calibration Bias:   {avg_model_prob - avg_outcome:+.2%}")
    print("-" * 40)
    
    # Break down by time
    print("\nBy Time Remaining:")
    df['time_bucket'] = pd.cut(df['seconds_remaining'], 
                              bins=[0, 300, 600, 1200, 1800, 2880],
                              labels=['<5m', '5-10m', '10-20m', '20-30m', '30m+'])
    
    time_stats = df.groupby('time_bucket').agg({
        'model_sq_error': 'mean',
        'market_sq_error': 'mean',
        'ticker': 'count'
    })
    time_stats.columns = ['Model Brier', 'Market Brier', 'Count']
    time_stats['Skill'] = 1 - (time_stats['Model Brier'] / time_stats['Market Brier'])
    print(time_stats)
    
    # Break down by game
    print("\nBy Game:")
    df['game'] = df['ticker'].apply(lambda x: re.search(r'-(26FEB04[A-Z]{6})-', x).group(1) if re.search(r'-(26FEB04[A-Z]{6})-', x) else 'Unknown')
    game_stats = df.groupby('game').agg({
        'model_sq_error': 'mean',
        'market_sq_error': 'mean',
        'ticker': 'count'
    })
    game_stats.columns = ['Model Brier', 'Market Brier', 'Count']
    game_stats['Skill'] = 1 - (game_stats['Model Brier'] / game_stats['Market Brier'])
    print(game_stats)

if __name__ == "__main__":
    main()
