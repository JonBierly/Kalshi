import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'data/nba_data.db'
WORST_GAMES = ['26JAN03MINMIA', '26JAN25MIAPHX', '26JAN02PORNOP', '25DEC26LACPOR', '26JAN19OKCCLE']
BEST_GAMES = ['25DEC15MEMLAC', '25DEC25SASOKC', '26JAN15CHALAL', '26JAN26PHICHA', '25DEC27BKNMIN']

def analyze_worst_games():
    conn = sqlite3.connect(DB_PATH)
    
    # Extract ticker, side, fill_price, model_fair_value, realized_pnl, etc.
    # We join with a subquery to identify parts of the ticker that match game IDs
    query = """
    SELECT 
        ticker, side, fill_price, model_fair_value, model_ci_lower, model_ci_upper,
        realized_pnl, seconds_remaining, market_spread
    FROM trades
    WHERE realized_pnl IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Filter out trades with an edge of < 2
    df = df[abs(df['model_fair_value'] - df['fill_price']) >= 2]

    # Filter for worst games
    # Tickers look like: KXNBASPREAD-26JAN03MINMIA-MIN5.5
    def matches_target(ticker):
        for g in WORST_GAMES:
            if g in ticker:
                return 'WORST', g
        for g in BEST_GAMES:
            if g in ticker:
                return 'BEST', g
        return None, None

    df['res'] = df['ticker'].apply(matches_target)
    df['type'] = df['res'].apply(lambda x: x[0])
    df['game_id'] = df['res'].apply(lambda x: x[1])
    target_df = df[df['game_id'].notnull()].copy()

    if target_df.empty:
        print("No trades found in database for these specific game IDs.")
        return

    for t_type in ['BEST', 'WORST']:
        print(f"\n{'='*20} {t_type} GAMES {'='*20}")
        games_list = BEST_GAMES if t_type == 'BEST' else WORST_GAMES
        type_df = target_df[target_df['type'] == t_type]
        
        for game in games_list:
            game_trades = type_df[type_df['game_id'] == game]
            if game_trades.empty:
                continue
            
        print(f"\n--- Game: {game} ---")
        total_pnl = game_trades['realized_pnl'].sum()
        avg_edge = abs(game_trades['fill_price'] - game_trades['model_fair_value']).mean()
        print(f"  PNL: ${total_pnl:.2f}")
        print(f"  Avg Entry Edge: {avg_edge:.1f}¢")
        
        # Check if we were consistently LONG or SHORT
        for side in ['buy', 'sell']:
            side_trades = game_trades[game_trades['side'] == side]
            if not side_trades.empty:
                side_pnl = side_trades['realized_pnl'].sum()
                print(f"    {side.upper()}: count={len(side_trades)}, pnl=${side_pnl:.2f}")

        # Show top 5 biggest losing trades in this game
        print("    Biggest Losing Trades:")
        losing = game_trades.sort_values('realized_pnl').head(3)
        for _, t in losing.iterrows():
            print(f"      {t['ticker']} ({t['side']}): fill={t['fill_price']}¢, fair={t['model_fair_value']:.1f}¢, pnl=${t['realized_pnl']:.2f}, secs_left={int(t['seconds_remaining'])}")

if __name__ == "__main__":
    analyze_worst_games()
