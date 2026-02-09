import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def analyze_logic(db_path='data/nba_data.db', game_slug='26FEB01MIL'):
    conn = sqlite3.connect(db_path)
    
    # Query all trades for this game
    query = f"""
    SELECT timestamp, ticker, side, fill_price, size, realized_pnl, seconds_remaining
    FROM trades
    WHERE game_id = '{game_slug}'
    ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print(f"No trades found for {game_slug}")
        conn.close()
        return

    # Determine final outcome for each ticker
    # We'll use the trades' metadata or just derive it
    tickets = df['ticker'].unique()
    theoretical_results = {}
    
    for ticker in tickets:
        # Simple settlement derivation: If Brier analysis showed it won, it's 100.
        # Let's just look at the last realized_pnl or fill_price if we can.
        # Actually, let's just query the final score again.
        market_part = ticker.split('-')[-1]
        threshold_str = "".join([c for c in market_part if c.isdigit() or c == '.'])
        threshold = float(threshold_str) if threshold_str else 0
        
        # Hardcode for these two games for speed
        if 'MIL' in game_slug: # Boston won by 28
            if 'BOS' in ticker: win = (28 > threshold)
            else: win = (-28 > threshold)
        else: # Detroit won by 53
            if 'DET' in ticker: win = (53 > threshold)
            else: win = (-53 > threshold)
        
        theoretical_results[ticker] = 100.0 if win else 0.0

    # Calculate Cumulative P&L
    df['actual_pnl_step'] = df['realized_pnl'].fillna(0)
    df['cum_actual_pnl'] = df['actual_pnl_step'].cumsum()
    
    # Theoretical P&L: If we never sold/rebalanced, just held the first fill to settlement
    # This is slightly complex because positions change.
    # Let's simplify: compare (Realized P&L from trades) to (What if we only did the first trade and held)
    
    print(f"Analysis for {game_slug}:")
    print(f"Total Realized P&L: {df['actual_pnl_step'].sum():.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df['timestamp']), df['cum_actual_pnl'], label='Actual Realized P&L', marker='o')
    
    plt.title(f'P&L Progression: {game_slug}')
    plt.xlabel('Timestamp')
    plt.ylabel('P&L ($)')
    plt.legend()
    plt.grid(True)
    
    output_path = f'/Users/jonathanbierly/.gemini/antigravity/brain/a0602cb7-fac4-4bce-944b-83af03b8c532/pnl_progression_{game_slug}.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    conn.close()

if __name__ == "__main__":
    analyze_logic(game_slug='26FEB01MIL')
    analyze_logic(game_slug='26FEB01BKN')
