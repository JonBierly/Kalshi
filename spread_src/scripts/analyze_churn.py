import sqlite3
import pandas as pd
import numpy as np
import math

def calculate_kalshi_fee(price_cents, size):
    if size <= 0: return 0.0
    p = price_cents / 100.0
    # Kalshi maker fee formula
    fee_dollars = 0.0175 * size * p * (1.0 - p)
    return math.ceil(fee_dollars * 100) / 100.0

def analyze_churn_refined(strategy=None):
    db_path = 'data/nba_data.db'
    conn = sqlite3.connect(db_path)
    
    # 1. Load active trades for churn analysis
    where_clause = ""
    if strategy:
        where_clause = f"AND strategy_id = '{strategy}'"
    
    query = f"SELECT ticker, side, fill_price, size, created_at, strategy_id FROM trades WHERE (status = 'filled' OR status = 'closed') {where_clause} ORDER BY created_at"
    df = pd.read_sql_query(query, conn)
    
    # 2. Get Global Stats
    where_clause_global = ""
    if strategy:
        where_clause_global = f"AND strategy_id = '{strategy}'"
    query_all_closed = f"SELECT count(*), sum(realized_pnl) FROM trades WHERE status = 'closed' {where_clause_global}"
    closed_stats = pd.read_sql_query(query_all_closed, conn).iloc[0]
    
    conn.close()

    if df.empty:
        print(f"No filled trades found{f' for strategy {strategy}' if strategy else ''}.")
        return
        
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['strategy_id'] = df['strategy_id'].fillna('unknown')
    
    print("=" * 100)
    print(f"REFINED CHURN ANALYSIS{' - Strategy: ' + strategy if strategy else ' - ALL STRATEGIES'}")
    print("=" * 100)
    
    churn_count = 0
    net_churn_pnl = 0.0
    total_fees = 0.0
    strategy_counts = {} # {strat: count}
    
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('created_at')
        
        for i in range(len(group) - 1):
            t1 = group.iloc[i]
            t2 = group.iloc[i+1]
            
            time_diff = (t2['created_at'] - t1['created_at']).total_seconds()
            
            # Identify round trips within 10 minutes
            if time_diff < 600 and t1['side'] != t2['side']:
                churn_count += 1
                
                strat = t2['strategy_id']
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
                
                p1, p2 = t1['fill_price'], t2['fill_price']
                size = min(t1['size'], t2['size'])
                
                if t1['side'] == 'buy':
                    pnl = (p2 - p1) * size / 100.0
                else: # t1 was sell (short entry)
                    pnl = (p1 - p2) * size / 100.0
                
                fee1 = calculate_kalshi_fee(p1, size)
                fee2 = calculate_kalshi_fee(p2, size)
                
                total_fees += (fee1 + fee2)
                net_churn_pnl += pnl
                
                if churn_count <= 10:
                    print(f"Ticker: {ticker[-15:]:<15} | {t1['side'].upper()}->{t2['side'].upper()} in {time_diff/60:>4.1f}m | PnL: ${pnl:>+6.2f} | Fees: ${fee1+fee2:4.2f} | Strat: {strat}")

    print("-" * 100)
    print(f"Total Churn Instances (<10m round-trip): {churn_count}")
    if not strategy:
        for s, c in strategy_counts.items():
            print(f"  - {s:<15}: {c}")
    print(f"Net Trading P&L from Churn:            ${net_churn_pnl:>+8.2f}")
    print(f"Estimated Fees on Churn:               ${total_fees:>+8.2f}")
    print(f"TOTAL PROFIT FROM CHURN:               ${(net_churn_pnl - total_fees):>+8.2f}")
    
    print("-" * 100)
    print(f"GLOBAL PERFORMANCE ({'STRATEGY ' + strategy.upper() if strategy else 'ALL'} CLOSED TRADES)")
    print(f"Total Closed Trades:                   {int(closed_stats[0])}")
    print(f"Total Realized P&L (Database):         ${closed_stats[1]:>+8.2f}")
    print("-" * 100)
    print("ANALYSIS: Use the strategy column to pinpoint which bot is over-trading.")
    print("=" * 100)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, help='Filter by strategy_id (simple_ev, rebalancer)')
    args = parser.parse_args()
    
    analyze_churn_refined(strategy=args.strategy)
