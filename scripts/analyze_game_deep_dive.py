import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
import re

def parse_edge_from_reason(reason):
    """Extract edge percentage from reason string if present."""
    # Look for patterns like "Edge +1.8%" or "Edge -55.0%"
    match = re.search(r'Edge\s+([+\-]?\d+\.?\d*)%', reason)
    if match:
        return float(match.group(1))
    return None

def analyze_game_deep_dive(state_file='paper_trading_state.json', target_ticker='KXNBAGAME-25NOV28ORLDET-DET'):
    if not os.path.exists(state_file):
        print(f"State file not found: {state_file}")
        return

    with open(state_file, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print("No trade history found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create output directory
    output_dir = 'reports/analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Global Profit Over Time
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['cumulative_pnl'], label='Total Portfolio P&L', color='green')
    plt.title('Total Portfolio Profit Over Time')
    plt.xlabel('Time')
    plt.ylabel('Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_profit.png')
    plt.close()
    print(f"Saved {output_dir}/total_profit.png")

    # 2. Profit by Game
    game_pnl = df.groupby('ticker')['pnl'].sum().sort_values()
    
    plt.figure(figsize=(12, 8))
    game_pnl.plot(kind='barh', color=game_pnl.apply(lambda x: 'green' if x >= 0 else 'red'))
    plt.title('Net Profit/Loss by Game')
    plt.xlabel('Profit ($)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/profit_by_game.png')
    plt.close()
    print(f"Saved {output_dir}/profit_by_game.png")

    # 3. Deep Dive into Target Game
    print(f"\nAnalyzing {target_ticker}...")
    game_df = df[df['ticker'] == target_ticker].copy()
    
    if game_df.empty:
        print(f"No trades found for {target_ticker}")
        return

    # Extract Edge and Price
    game_df['edge'] = game_df['reason'].apply(parse_edge_from_reason)
    game_df['market_price'] = game_df['exit_price'] # Price we sold at
    
    game_df['model_price'] = game_df.apply(
        lambda row: row['market_price'] + row['edge'] if pd.notnull(row['edge']) else None, 
        axis=1
    )

    # Plot 3: Contracts Traded Over Time (Sells)
    plt.figure(figsize=(12, 6))
    plt.bar(game_df['timestamp'], game_df['contracts'], width=0.001, color='red', label='Contracts Sold')
    plt.title(f'Selling Activity: {target_ticker}')
    plt.xlabel('Time')
    plt.ylabel('Contracts Sold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/selling_activity.png')
    plt.close()
    print(f"Saved {output_dir}/selling_activity.png")

    # Plot 4: Market vs Model Price at Trade Execution
    plt.figure(figsize=(12, 6))
    
    # Plot Market Price (Exit Price)
    plt.plot(game_df['timestamp'], game_df['market_price'], 'o-', label='Market Price (Sell)', color='black', alpha=0.7)
    
    # Plot Model Price (Inferred)
    mask = game_df['model_price'].notnull()
    if mask.any():
        plt.plot(game_df.loc[mask, 'timestamp'], game_df.loc[mask, 'model_price'], 'x', label='Model Price (Est.)', color='blue', markersize=10)
        
        # Draw lines connecting them to visualize the edge
        for idx, row in game_df[mask].iterrows():
            plt.plot([row['timestamp'], row['timestamp']], [row['market_price'], row['model_price']], 'k--', alpha=0.2)

    plt.title(f'Market vs Model Price at Execution: {target_ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price (Cents)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_vs_model.png')
    plt.close()
    print(f"Saved {output_dir}/market_vs_model.png")
    
    # Print summary stats for this game
    total_profit = game_df['pnl'].sum()
    avg_sell_price = (game_df['exit_price'] * game_df['contracts']).sum() / game_df['contracts'].sum()
    avg_entry_cost = (game_df['entry_price'] * game_df['contracts']).sum() / game_df['contracts'].sum()
    
    print(f"Summary for {target_ticker}:")
    print(f"Total Realized Profit: ${total_profit:,.2f}")
    print(f"Total Contracts Sold:  {game_df['contracts'].sum()}")
    print(f"Avg Entry Price:       {avg_entry_cost:.2f}¢")
    print(f"Avg Exit Price:        {avg_sell_price:.2f}¢")
    print(f"Avg Profit per Share:  {avg_sell_price - avg_entry_cost:.2f}¢")

if __name__ == "__main__":
    analyze_game_deep_dive()
