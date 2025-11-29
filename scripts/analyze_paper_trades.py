"""
Analyze paper trading performance.

Shows metrics, win rate, ROI, and detailed trade history including active trading actions.
"""

import json
import os
from datetime import datetime


def analyze_paper_trades(state_file='paper_trading_state.json'):
    """Analyze paper trading performance from saved state."""
    
    if not os.path.exists(state_file):
        print(f"No state file found at {state_file}")
        print("Run paper trading first to generate history.")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    starting_balance = state.get('starting_balance', 10000)
    current_balance = state.get('balance', starting_balance)
    trade_history = state.get('history', [])
    open_positions = state.get('positions', {})  # Note: key is 'positions' in JSON, not 'open_positions'
    
    # Calculate current portfolio value (Balance + Exposure)
    total_exposure = 0
    for pos in open_positions.values():
        contracts = pos.get('contracts', 0)
        avg_price = pos.get('avg_entry_price', 0)
        total_exposure += contracts * (avg_price / 100)
        
    total_equity = current_balance + total_exposure
    total_pnl = total_equity - starting_balance
    roi = (total_pnl / starting_balance) * 100 if starting_balance > 0 else 0
    
    print("="*80)
    print("PAPER TRADING PERFORMANCE ANALYSIS")
    print("="*80)
    print()
    
    print(f"Starting Balance:  ${starting_balance:>10,.2f}")
    print(f"Current Balance:   ${current_balance:>10,.2f}")
    print(f"Open Exposure:     ${total_exposure:>10,.2f}")
    print(f"Total Equity:      ${total_equity:>10,.2f}")
    print(f"Total P&L:         ${total_pnl:>+10,.2f}")
    print(f"ROI:               {roi:>+10.2f}%")
    print()
    
    # Analyze Closed Trades (Realized P&L)
    # Filter for SELL and SETTLE actions which have P&L
    closed_trades = [t for t in trade_history if t.get('action') in ('SELL', 'SETTLE')]
    
    if closed_trades:
        wins = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        total_count = len(closed_trades)
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        
        realized_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        print("="*80)
        print("REALIZED PERFORMANCE")
        print("="*80)
        print(f"Realized P&L:      ${realized_pnl:>+10,.2f}")
        print(f"Total Trades:      {total_count}")
        print(f"Win Rate:          {win_rate:.1f}% ({win_count}W - {loss_count}L)")
        
        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            print(f"Avg Win:           ${avg_win:>10,.2f}")
            
        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            print(f"Avg Loss:          ${avg_loss:>10,.2f}")
            
        print()
        
        # Best Trades
        print("TOP 5 BEST TRADES")
        print("-" * 40)
        best_trades = sorted(closed_trades, key=lambda x: x.get('pnl', 0), reverse=True)[:5]
        for i, t in enumerate(best_trades, 1):
            ticker = t.get('ticker', '').replace('KXNBAGAME-', '')[:20]
            print(f"{i}. {ticker:<20} ${t.get('pnl', 0):>+8.2f} ({t.get('action')})")
        print()

    # Detailed Trade Log
    print("="*80)
    print("TRADE HISTORY LOG (Last 50 Actions)")
    print("="*80)
    print(f"{'TIME':<10} {'ACTION':<8} {'TICKER':<30} {'QTY':<5} {'PRICE':<8} {'P&L':<10} {'REASON'}")
    print("-" * 100)
    
    # Show last 50 actions
    recent_history = trade_history[-50:] if len(trade_history) > 50 else trade_history
    
    for trade in recent_history:
        # Parse timestamp
        try:
            ts = datetime.fromisoformat(trade['timestamp'])
            time_str = ts.strftime('%H:%M:%S')
        except:
            time_str = "Unknown"
            
        ticker = trade.get('ticker', '').replace('KXNBAGAME-', '')[:30]
        action = trade.get('action', 'UNKNOWN')
        qty = trade.get('contracts', 0)
        
        if action == 'BUY':
            price_str = f"{trade.get('entry_price', 0):.0f}¢"
            pnl_str = "-"
        elif action == 'SELL':
            price_str = f"{trade.get('exit_price', 0):.0f}¢"
            pnl_str = f"${trade.get('pnl', 0):+.2f}"
        elif action == 'SETTLE':
            price_str = f"{trade.get('exit_price', 0):.0f}¢"
            pnl_str = f"${trade.get('pnl', 0):+.2f}"
        else:
            price_str = "-"
            pnl_str = "-"
            
        reason = trade.get('reason', '')[:40]
        
        print(f"{time_str:<10} {action:<8} {ticker:<30} {qty:<5} {price_str:<8} {pnl_str:<10} {reason}")
    
    print("-" * 100)
    print()


if __name__ == "__main__":
    analyze_paper_trades()
