"""
Paper Trading Engine for simulating trades without real money.

Tracks virtual balance, open positions, and P&L for testing trading strategies.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Position:
    """Represents a single trading position."""
    game_id: str
    ticker: str
    contracts: int
    avg_entry_price: float  # Weighted average entry price in cents
    timestamp: str
    
    def add(self, new_contracts: int, price: float):
        """Add to position, updating average cost basis."""
        total_cost = (self.contracts * self.avg_entry_price) + (new_contracts * price)
        self.contracts += new_contracts
        self.avg_entry_price = total_cost / self.contracts
        
    def reduce(self, contracts_to_sell: int) -> float:
        """
        Reduce position size.
        Returns realized P&L for the portion sold (based on avg entry price).
        """
        if contracts_to_sell > self.contracts:
            raise ValueError(f"Cannot sell {contracts_to_sell} contracts, only have {self.contracts}")
        
        self.contracts -= contracts_to_sell
        return contracts_to_sell * self.avg_entry_price
    
    def current_value(self, market_price: float) -> float:
        """Calculate current position value."""
        return self.contracts * (market_price / 100)
    
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.contracts * (self.avg_entry_price / 100)
    
    def unrealized_pnl(self, market_price: float) -> float:
        """Unrealized profit/loss."""
        return self.current_value(market_price) - self.cost_basis()
    
    def settle(self, outcome: bool) -> float:
        """
        Settle position when game ends.
        
        Args:
            outcome: True if home team won (yes pays $1), False otherwise
        
        Returns:
            Realized P&L
        """
        if outcome:
            # Won: Get $1 per contract, paid entry_price
            pnl = self.contracts * (1 - self.avg_entry_price / 100)
        else:
            # Lost: Lose entry_price per contract
            pnl = -self.cost_basis()
        
        return pnl


@dataclass
class ClosedTrade:
    """Represents a closed trade (sell or settlement)."""
    game_id: str
    ticker: str
    action: str  # 'SELL' or 'SETTLE'
    contracts: int
    entry_price: float
    exit_price: float
    pnl: float
    timestamp: str
    reason: str = ""


class PaperTradingEngine:
    """Manages virtual trading portfolio."""
    
    def __init__(self, starting_balance: float = 10000.0, state_file: str = 'paper_trading_state.json'):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.state_file = state_file
        
        self.open_positions: Dict[str, Position] = {}  # game_id -> Position
        self.trade_history: List[ClosedTrade] = []
        
        # Try to load existing state
        self.load_state()
    
    def get_total_exposure(self) -> float:
        """Get total capital tied up in open positions."""
        return sum(pos.cost_basis() for pos in self.open_positions.values())
    
    def get_available_balance(self) -> float:
        """Get balance available for new trades."""
        return self.balance - self.get_total_exposure()
        
    def load_state(self):
        """Load state from JSON file if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.starting_balance)
                    
                    # Load positions
                    self.open_positions = {}
                    for pid, pdata in data.get('positions', {}).items():
                        self.open_positions[pid] = Position(**pdata)
                        
                    # Load history
                    self.trade_history = []
                    for hdata in data.get('history', []):
                        self.trade_history.append(ClosedTrade(**hdata))
                        
                print(f"Loaded existing state: ${self.balance:,.2f} balance, {len(self.open_positions)} open positions")
            except Exception as e:
                print(f"Error loading state: {e}")
                self.balance = self.starting_balance
        else:
            print(f"Starting new paper trading session with ${self.starting_balance:,.2f}")

    def save_state(self):
        """Save current state to JSON file."""
        data = {
            'balance': self.balance,
            'positions': {k: asdict(v) for k, v in self.open_positions.items()},
            'history': [asdict(h) for h in self.trade_history],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def execute_trade(self, game_id: str, ticker: str, action: str, contracts: int, price: float, reason: str = "") -> Tuple[bool, str]:
        """
        Execute a trade (BUY or SELL).
        
        Args:
            game_id: Game identifier
            ticker: Market ticker
            action: 'BUY' or 'SELL'
            contracts: Number of contracts
            price: Price in cents (1-99)
            reason: Reason for trade
            
        Returns:
            (success, message)
        """
        if contracts <= 0:
            return False, "Contracts must be positive"
            
        cost = contracts * (price / 100.0)
        
        if action == 'BUY':
            # Check funds
            if cost > self.balance:
                return False, f"Insufficient funds: Need ${cost:.2f}, have ${self.balance:.2f}"
            
            # Update balance
            self.balance -= cost
            
            # Update/Create position
            if game_id in self.open_positions:
                pos = self.open_positions[game_id]
                old_avg = pos.avg_entry_price
                pos.add(contracts, price)
                msg = f"Added {contracts} contracts to {ticker} @ {price}¢ (Avg: {old_avg:.1f}¢ -> {pos.avg_entry_price:.1f}¢)"
            else:
                self.open_positions[game_id] = Position(
                    game_id=game_id,
                    ticker=ticker,
                    contracts=contracts,
                    avg_entry_price=price,
                    timestamp=datetime.now().isoformat()
                )
                msg = f"Opened new position: {contracts} contracts of {ticker} @ {price}¢"
                
            return True, msg
            
        elif action == 'SELL':
            if game_id not in self.open_positions:
                return False, "No position to sell"
            
            pos = self.open_positions[game_id]
            if contracts > pos.contracts:
                return False, f"Cannot sell {contracts}, only have {pos.contracts}"
            
            # Calculate P&L
            cost_basis_of_sale = contracts * (pos.avg_entry_price / 100.0)
            proceeds = contracts * (price / 100.0)
            pnl = proceeds - cost_basis_of_sale
            
            # Update balance
            self.balance += proceeds
            
            # Record trade
            trade = ClosedTrade(
                game_id=game_id,
                ticker=ticker,
                action='SELL',
                contracts=contracts,
                entry_price=pos.avg_entry_price,
                exit_price=price,
                pnl=pnl,
                timestamp=datetime.now().isoformat(),
                reason=reason
            )
            self.trade_history.append(trade)
            
            # Reduce position
            pos.reduce(contracts)
            
            # Remove if empty
            if pos.contracts == 0:
                del self.open_positions[game_id]
                msg = f"Closed position: Sold all {contracts} {ticker} @ {price}¢ (P&L: ${pnl:+.2f})"
            else:
                msg = f"Reduced position: Sold {contracts} {ticker} @ {price}¢ (P&L: ${pnl:+.2f})"
                
            return True, msg
            
        else:
            return False, f"Invalid action: {action}"

    def close_position(self, game_id: str, outcome: bool) -> Tuple[bool, str, float]:
        """
        Settle a position based on game outcome.
        outcome: True if YES won (pays $1), False if NO won (pays $0)
        """
        if game_id not in self.open_positions:
            return False, "Position not found", 0.0
            
        pos = self.open_positions[game_id]
        
        # Calculate P&L
        payout = 1.00 if outcome else 0.00
        total_value = pos.contracts * payout
        cost_basis = pos.cost_basis()
        pnl = total_value - cost_basis
        
        # Update balance
        self.balance += total_value
        
        # Record history
        closed_pos = ClosedTrade(
            game_id=game_id,
            ticker=pos.ticker,
            action='SETTLE',
            contracts=pos.contracts,
            entry_price=pos.avg_entry_price,
            exit_price=payout * 100,
            pnl=pnl,
            timestamp=datetime.now().isoformat(),
            reason="Game Finished"
        )
        self.trade_history.append(closed_pos)
        
        # Remove position
        del self.open_positions[game_id]
        
        status = "WON" if outcome else "LOST"
        return True, f"Settled {pos.ticker}: {status} (P&L: ${pnl:+.2f})", pnl

    def get_portfolio_summary(self):
        """Get current portfolio stats."""
        total_exposure = sum(p.cost_basis() for p in self.open_positions.values())
        
        # Calculate realized P&L
        realized_pnl = sum(t.pnl for t in self.trade_history)
        
        # Calculate win rate
        wins = sum(1 for t in self.trade_history if t.pnl > 0)
        losses = sum(1 for t in self.trade_history if t.pnl <= 0)
        total_trades = len(self.trade_history)
        
        roi = (realized_pnl / self.starting_balance) * 100 if self.starting_balance > 0 else 0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = sum(t.pnl for t in self.trade_history if t.pnl > 0) / wins if wins > 0 else 0
        avg_loss = sum(t.pnl for t in self.trade_history if t.pnl <= 0) / losses if losses > 0 else 0
        
        return {
            'balance': self.balance,
            'total_exposure': total_exposure,
            'total_pnl': realized_pnl,
            'roi_percent': roi,
            'open_positions': len(self.open_positions),
            'closed_trades': len(self.trade_history),
            'wins': wins,
            'losses': losses,
            'win_rate_percent': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trade_history)
        }
    
