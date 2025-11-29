from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from src.trading.paper_trading import Position

@dataclass
class TradeSignal:
    action: str  # 'BUY', 'SELL', 'HOLD'
    contracts: int
    price: float
    reason: str
    edge: float
    target_contracts: int
    current_contracts: int
    expected_value: float = 0.0

class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.05,  # 5% max per game
        max_total_exposure_pct: float = 0.30,  # 30% max total
        min_edge: float = 0.03,  # 3% minimum edge
        max_ci_width: float = 0.25,  # 25% max confidence interval width
        min_contracts: int = 1,  # Minimum trade size
        rebalance_threshold: int = 2  # Min contracts diff to trigger rebalance
    ):
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.min_edge = min_edge
        self.max_ci_width = max_ci_width
        self.min_contracts = min_contracts
        self.rebalance_threshold = rebalance_threshold

class TradingStrategy:
    def __init__(self, fractional_kelly: float = 0.25, risk_manager: RiskManager = None):
        self.fractional_kelly = fractional_kelly
        self.risk_manager = risk_manager or RiskManager()

    def calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """
        Calculate Kelly fraction.
        f = (bp - q) / b
        where:
        b = net odds received on the wager (payout / stake - 1)
        p = probability of winning
        q = probability of losing (1 - p)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
            
        # For binary options paying $1:
        # If price is 60c, you bet 0.60 to win 0.40 profit
        # b = 0.40 / 0.60 = 0.666...
        
        if odds <= 0 or odds >= 1:
            return 0.0
            
        b = (1 - odds) / odds
        p = win_prob
        q = 1 - p
        
        f = (b * p - q) / b
        return max(0.0, f)

    def calculate_target_position(self, model_prob: float, market_price: float, bankroll: float) -> int:
        """
        Calculate ideal number of contracts to hold based on Kelly Criterion.
        """
        market_prob = market_price / 100.0
        edge = model_prob - market_prob
        
        # 1. Check Edge
        if edge < self.risk_manager.min_edge:
            return 0
            
        # 2. Calculate Kelly
        raw_kelly = self.calculate_kelly_fraction(model_prob, market_prob)
        adj_kelly = raw_kelly * self.fractional_kelly
        
        # 3. Apply Max Position Limit
        target_allocation_pct = min(adj_kelly, self.risk_manager.max_position_pct)
        
        # 4. Convert to Contracts
        target_capital = bankroll * target_allocation_pct
        contract_price = market_price / 100.0
        
        if contract_price <= 0:
            return 0
            
        target_contracts = int(target_capital / contract_price)
        return target_contracts

    def evaluate_market(
        self,
        game_id: str,
        model_result: Dict,
        market_price: float,
        bankroll: float,
        current_position: Optional[Position] = None
    ) -> TradeSignal:
        """
        Evaluate market and generate BUY/SELL/HOLD signal to reach target position.
        """
        model_prob = model_result['probability']
        market_prob = market_price / 100.0
        edge = model_prob - market_prob
        
        # Check Confidence Interval Width
        ci_width = model_result['ci_95_upper'] - model_result['ci_95_lower']
        if ci_width > self.risk_manager.max_ci_width:
            # Too uncertain -> Target 0
            target_contracts = 0
            reason = f"CI width {ci_width:.1%} > {self.risk_manager.max_ci_width:.1%}"
        else:
            # Calculate Target
            target_contracts = self.calculate_target_position(model_prob, market_price, bankroll)
            reason = f"Targeting {target_contracts} contracts (Edge: {edge:+.1%})"

        # Current State
        current_contracts = current_position.contracts if current_position else 0
        diff = target_contracts - current_contracts
        
        # Determine Action
        if diff == 0:
            return TradeSignal('HOLD', 0, market_price, "At target", edge, target_contracts, current_contracts)
            
        if diff > 0:
            # BUY Signal
            if diff < self.risk_manager.min_contracts:
                return TradeSignal('HOLD', 0, market_price, f"Buy {diff} below min {self.risk_manager.min_contracts}", edge, target_contracts, current_contracts)
            
            return TradeSignal(
                'BUY', 
                diff, 
                market_price, 
                f"Rebalancing: Buy {diff} to reach {target_contracts}", 
                edge, 
                target_contracts, 
                current_contracts,
                expected_value=diff * (model_prob - market_prob)
            )
            
        else:
            # SELL Signal (diff is negative)
            sell_amount = abs(diff)
            
            if sell_amount < self.risk_manager.rebalance_threshold and target_contracts > 0:
                # Avoid churn for small reductions unless closing completely
                return TradeSignal('HOLD', 0, market_price, f"Sell {sell_amount} below threshold", edge, target_contracts, current_contracts)
                
            action_reason = "Take Profit/Stop Loss" if target_contracts > 0 else "Closing Position"
            if edge < self.risk_manager.min_edge:
                action_reason = f"Edge {edge:+.1%} too low"
            
            return TradeSignal(
                'SELL', 
                sell_amount, 
                market_price, 
                f"{action_reason}: Sell {sell_amount} to reach {target_contracts}", 
                edge, 
                target_contracts, 
                current_contracts
            )
