from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from src.trading.paper_trading import Position

@dataclass
class TradeSignal:
    action: str  # 'BUY', 'SELL', 'HOLD'
    contracts: int
    price: float
    side: str  # 'YES' or 'NO'
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

    def calculate_target_position(self, model_prob: float, market_price_yes: float, market_price_no: float, bankroll: float) -> Tuple[int, str]:
        """
        Calculate ideal number of contracts and side (YES/NO) based on Kelly Criterion.
        """
        # Calculate Edge for YES
        market_prob_yes = market_price_yes / 100.0
        edge_yes = model_prob - market_prob_yes
        
        # Calculate Edge for NO
        # Model prob for NO is 1 - model_prob
        model_prob_no = 1.0 - model_prob
        market_prob_no = market_price_no / 100.0
        edge_no = model_prob_no - market_prob_no
        
        target_side = 'YES'
        target_contracts = 0
        
        # Determine which side (if any) to bet on
        if edge_yes > self.risk_manager.min_edge:
            # Bet YES
            target_side = 'YES'
            raw_kelly = self.calculate_kelly_fraction(model_prob, market_prob_yes)
            contract_price = market_prob_yes
            
        elif edge_no > self.risk_manager.min_edge:
            # Bet NO
            target_side = 'NO'
            raw_kelly = self.calculate_kelly_fraction(model_prob_no, market_prob_no)
            contract_price = market_prob_no
            
        else:
            # No significant edge
            return 0, 'YES'

        # Apply Risk Management
        adj_kelly = raw_kelly * self.fractional_kelly
        target_allocation_pct = min(adj_kelly, self.risk_manager.max_position_pct)
        
        target_capital = bankroll * target_allocation_pct
        
        if contract_price <= 0:
            return 0, target_side
            
        target_contracts = int(target_capital / contract_price)
        return target_contracts, target_side

    def evaluate_market(
        self,
        game_id: str,
        model_result: Dict,
        market_price_yes: float,
        market_price_no: float,
        bankroll: float,
        current_position: Optional[Position] = None
    ) -> TradeSignal:
        """
        Evaluate market and generate BUY/SELL/HOLD signal to reach target position.
        """
        model_prob = model_result['probability']
        seconds_remaining = model_result.get('seconds_remaining', 0)
        score_diff = model_result.get('score_diff', 0)
        
        # 1. HARD CUTOFF: Close all positions when < 2 minutes (120s) remain
        # Markets are too volatile and data latency is too high to trade safely
        if seconds_remaining < 120:
            current_contracts = current_position.contracts if current_position else 0
            current_side = current_position.side if current_position else 'YES'
            
            if current_contracts > 0:
                return TradeSignal(
                    'SELL',
                    current_contracts,
                    market_price_yes if current_side == 'YES' else market_price_no,
                    current_side,
                    f"Hard Cutoff: <2 mins remaining",
                    0.0,
                    0,
                    current_contracts
                )
            else:
                return TradeSignal(
                    'HOLD',
                    0,
                    market_price_yes,
                    'YES',
                    "Hard Cutoff: <2 mins, no new trades",
                    0.0,
                    0,
                    0
                )

        # 2. MERCY RULE: Adjust probability for impossible comebacks
        # If a team needs > 0.3 points per second to catch up, they are done.
        # This prevents the model from "buying the dip" on a guaranteed loss.
        catchup_rate = model_result.get('required_catchup_rate', 0)
        if catchup_rate > 0.3:
            if score_diff > 0: # Home is leading
                # Home effectively 100% to win
                model_prob = 0.999 
                # If we held NO, this ensures we sell. If we want YES, we might buy if market < 99.
            else: # Away is leading (Home score diff is negative)
                # Home effectively 0% to win
                model_prob = 0.001
                # If we held YES, this ensures we sell.
        
        # Check Confidence Interval Width
        ci_width = model_result['ci_95_upper'] - model_result['ci_95_lower']
        if ci_width > self.risk_manager.max_ci_width:
            target_contracts = 0
            target_side = 'YES' # Default
            reason = f"CI width {ci_width:.1%} > {self.risk_manager.max_ci_width:.1%}"
            edge = 0.0
        else:
            # Calculate Target
            target_contracts, target_side = self.calculate_target_position(
                model_prob, market_price_yes, market_price_no, bankroll
            )
            
            # Calculate edge for display
            if target_side == 'YES':
                edge = model_prob - (market_price_yes / 100.0)
            else:
                edge = (1.0 - model_prob) - (market_price_no / 100.0)
                
            reason = f"Targeting {target_contracts} {target_side} (Edge: {edge:+.1%})"

        # Current State
        current_contracts = current_position.contracts if current_position else 0
        current_side = current_position.side if current_position else 'YES'
        
        # Logic for Side Switching
        # If we hold YES but want NO (or vice versa), we must SELL ALL first.
        if current_contracts > 0 and current_side != target_side:
            # We are on the wrong side. Sell everything.
            # Note: We return a SELL signal for the CURRENT side.
            # The next iteration will see 0 contracts and then issue a BUY for the NEW side.
            return TradeSignal(
                'SELL',
                current_contracts,
                market_price_yes if current_side == 'YES' else market_price_no,
                current_side,
                f"Switching sides: Sell {current_side} to target {target_side}",
                edge,
                target_contracts,
                current_contracts
            )
            
        # If sides match (or we have no position), proceed with standard rebalancing
        diff = target_contracts - current_contracts
        market_price = market_price_yes if target_side == 'YES' else market_price_no
        
        if diff == 0:
            return TradeSignal('HOLD', 0, market_price, target_side, "At target", edge, target_contracts, current_contracts)
            
        if diff > 0:
            # BUY Signal
            if diff < self.risk_manager.min_contracts:
                return TradeSignal('HOLD', 0, market_price, target_side, f"Buy {diff} below min {self.risk_manager.min_contracts}", edge, target_contracts, current_contracts)
            
            return TradeSignal(
                'BUY', 
                diff, 
                market_price,
                target_side,
                f"Rebalancing: Buy {diff} {target_side} to reach {target_contracts}", 
                edge, 
                target_contracts, 
                current_contracts,
                expected_value=diff * edge # Approx EV
            )
            
        else:
            # SELL Signal (diff is negative)
            sell_amount = abs(diff)
            
            if sell_amount < self.risk_manager.rebalance_threshold and target_contracts > 0:
                return TradeSignal('HOLD', 0, market_price, target_side, f"Sell {sell_amount} below threshold", edge, target_contracts, current_contracts)
                
            action_reason = "Take Profit/Stop Loss" if target_contracts > 0 else "Closing Position"
            return TradeSignal(
                'SELL', 
                sell_amount, 
                market_price,
                target_side,
                f"{action_reason}: Sell {sell_amount} to reach {target_contracts}", 
                edge, 
                target_contracts, 
                current_contracts
            )
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
