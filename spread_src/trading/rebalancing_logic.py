"""
Rebalancing Trader - Dynamic Portfolio Optimization logic.

Calculates optimal target positions based on Kelly math and 
generates rebalancing actions (Scale Up, Passive Harvest, Toxic Exit).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import scipy.stats as stats
import math

def calculate_kalshi_fee(price_cents: float, size: int) -> float:
    """
    Calculate Kalshi maker fee.
    fees = round up(0.0175 x C x P x (1-P))
    P = price in dollars, C = contracts
    Returns fee in dollars.
    """
    if size <= 0:
        return 0.0
    p = price_cents / 100.0
    fee_dollars = 0.0175 * size * p * (1.0 - p)
    return math.ceil(fee_dollars * 100) / 100.0



@dataclass
class RebalancingAction:
    """A rebalancing action to take in a market."""
    ticker: str
    action: str  # 'buy' or 'sell'
    size: int  # number of contracts
    price: int  # limit price in cents
    reason: str  # reason for the action
    edge: float  # edge percentage
    target_pos: int  # the target position after this action
    is_toxic_exit: bool = False
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    def __repr__(self):
        marker = "🔥 TOXIC EXIT" if self.is_toxic_exit else "⚖️ REBALANCE"
        return f"{marker} {self.ticker} {self.action.upper()} {self.size} @ {self.price}¢ ({self.reason}, edge: {self.edge:.1%})"


class RebalancingTrader:
    """
    Logic for dynamic portfolio rebalancing.
    
    1. Every 15s, calculate "Optimal Size" via Kelly Math.
    2. Scale Up: If Current < Optimal, place "Beat-by-1" Maker buy.
    3. Passive Harvesting: If Current > Optimal, place "Beat-by-1" Maker sell.
    4. Toxic Flow: If Fair < Cost and Edge < 0, Exit Immediately.
    5. Hysteresis: Only trade if delta >= 2 contracts.
    """
    
    def __init__(
        self, 
        bankroll: float = 15.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.04,
        max_ticker_exposure: float = 5.0,
        hysteresis_buffer: int = 2,
        min_harvest_edge: float = 0.01  # Don't harvest unless edge drops below 1%
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_ticker_exposure = max_ticker_exposure
        self.hysteresis_buffer = hysteresis_buffer
        self.min_harvest_edge = min_harvest_edge
    
    def evaluate_rebalancing(
        self,
        ticker: str,
        current_pos: int,
        cost_basis: float,
        model_prob: float,
        ci_lower: float,
        ci_upper: float,
        bid: Optional[int],
        ask: Optional[int],
        pending_pos: int = 0
    ) -> List[RebalancingAction]:
        """
        Evaluate a market and return rebalancing actions.
        """
        actions = []
        
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return actions

        fair_cents = model_prob * 100
        ci_lower_cents = ci_lower * 100
        ci_upper_cents = ci_upper * 100
        
        # 1. TOXIC FLOW CHECK
        # If Fair < Cost and Edge < 0 (using CI bounds), dump the position
        # For long position (current_pos > 0)
        if current_pos > 0:
            # Check edge to SELL (passive harvesting / exit)
            # Edge = (Ask-1) - Fair (in cents)
            sell_price = max(5, min(95, ask - 1))
            sell_edge = (sell_price - ci_upper_cents) / 100
            
            if fair_cents < (cost_basis - 1) and sell_edge < 0:
                # Toxic flow: model fair value dropped below cost and we have no edge to sell at ask-1
                # Actually, requirement says "Exit immediately to recover capital"
                # We'll use Maker exit (Ask-1) but flag it as toxic.
                return [RebalancingAction(
                    ticker=ticker,
                    action='sell',
                    size=abs(current_pos),
                    price=sell_price,
                    reason="Toxic Flow Exit",
                    edge=sell_edge,
                    target_pos=0,
                    is_toxic_exit=True
                )]
        
        # For short position (current_pos < 0)
        elif current_pos < 0:
            # Check edge to BUY (cover)
            # Edge = Fair - (Bid+1)
            buy_price = max(5, min(95, bid + 1))
            buy_edge = (ci_lower_cents - buy_price) / 100
            
            # Short position cost basis is effectively what we sold at.
            # If fair value is now ABOVE what we sold at, and we have no edge to buy back...
            if fair_cents > (cost_basis + 1) and buy_edge < 0:
                return [RebalancingAction(
                    ticker=ticker,
                    action='buy',
                    size=abs(current_pos),
                    price=buy_price,
                    reason="Toxic Flow Exit",
                    edge=buy_edge,
                    target_pos=0,
                    is_toxic_exit=True
                )]

        # 2. CALCULATE OPTIMAL TARGET POSITION
        best_target = 0
        best_edge = -1.0
        
        # Check BUYS (Scale Up)
        buy_price = max(5, min(95, bid + 1))
        buy_edge = (ci_lower_cents - buy_price) / 100
        if buy_edge >= self.min_edge:
            # Kelly size based on TOTAL bankroll
            size = self._calculate_kelly_size(buy_edge, buy_price)
            # Cap by ticker exposure
            max_contracts = int(self.max_ticker_exposure / (buy_price / 100.0))
            best_target = min(size, max_contracts)
            best_edge = buy_edge

        # Check SELLS (Passive Harvesting)
        sell_price = max(5, min(95, ask - 1))
        sell_edge = (sell_price - ci_upper_cents) / 100
        if sell_edge >= self.min_edge:
            size = self._calculate_kelly_size(sell_edge, 100 - sell_price)
            max_contracts = int(self.max_ticker_exposure / ((100 - sell_price) / 100.0))
            target = -min(size, max_contracts)
            if sell_edge > best_edge:
                best_target = target
                best_edge = sell_edge

        # 3. HOLD LOGIC (REDUCE CHURN)
        # If we have a position, and the model still sees an edge better than min_harvest_edge,
        # we don't dump to 0 just because it's below min_edge.
        total_pos = current_pos + pending_pos
        
        # Determine if we have a "Hold" reason
        if total_pos > 0 and buy_edge >= self.min_harvest_edge:
            # We are long and there's still some positive edge to hold
            # Keep the target at at least the current size to avoid chip-away selling
            if best_target >= 0:
                best_target = max(best_target, total_pos)
        elif total_pos < 0 and sell_edge >= self.min_harvest_edge:
            # We are short and there's still edge to hold
            if best_target <= 0:
                best_target = min(best_target, total_pos)

        # 4. COMPARE TO CURRENT AND APPLY HYSTERESIS
        # CRITICAL: We compare target to (current + pending) to avoid redundant orders
        diff = best_target - total_pos
        
        # We need to take action if delta >= buffer
        if abs(diff) >= self.hysteresis_buffer:
            action_type = 'buy' if diff > 0 else 'sell'
            action_size = abs(diff)
            price = max(5, min(95, (bid + 1) if action_type == 'buy' else (ask - 1)))
            final_edge = buy_edge if action_type == 'buy' else sell_edge
            
            # BUG FIX: Only rebalance if the action itself has a positive edge.
            # Otherwise we are just "chipping away" at positions by taking garbage prices.
            if final_edge >= self.min_harvest_edge:
                actions.append(RebalancingAction(
                    ticker=ticker,
                    action=action_type,
                    size=action_size,
                    price=price,
                    reason="Rebalancing",
                    edge=final_edge,
                    target_pos=best_target
                ))
            else:
                # Edge is negative or too small, don't rebalance yet
                # Unless we are very far from target? No, standard rebalancing should be +EV.
                pass

        return actions

    def _calculate_kelly_size(self, edge: float, price_or_margin: float) -> int:
        """
        Kelly size calculation.
        f = edge / (1 - price_in_dollars)
        """
        if edge <= 0: return 0
        p = price_or_margin / 100.0
        kelly_f = edge / (1.0 - p)
        
        # Apply fraction
        kelly_f *= self.kelly_fraction
        
        # Total bankroll dollars to bet
        kelly_dollars = self.bankroll * kelly_f
        size = int(round(kelly_dollars / p))
        
        return max(0, size)

    def _get_current_ticker_exposure(self, pos: int, cost: float) -> float:
        if pos == 0:
            return 0.0
        if pos > 0:
            return (cost / 100.0) * pos
        else:
            return ((100 - cost) / 100.0) * abs(pos)


class MultiAssetRebalancer:
    """
    Multi-asset portfolio optimization for a single game.
    Uses w = Σ⁻¹E to find optimal Kelly weights.
    """
    
    def __init__(
        self,
        bankroll: float = 15.0,
        kelly_fraction: float = 0.20,
        max_ticker_exposure: float = 5.0,
        scale_up_band: float = 0.10,  # 10% band for buying
        derisk_band: float = 0.20,    # 20% band for selling (asymmetric)
        min_trade_spread: int = 6     # Don't entry/scale-up if spread < 6c
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_ticker_exposure = max_ticker_exposure
        self.scale_up_band = scale_up_band
        self.derisk_band = derisk_band
        self.min_trade_spread = min_trade_spread

    def calculate_optimal_weights(
        self,
        tickers: List[str],
        probs: List[float],
        prices: List[float],
        distribution: stats.rv_continuous  # The Student-T distribution object
    ) -> Dict[str, float]:
        """
        Calculate target Kelly weights for a set of markets in one game.
        
        Args:
            tickers: List of tickers
            probs: Model probabilities P(hit)
            prices: Current market prices (normalized to 0-1)
            distribution: The underlying Student-T distribution used to generate probs
            
        Returns:
            Dict mapping ticker to target weight (0.0 to 1.0)
        """
        n = len(tickers)
        if n == 0:
            return {}
            
        # 1. Calculate Edge vector
        edges = np.array(probs) - np.array(prices)
        
        # 2. Build Covariance Matrix Σ
        # Cov(i, j) = P(i ∩ j) - P(i)P(j)
        # Since all i, j are derived from the same distribution, 
        # P(i ∩ j) is just P(final_score_diff > max(threshold_i, threshold_j))
        # Note: We need to handle team direction (is_home).
        
        sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    sigma[i, j] = probs[i] * (1 - probs[i])
                else:
                    # COVARIANCE MATH:
                    # Cov(i, j) = P(i ∩ j) - P(i)P(j)
                    
                    ticker_i, ticker_j = tickers[i], tickers[j]
                    
                    # Extract team from ticker (format: KXNBASPREAD-26JAN07HOUPOR-HOU14)
                    team_i = ticker_i.split('-')[-1][:3]
                    team_j = ticker_j.split('-')[-1][:3]
                    
                    if team_i != team_j:
                        # Markets for opposite teams are mutually exclusive
                        # P(Team A wins by > 5 AND Team B wins by > 5) = 0
                        joint_prob = 0.0
                    else:
                        # Markets for same team are subsets (ladder)
                        # P(Win > 10 AND Win > 5) = P(Win > 10)
                        # Thresholds are implicit in their P(hit) values (monotone)
                        joint_prob = min(probs[i], probs[j])
                        
                    sigma[i, j] = joint_prob - (probs[i] * probs[j])

        # 3. Adjust Edge for Fees
        # Fee per contract (maker) is 0.0175 * P * (1-P)
        # We subtract this friction from the edge
        adjusted_edges = np.zeros(n)
        for i in range(n):
            p = prices[i]
            # Probable fee per contract
            fee_per = 0.0175 * p * (1.0 - p)
            adjusted_edges[i] = edges[i] - fee_per

        # 4. Solve w = Σ⁻¹E
        try:
            # Tikhonov Regularization: Add diagonal load to prevent singular matrix
            # This is essential for highly correlated spread ladders
            reg_sigma = sigma + np.eye(n) * 0.01  # 1% load
            unconstrained_w = np.linalg.solve(reg_sigma, adjusted_edges)
        except np.linalg.LinAlgError:
            # Fallback to independent Kelly if Σ is singular
            unconstrained_w = adjusted_edges / (np.array(probs) * (1 - np.array(probs)) + 1e-6)

        # 4. Apply Kelly Fraction and Constraints
        # Kelly fraction
        w = unconstrained_w * self.kelly_fraction
        
        # Clip by max ticker exposure
        max_w = self.max_ticker_exposure / self.bankroll
        w = np.clip(w, -max_w, max_w)
        
        # SAFETY CONSTRAINT: NO LONG ON NEGATIVE EDGE
        # If price > prob (edge < 0), weight cannot be positive (buy)
        # We can still be short (sell) as that harvests the negative edge
        for i in range(len(tickers)):
            if edges[i] < 0 and w[i] > 0:
                w[i] = 0.0
        
        return dict(zip(tickers, w))

    def evaluate_rebalancing(
        self,
        tickers: List[str],
        current_weights: Dict[str, float], # current position / bankroll
        optimal_weights: Dict[str, float], # smoothed optimal weights from SMA
        bids: Dict[str, int],
        asks: Dict[str, int],
        individual_edges: Dict[str, float], # New: individual model edge per ticker
        current_positions: Dict[str, int] = None, # New: actual contract counts
        is_toxic: Dict[str, bool] = None,
        min_scale_up_edge: float = 0.02 # 2% default min edge to buy
    ) -> List[RebalancingAction]:
        """
        Compare smoothed optimal to current and generate actions.
        """
        actions = []
        is_toxic = is_toxic or {}
        
        for ticker in tickers:
            opt_w = optimal_weights.get(ticker, 0.0)
            curr_w = current_weights.get(ticker, 0.0)
            diff = opt_w - curr_w
            curr_pos = current_positions.get(ticker, 0)
            
            # Market spread calculation
            bid_p = bids.get(ticker, 0)
            ask_p = asks.get(ticker, 100)
            market_spread = ask_p - bid_p
            
            # ASYMMETRIC BANDS (with small epsilon for float robustness)
            # Use a dampened target weight (dampened_opt_w) to slow down reactions
            dampened_opt_w = opt_w 
            
            if diff > self.scale_up_band + 1e-9:
                # SCALE UP (BUY)
                # Apply min_trade_spread filter for entries/scale-ups
                if market_spread < self.min_trade_spread:
                    continue
                    
                price = max(5, min(95, bid_p + 1))
                # Re-calculate model_prob from mid-price and mid-edge
                mid_price = (bid_p + ask_p) / 2.0 / 100.0
                model_prob = individual_edges.get(ticker, 0.0) + mid_price
                exec_edge = model_prob - (price / 100.0)

                fee_impact = 0.0175 * (price/100.0) * (1.0 - price/100.0)
                
                if (exec_edge - fee_impact) < min_scale_up_edge:
                    continue
                    
                action_type = 'buy'
                reason = "Multi-Asset Scale Up"
                dampened_opt_w = curr_w + (diff * 0.5) # Only close 50% of the gap
                
            elif diff < -self.derisk_band - 1e-9:
                # DERISK (SELL)
                # If market spread is too tight, skip de-risking unless emergency
                # This prevents "freak selling" when the spread tightens
                if market_spread < self.min_trade_spread and abs(curr_pos) > 0:
                    continue
                    
                # For de-risking (selling), we allow up to 99c because we are RELEASING capital
                price = max(1, min(99, ask_p - 1))
                mid_price = (bid_p + ask_p) / 2.0 / 100.0
                model_prob = individual_edges.get(ticker, 0.0) + mid_price
                exec_edge = (price / 100.0) - model_prob
                
                if exec_edge < 0.01: # Minimal edge to bother de-risking if not forced
                    continue

                action_type = 'sell'
                reason = "Multi-Asset De-risk"
                dampened_opt_w = curr_w + (diff * 0.5) # Only close 50% of the gap
                
            elif (curr_pos > 0 and bid_p >= 95):
                # GUARANTEED HARVESTING (LONG)
                # Allow 99c for harvesting
                price = max(1, min(99, ask_p - 1))
                mid_price = (bid_p + ask_p) / 2.0 / 100.0
                model_prob = individual_edges.get(ticker, 0.0) + mid_price
                exec_edge = (price / 100.0) - model_prob
                
                if exec_edge < 0.0: continue
                
                action_type = 'sell'
                reason = "Guaranteed Harvesting"
                dampened_opt_w = 0 # Force exit
            elif (curr_pos < 0 and ask_p <= 5):
                # GUARANTEED HARVESTING (SHORT)
                # Allow 1c for harvesting
                price = max(1, min(99, bid_p + 1))
                mid_price = (bid_p + ask_p) / 2.0 / 100.0
                model_prob = individual_edges.get(ticker, 0.0) + mid_price
                exec_edge = model_prob - (price / 100.0)
                
                if exec_edge < 0.0: continue

                action_type = 'buy'
                reason = "Guaranteed Harvesting"
                dampened_opt_w = 0 # Force exit
            elif is_toxic.get(ticker, False):
                # TOXIC EXIT (BYPASS BANDS)
                # Allow full range for toxic exit
                if curr_pos > 0:
                    action_type = 'sell'
                    price = max(1, min(99, ask_p - 1))
                else:
                    action_type = 'buy'
                    price = max(1, min(99, bid_p + 1))
                
                mid_price = (bid_p + ask_p) / 2.0 / 100.0
                model_prob = individual_edges.get(ticker, 0.0) + mid_price
                exec_edge = (price / 100.0) - model_prob if action_type == 'sell' else model_prob - (price / 100.0)
                
                if exec_edge < -0.05: # Even for toxic, don't dump into a black hole?
                    pass

                reason = "Toxic Exit"
                dampened_opt_w = 0 # Force exit
            else:
                continue

            # 2. Calculate Size robustly using Target Contracts
            # Use the dampened target weight
            if dampened_opt_w > 0:
                # Target is LONG: dollar_cost is the market price (how much we pay)
                target_pos = int((dampened_opt_w * self.bankroll) / (max(price, 1) / 100.0))
            elif dampened_opt_w < 0:
                # Target is SHORT: dollar_cost is the exposure cost (100 - price)
                sell_exposure = 100 - price
                target_pos = -int((abs(dampened_opt_w) * self.bankroll) / (max(sell_exposure, 1) / 100.0))
            else:
                target_pos = 0

            size = abs(target_pos - curr_pos)

            if size > 0:
                actions.append(RebalancingAction(
                    ticker=ticker,
                    action=action_type,
                    size=size,
                    price=int(price),
                    reason=reason,
                    edge=exec_edge,
                    target_pos=target_pos
                ))
                
        return actions
