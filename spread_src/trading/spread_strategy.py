"""
Spread trading strategy - find edges in spread markets.

Combines:
1. Prediction edge: Model prob vs market price
2. Arbitrage edge: Cross-market inconsistencies
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from spread_src.data.spread_markets import SpreadMarket, check_spread_arbitrage, parse_spread_ticker


@dataclass
class SpreadTradeSignal:
    """Signal for trading a spread market."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    ticker: str
    market_team: str
    spread: float
    side: str  # 'YES' or 'NO'
    contracts: int
    price: float
    edge: float
    reason: str
    signal_type: str  # 'PREDICTION' or 'ARBITRAGE'
    
    model_prob: float = 0.0
    market_prob: float = 0.0
    ci_90_lower: float = 0.0
    ci_90_upper: float = 0.0


class SpreadTradingStrategy:
    """
    Trading strategy for spread markets.
    """
    
    def __init__(self, 
                 min_edge: float = 0.05,
                 min_confidence: float = 0.90,
                 min_arbitrage: float = 0.03,
                 fractional_kelly: float = 0.5):
        """
        Args:
            min_edge: Minimum prediction edge (5%)
            min_confidence: Require 90% confidence in edge
            min_arbitrage: Minimum arbitrage profit (3¢)
            fractional_kelly: Kelly fraction for position sizing
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_arbitrage = min_arbitrage
        self.fractional_kelly = fractional_kelly
    
    def evaluate_markets(self,
                        markets: List[SpreadMarket],
                        model_predictions: Dict,
                        bankroll: float) -> List[SpreadTradeSignal]:
        """
        Evaluate all spread markets for a game.
        
        Args:
            markets: List of spread markets for this game
            model_predictions: Dict from SpreadDistributionModel
            bankroll: Available capital
        
        Returns:
            List of trade signals
        """
        signals = []
        
        # 1. Check for arbitrage first (risk-free)
        arb_opportunities = check_spread_arbitrage(markets)
        if arb_opportunities:
            arb_signals = self._create_arbitrage_signals(arb_opportunities, markets, bankroll)
            signals.extend(arb_signals)
        
        # 2. Check prediction-based edges
        for market in markets:
            signal = self._evaluate_single_market(market, model_predictions, bankroll)
            if signal and signal.action != 'HOLD':
                signals.append(signal)
        
        return signals
    
    def _evaluate_single_market(self,
                                market: SpreadMarket,
                                model_predictions: Dict,
                                bankroll: float) -> Optional[SpreadTradeSignal]:
        """
        Evaluate a single spread market.
        
        Returns:
            TradeSignal or None
        """
        # Parse market
        try:
            ticker_info = parse_spread_ticker(market.ticker)
        except ValueError:
            return None
        
        spread = ticker_info['spread_value']
        team = ticker_info['spread_team']
        home_team = ticker_info['home_team']
        
        # Find this spread in model predictions
        if spread not in model_predictions['thresholds']:
            # Model didn't predict this specific spread
            # Could interpolate, but skip for now
            return None
        
        idx = model_predictions['thresholds'].index(spread)
        
        # Determine if this is home or away market
        is_home_market = (team == home_team)
        
        if is_home_market:
            # Market: "Home wins by > X"
            model_prob = model_predictions['probabilities'][idx]
            ci_90_lower = model_predictions['ci_90_lower'][idx]
            ci_90_upper = model_predictions['ci_90_upper'][idx]
        else:
            # Market: "Away wins by > X"
            # Need to adjust - model predicts P(home_diff > X)
            # We want P(away_diff > X) = P(home_diff < -X) = 1 - P(home_diff > -X)
            
            # Find -X in predictions
            neg_spread_idx = None
            for i, t in enumerate(model_predictions['thresholds']):
                if abs(t + spread) < 0.1:  # Close to -spread
                    neg_spread_idx = i
                    break
            
            if neg_spread_idx is None:
                return None  # Can't compute
            
            model_prob = 1 - model_predictions['probabilities'][neg_spread_idx]
            ci_90_lower = 1 - model_predictions['ci_90_upper'][neg_spread_idx]
            ci_90_upper = 1 - model_predictions['ci_90_lower'][neg_spread_idx]
        
        # Market price
        market_prob = market.yes_ask / 100.0
        
        # Edge
        edge = model_prob - market_prob
        
        # 90% confidence check
        if ci_90_lower <= market_prob:
            # Not confident enough
            return SpreadTradeSignal(
                action='HOLD',
                ticker=market.ticker,
                market_team=team,
                spread=spread,
                side='YES',
                contracts=0,
                price=market.yes_ask,
                edge=edge,
                reason=f"Not 90% confident (CI lower {ci_90_lower:.1%} <= market {market_prob:.1%})",
                signal_type='PREDICTION',
                model_prob=model_prob,
                market_prob=market_prob,
                ci_90_lower=ci_90_lower,
                ci_90_upper=ci_90_upper
            )
        
        # Edge check
        if edge < self.min_edge:
            # Edge too small
            return SpreadTradeSignal(
                action='HOLD',
                ticker=market.ticker,
                market_team=team,
                spread=spread,
                side='YES',
                contracts=0,
                price=market.yes_ask,
                edge=edge,
                reason=f"Edge {edge:+.1%} < minimum {self.min_edge:.1%}",
                signal_type='PREDICTION',
                model_prob=model_prob,
                market_prob=market_prob
            )
        
        # We have edge! Calculate position size
        contracts = self._calculate_position_size(
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            bankroll=bankroll
        )
        
        return SpreadTradeSignal(
            action='BUY',
            ticker=market.ticker,
            market_team=team,
            spread=spread,
            side='YES',
            contracts=contracts,
            price=market.yes_ask,
            edge=edge,
            reason=f"Edge {edge:+.1%}, 90% confident",
            signal_type='PREDICTION',
            model_prob=model_prob,
            market_prob=market_prob,
            ci_90_lower=ci_90_lower,
            ci_90_upper=ci_90_upper
        )
    
    def _calculate_position_size(self,
                                 model_prob: float,
                                 market_prob: float,
                                 edge: float,
                                 bankroll: float) -> int:
        """
        Kelly criterion for position sizing.
        """
        # Kelly fraction
        b = (1 - market_prob) / market_prob  # Net odds
        p = model_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, kelly_fraction)
        
        # Apply fractional Kelly
        adj_kelly = kelly_fraction * self.fractional_kelly
        
        # Cap at 5% of bankroll
        adj_kelly = min(adj_kelly, 0.05)
        
        # Calculate contracts
        capital = bankroll * adj_kelly
        contracts = int(capital / market_prob)
        
        return max(contracts, 1)  # At least 1 contract
    
    def _create_arbitrage_signals(self,
                                  arb_opportunities: List[Dict],
                                  markets: List[SpreadMarket],
                                  bankroll: float) -> List[SpreadTradeSignal]:
        """
        Create trade signals for arbitrage opportunities.
        """
        signals = []
        
        for arb in arb_opportunities:
            if arb['arbitrage_cents'] < self.min_arbitrage * 100:
                continue
            
            # Find the markets involved
            lower_market = None
            higher_market = None
            
            for market in markets:
                info = parse_spread_ticker(market.ticker)
                if (info['spread_team'] == arb['team'] and 
                    abs(info['spread_value'] - arb['lower_spread']) < 0.1):
                    lower_market = market
                if (info['spread_team'] == arb['team'] and 
                    abs(info['spread_value'] - arb['higher_spread']) < 0.1):
                    higher_market = market
            
            if not (lower_market and higher_market):
                continue
            
            # Arbitrage strategy: Buy lower spread, Sell higher spread
            # Risk-free profit!
            
            # Equal position size
            contracts = int(bankroll * 0.02 / (lower_market.yes_ask / 100))  # 2% of bankroll
            
            signals.append(SpreadTradeSignal(
                action='BUY',
                ticker=lower_market.ticker,
                market_team=arb['team'],
                spread=arb['lower_spread'],
                side='YES',
                contracts=contracts,
                price=lower_market.yes_ask,
                edge=arb['arbitrage_cents'] / 100,
                reason=f"ARBITRAGE: Buy >{arb['lower_spread']}",
                signal_type='ARBITRAGE'
            ))
            
            signals.append(SpreadTradeSignal(
                action='SELL',
                ticker=higher_market.ticker,
                market_team=arb['team'],
                spread=arb['higher_spread'],
                side='YES',
                contracts=contracts,
                price=higher_market.yes_bid,
                edge=arb['arbitrage_cents'] / 100,
                reason=f"ARBITRAGE: Sell >{arb['higher_spread']}",
                signal_type='ARBITRAGE'
            ))
        
        return signals
