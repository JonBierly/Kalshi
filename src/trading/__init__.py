"""Trading module for automated and paper trading."""

from .paper_trading import PaperTradingEngine, Position, ClosedTrade
from .strategy import TradingStrategy, RiskManager, TradeSignal

__all__ = [
    'PaperTradingEngine',
    'Position',
    'ClosedTrade',
    'TradingStrategy',
    'RiskManager',
    'TradeSignal'
]
