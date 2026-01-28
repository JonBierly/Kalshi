import unittest
from unittest.mock import MagicMock, patch
from spread_src.execution.portfolio import Portfolio

class TestPortfolioCashFix(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(max_exposure=100.0, db_path=':memory:')
        self.portfolio.cash = 1000.0
        self.portfolio.realized_pnl = 50.0
        self.portfolio.positions = {'TEST-TICKER': 10}
        self.portfolio.cost_basis = {'TEST-TICKER': 40.0}

    def test_settle_market_no_longer_updates_cash_or_pnl(self):
        """Verify settle_market doesn't change cash/pnl (API is truth)."""
        initial_cash = self.portfolio.cash
        initial_pnl = self.portfolio.realized_pnl
        
        # Settle the market
        self.portfolio.settle_market('TEST-TICKER', outcome=True)
        
        # Cash and P&L should remain unchanged
        self.assertEqual(self.portfolio.cash, initial_cash)
        self.assertEqual(self.portfolio.realized_pnl, initial_pnl)

    @patch('sqlite3.connect')
    def test_settle_unsettled_trades_no_longer_updates_cash_or_pnl(self, mock_connect):
        """Verify settle_unsettled_trades doesn't change cash/pnl (API is truth)."""
        initial_cash = self.portfolio.cash
        initial_pnl = self.portfolio.realized_pnl
        
        # Mock database items
        mock_cursor = mock_connect.return_value.cursor.return_value
        mock_cursor.fetchall.return_value = [
            (1, 'TEST-TICKER', 'buy', 40.0, 10)
        ]
        
        # Mock Kalshi client
        mock_kalshi = MagicMock()
        mock_kalshi.get_market_details.return_value = {
            'status': 'finalized',
            'result': 'yes'
        }
        
        # Run settlement
        self.portfolio.settle_unsettled_trades(mock_kalshi, MagicMock())
        
        # Cash and P&L should remain unchanged
        self.assertEqual(self.portfolio.cash, initial_cash)
        self.assertEqual(self.portfolio.realized_pnl, initial_pnl)

if __name__ == '__main__':
    unittest.main()
