#!/usr/bin/env python
"""
Test script for market maker system.

Tests logging, settlement, and database functionality without needing live games.
"""

import sys
import os
from datetime import datetime, date

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spread_src.execution.portfolio import Portfolio
from spread_src.execution.trade_logger import TradeLogger


def test_database_connection():
    """Test 1: Database connectivity"""
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    
    try:
        logger = TradeLogger('data/nba_data.db')
        print("✅ TradeLogger initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_order_logging():
    """Test 2: Order placement logging"""
    print("\n" + "="*60)
    print("TEST 2: Order Logging")
    print("="*60)
    
    try:
        logger = TradeLogger('data/nba_data.db')
        
        # Log a fake order
        trade_id = logger.log_order_placed(
            ticker='TEST-NBAHOU-123-B10.5',
            side='buy',
            price=45.0,
            size=3,
            game_id='0012345678',
            model_fair=48.5,
            ci_lower=0.40,
            ci_upper=0.55,
            market_spread=2.0,
            seconds_remaining=1800,
            position_before=0
        )
        
        print(f"✅ Order logged with trade_id: {trade_id}")
        return trade_id
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_fill_logging(trade_id):
    """Test 3: Fill logging"""
    print("\n" + "="*60)
    print("TEST 3: Fill Logging")
    print("="*60)
    
    if not trade_id:
        print("⚠️  Skipped (no trade_id from previous test)")
        return False
    
    try:
        logger = TradeLogger('data/nba_data.db')
        
        # Log a fake fill
        logger.log_order_filled(
            trade_id=trade_id,
            fill_price=45.0,
            position_after=3
        )
        
        print(f"✅ Fill logged for trade_id: {trade_id}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_prediction_logging():
    """Test 4: Model prediction logging"""
    print("\n" + "="*60)
    print("TEST 4: Prediction Logging")
    print("="*60)
    
    try:
        logger = TradeLogger('data/nba_data.db')
        
        # Log a fake prediction
        pred_id = logger.log_prediction(
            game_id='0012345678',
            ticker='TEST-NBAHOU-123-B10.5',
            seconds_remaining=1800,
            score_diff=5,
            predicted_prob=0.485,
            ci_lower=0.40,
            ci_upper=0.55
        )
        
        print(f"✅ Prediction logged with prediction_id: {pred_id}")
        return pred_id
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_position_closing(trade_id):
    """Test 5: Position closing"""
    print("\n" + "="*60)
    print("TEST 5: Position Closing")
    print("="*60)
    
    if not trade_id:
        print("⚠️  Skipped (no trade_id from previous test)")
        return False
    
    try:
        logger = TradeLogger('data/nba_data.db')
        
        # Log position close
        logger.log_position_closed(
            trade_id=trade_id,
            realized_pnl=2.50
        )
        
        print(f"✅ Position closed for trade_id: {trade_id}")
        print(f"   Realized P&L: $2.50")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_portfolio_integration():
    """Test 6: Portfolio with logging"""
    print("\n" + "="*60)
    print("TEST 6: Portfolio Integration")
    print("="*60)
    
    try:
        portfolio = Portfolio(initial_capital=20.0, db_path='data/nba_data.db')
        print("✅ Portfolio initialized with TradeLogger")
        
        # Simulate a fill
        portfolio.update_fill(
            ticker='TEST-NBAHOU-456-B12.5',
            side='buy',
            price=52.0,
            size=2,
            trade_id=None  # Won't log to DB without trade_id
        )
        
        print(f"✅ Fill processed")
        print(f"   Cash: ${portfolio.cash:.2f}")
        print(f"   Position: {portfolio.positions.get('TEST-NBAHOU-456-B12.5', 0)}")
        
        # Settle the position
        portfolio.settle_market('TEST-NBAHOU-456-B12.5', outcome=True)
        print(f"✅ Position settled")
        print(f"   Cash after settlement: ${portfolio.cash:.2f}")
        print(f"   Realized P&L: ${portfolio.realized_pnl:+.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_session_metrics():
    """Test 7: Session metrics update"""
    print("\n" + "="*60)
    print("TEST 7: Session Metrics")
    print("="*60)
    
    try:
        logger = TradeLogger('data/nba_data.db')
        
        # Update metrics for today
        logger.update_session_metrics(date.today())
        
        print(f"✅ Session metrics updated for {date.today()}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def query_test_data():
    """Query and display test data"""
    print("\n" + "="*60)
    print("TEST DATA SUMMARY")
    print("="*60)
    
    import sqlite3
    
    conn = sqlite3.connect('data/nba_data.db')
    cursor = conn.cursor()
    
    # Count test trades
    cursor.execute("SELECT COUNT(*) FROM trades WHERE ticker LIKE 'TEST-%'")
    test_trades = cursor.fetchone()[0]
    print(f"\nTest trades created: {test_trades}")
    
    # Count test predictions
    cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE ticker LIKE 'TEST-%'")
    test_predictions = cursor.fetchone()[0]
    print(f"Test predictions created: {test_predictions}")
    
    # Show latest test trades
    print("\nLatest test trades:")
    cursor.execute("""
        SELECT trade_id, ticker, side, order_price, fill_price, status, realized_pnl
        FROM trades
        WHERE ticker LIKE 'TEST-%'
        ORDER BY trade_id DESC
        LIMIT 5
    """)
    
    print(f"{'ID':<6} {'Ticker':<25} {'Side':<6} {'Order':<8} {'Fill':<8} {'Status':<10} {'P&L':<8}")
    print("-" * 80)
    for row in cursor.fetchall():
        pnl = f"${row[6]:+.2f}" if row[6] is not None else "N/A"
        print(f"{row[0]:<6} {row[1]:<25} {row[2]:<6} {row[3]:<8.1f} {row[4] or 'N/A':<8} {row[5]:<10} {pnl:<8}")
    
    conn.close()


def cleanup_test_data():
    """Clean up test data"""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    response = input("\nDelete test data from database? (y/n): ")
    
    if response.lower() == 'y':
        import sqlite3
        conn = sqlite3.connect('data/nba_data.db')
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trades WHERE ticker LIKE 'TEST-%'")
        trades_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM model_predictions WHERE ticker LIKE 'TEST-%'")
        predictions_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {trades_deleted} test trades")
        print(f"✅ Deleted {predictions_deleted} test predictions")
    else:
        print("⚠️  Test data preserved")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MARKET MAKER SYSTEM TEST SUITE")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"Database: data/nba_data.db")
    
    results = {}
    
    # Run tests
    results['database'] = test_database_connection()
    trade_id = test_order_logging()
    results['order_logging'] = trade_id is not None
    results['fill_logging'] = test_fill_logging(trade_id)
    pred_id = test_prediction_logging()
    results['prediction_logging'] = pred_id is not None
    results['position_closing'] = test_position_closing(trade_id)
    results['portfolio_integration'] = test_portfolio_integration()
    results['session_metrics'] = test_session_metrics()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    # Show test data
    query_test_data()
    
    # Cleanup option
    cleanup_test_data()


if __name__ == "__main__":
    main()
