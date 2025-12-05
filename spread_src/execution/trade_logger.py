"""
Trade logging for analytics and performance tracking.

Logs trades, model predictions, and session metrics to SQLite database.
"""

import sqlite3
from datetime import datetime, date
from typing import Optional, Dict
import os


class TradeLogger:
    """Log trading activity to SQLite database for analytics."""
    
    def __init__(self, db_path='data/nba_data.db'):
        """
        Initialize trade logger.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Market info
                ticker TEXT NOT NULL,
                game_id TEXT,
                
                -- Order details
                side TEXT NOT NULL,
                order_price REAL,
                fill_price REAL,
                size INTEGER,
                
                -- Model context
                model_fair_value REAL,
                model_ci_lower REAL,
                model_ci_upper REAL,
                market_spread REAL,
                seconds_remaining INTEGER,
                
                -- Position tracking
                position_before INTEGER,
                position_after INTEGER,
                
                -- P&L
                realized_pnl REAL,
                status TEXT,
                
                -- Metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                filled_at DATETIME,
                closed_at DATETIME
            )
        ''')
        
        # Table 2: Model Predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Game context
                game_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                
                -- Game state
                seconds_remaining INTEGER,
                actual_score_diff INTEGER,
                
                -- Model prediction
                predicted_prob REAL,
                ci_lower REAL,
                ci_upper REAL,
                
                -- Outcome
                actual_outcome BOOLEAN,
                prediction_error REAL,
                
                -- Metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                settled_at DATETIME
            )
        ''')
        
        # Table 3: Performance Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                session_start DATETIME,
                session_end DATETIME,
                
                -- Trading stats
                total_trades INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                total_wins INTEGER DEFAULT 0,
                total_losses INTEGER DEFAULT 0,
                
                -- P&L metrics
                gross_pnl REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                avg_pnl_per_trade REAL,
                win_rate REAL,
                
                -- Risk metrics
                max_position_size INTEGER,
                max_exposure REAL,
                max_drawdown REAL,
                
                -- Model metrics
                avg_prediction_error REAL,
                prediction_count INTEGER DEFAULT 0,
                
                -- Metadata
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_order_placed(
        self,
        ticker: str,
        side: str,
        price: float,
        size: int,
        game_id: Optional[str] = None,
        model_fair: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        market_spread: Optional[float] = None,
        seconds_remaining: Optional[int] = None,
        position_before: int = 0
    ) -> int:
        """
        Log order placement.
        
        Returns:
            trade_id for this order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (
                ticker, game_id, side, order_price, size,
                model_fair_value, model_ci_lower, model_ci_upper,
                market_spread, seconds_remaining, position_before,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'placed')
        ''', (ticker, game_id, side, price, size, model_fair, ci_lower, ci_upper,
              market_spread, seconds_remaining, position_before))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def log_order_filled(
        self,
        trade_id: int,
        fill_price: float,
        position_after: int,
        timestamp: Optional[datetime] = None
    ):
        """Log order fill."""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades
            SET fill_price = ?,
                position_after = ?,
                status = 'filled',
                filled_at = ?
            WHERE trade_id = ?
        ''', (fill_price, position_after, timestamp, trade_id))
        
        conn.commit()
        conn.close()
    
    def log_order_canceled(self, trade_id: int):
        """Log order cancellation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades
            SET status = 'canceled'
            WHERE trade_id = ?
        ''', (trade_id,))
        
        conn.commit()
        conn.close()
    
    def log_position_closed(
        self,
        trade_id: int,
        realized_pnl: float,
        timestamp: Optional[datetime] = None
    ):
        """Log position closure with P&L."""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades
            SET realized_pnl = ?,
                status = 'closed',
                closed_at = ?
            WHERE trade_id = ?
        ''', (realized_pnl, timestamp, trade_id))
        
        conn.commit()
        conn.close()
    
    def log_prediction(
        self,
        game_id: str,
        ticker: str,
        seconds_remaining: int,
        score_diff: int,
        predicted_prob: float,
        ci_lower: float,
        ci_upper: float
    ) -> int:
        """
        Log model prediction.
        
        Returns:
            prediction_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_predictions (
                game_id, ticker, seconds_remaining, actual_score_diff,
                predicted_prob, ci_lower, ci_upper
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (game_id, ticker, seconds_remaining, score_diff,
              predicted_prob, ci_lower, ci_upper))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_outcome: bool,
        timestamp: Optional[datetime] = None
    ):
        """Update prediction with actual outcome."""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predicted prob to calculate error
        cursor.execute('''
            SELECT predicted_prob FROM model_predictions
            WHERE prediction_id = ?
        ''', (prediction_id,))
        
        result = cursor.fetchone()
        if result:
            predicted_prob = result[0]
            prediction_error = abs(predicted_prob - (1.0 if actual_outcome else 0.0))
            
            cursor.execute('''
                UPDATE model_predictions
                SET actual_outcome = ?,
                    prediction_error = ?,
                    settled_at = ?
                WHERE prediction_id = ?
            ''', (actual_outcome, prediction_error, timestamp, prediction_id))
        
        conn.commit()
        conn.close()
    
    def update_session_metrics(self, target_date: date):
        """
        Calculate and update aggregate metrics for a date.
        
        Args:
            target_date: Date to calculate metrics for
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trade statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN fill_price IS NOT NULL THEN 1 END) as total_fills,
                COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as total_wins,
                COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as total_losses,
                SUM(realized_pnl) as gross_pnl,
                AVG(realized_pnl) as avg_pnl,
                MAX(ABS(position_after)) as max_position
            FROM trades
            WHERE DATE(created_at) = ?
        ''', (target_date,))
        
        trade_stats = cursor.fetchone()
        
        # Get prediction statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as prediction_count,
                AVG(prediction_error) as avg_error
            FROM model_predictions
            WHERE DATE(created_at) = ? AND actual_outcome IS NOT NULL
        ''', (target_date,))
        
        pred_stats = cursor.fetchone()
        
        # Calculate win rate
        total_closed = (trade_stats[2] or 0) + (trade_stats[3] or 0)
        win_rate = (trade_stats[2] / total_closed) if total_closed > 0 else 0.0
        
        # Insert or update metrics
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics (
                date, total_trades, total_fills, total_wins, total_losses,
                gross_pnl, net_pnl, avg_pnl_per_trade, win_rate,
                max_position_size, prediction_count, avg_prediction_error,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            target_date,
            trade_stats[0] or 0,
            trade_stats[1] or 0,
            trade_stats[2] or 0,
            trade_stats[3] or 0,
            trade_stats[4] or 0.0,
            trade_stats[4] or 0.0,  # net_pnl same as gross for now
            trade_stats[5],
            win_rate,
            trade_stats[6],
            pred_stats[0] or 0,
            pred_stats[1],
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
