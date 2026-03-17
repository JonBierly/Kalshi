import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import DatabaseManager

def evaluate_calibrated_trades():
    db = DatabaseManager()
    
    # Query all trades since model was deployed/trained on Feb 10th
    query_str = """
        SELECT side, fill_price, model_fair_value, realized_pnl, size, status
        FROM trades
        WHERE timestamp >= '2026-02-10' AND status = 'closed' AND realized_pnl IS NOT NULL
    """
    trades_df = pd.read_sql(query_str, db.engine)
    
    if len(trades_df) == 0:
        print("No resolved trades found for backtest.")
        return
        
    print(f"Loaded {len(trades_df)} resolved trades from database.")
    
    # Load beta calibrator
    calibrator = joblib.load('models/beta_calibrator_v1.pkl')
    
    # model_fair_value is the raw probability * 100 (e.g. 64.5%)
    raw_probs = np.clip(trades_df['model_fair_value'].values / 100.0, 1e-6, 1.0 - 1e-6)
    
    # Calibrate probabilities
    cal_probs = calibrator.predict(raw_probs)
    
    # Fill price is out of 100 (e.g. 55 cents)
    prices = trades_df['fill_price'].values / 100.0
    
    # Edge = (Prob - Price) for BUY, (Price - Prob) for SELL
    is_buy = trades_df['side'].str.upper() == 'BUY'
    
    raw_edge = np.where(is_buy, raw_probs - prices, prices - raw_probs)
    cal_edge = np.where(is_buy, cal_probs - prices, prices - cal_probs)

    # Minimum edge threshold used to maintain a trade (the simple_ev trader cancels < 2%)
    MIN_EDGE_THRESHOLD = 0.02
    
    raw_would_keep = raw_edge >= MIN_EDGE_THRESHOLD
    cal_would_keep = cal_edge >= MIN_EDGE_THRESHOLD
    
    # Only evaluate trades the models would ACTUALLY choose to keep based on the current threshold
    actual_pnl = trades_df.loc[raw_would_keep, 'realized_pnl'].sum()
    calibrated_pnl = trades_df.loc[cal_would_keep, 'realized_pnl'].sum()
    
    # Trades that the raw model approved, but calibrator blocked
    blocked_trades_mask = raw_would_keep & ~cal_would_keep
    blocked_trades_count = blocked_trades_mask.sum()
    blocked_pnl = trades_df.loc[blocked_trades_mask, 'realized_pnl'].sum()

    # Trades that the raw model blocked, but calibrator approved
    added_trades_mask = ~raw_would_keep & cal_would_keep
    added_trades_count = added_trades_mask.sum()
    added_pnl = trades_df.loc[added_trades_mask, 'realized_pnl'].sum()
    
    print("\n" + "="*50)
    print("BETA CALIBRATOR BACKTEST RESULTS (Recent 163 Games)")
    print("="*50)
    print(f"Total resolved trades evaluated by Raw Mode: {raw_would_keep.sum()}")
    print(f"Trades the Calibrator would have BLOCKED: {blocked_trades_count}")
    print(f"Trades the Calibrator would have ADDED: {added_trades_count}")
    
    print("\n--- Profit / Loss Comparison ---")
    print(f"Current PnL with Raw Probabilities:     ${actual_pnl:,.2f}")
    print(f"Simulated PnL with Calibrated Probs:    ${calibrated_pnl:,.2f}")
    print("-"*50)
    print(f"Net PnL Improvement:                    ${(calibrated_pnl - actual_pnl):,.2f}")
    print("="*50)
    print(f"Details:")
    print(f"• The {blocked_trades_count} blocked trades generated ${blocked_pnl:,.2f} of loss. Skipping them SAVED ${-blocked_pnl:,.2f}.")
    print(f"• The {added_trades_count} added trades generated ${added_pnl:,.2f} in PnL.")
    print(f"• Net effect on PnL: ${(-blocked_pnl + added_pnl):,.2f}")

if __name__ == "__main__":
    evaluate_calibrated_trades()
