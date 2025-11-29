"""
Paper Trading Script - Test betting strategy without real money.

Runs live tracking with virtual portfolio and displays trade signals.
"""

import os
import sys
import time
from datetime import datetime

from src.inference.tracker import OddsTracker
from src.trading.paper_trading import PaperTradingEngine
from src.trading.strategy import TradingStrategy
from src.data.database import DatabaseManager


def get_model_prediction(tracker, game, ticker):
    """Get prediction from tracker's model (copied from track_odds.py)."""
    # Setup context if needed
    if tracker.orch.prediction_engine.current_game_id != game['gameId']:
        try:
            tracker.orch.setup_game_context(
                game['gameId'], 
                game['homeTeam']['teamId'], 
                game['awayTeam']['teamId']
            )
            tracker.orch.feature_engine.reset()
        except Exception:
            return None
    
    # Get live data  
    try:
        live_data = tracker.orch.live_client.get_live_game_data(game['gameId'])
    except Exception:
        # Game hasn't started yet or API error
        return None
    
    if not live_data:
        return None
    
    # Check if game has actually started
    try:
        home_stats = live_data['homeTeam']['statistics']
        away_stats = live_data['awayTeam']['statistics']
        
        if not home_stats or not away_stats:
            return None
    except (KeyError, TypeError):
        return None
    
    # Update feature engine
    try:
        tracker.orch.feature_engine.home_stats = {
            'fgm': home_stats['fieldGoalsMade'], 
            'fga': home_stats['fieldGoalsAttempted'],
            'fg3m': home_stats['threePointersMade'], 
            'to': home_stats['turnovers'],
            'reb': home_stats['reboundsTotal']
        }
        tracker.orch.feature_engine.away_stats = {
            'fgm': away_stats['fieldGoalsMade'], 
            'fga': away_stats['fieldGoalsAttempted'],
            'fg3m': away_stats['threePointersMade'], 
            'to': away_stats['turnovers'],
            'reb': away_stats['reboundsTotal']
        }
    except (KeyError, TypeError):
        return None
    
    # Calculate features
    remaining_time = 0
    period = live_data.get('period', 0)
    
    if 'gameClock' in live_data:
        t_str = live_data['gameClock'].replace('PT', '').replace('M', ':').replace('S', '')
        if ':' in t_str:
            try:
                m, s = t_str.split(':')
                remaining_time = int(m) * 60 + float(s)
            except (ValueError, AttributeError):
                pass
    
    total_seconds = remaining_time
    if period <= 4:
        total_seconds += (4 - period) * 720
    
    try:
        live_features = {
            'score_diff': live_data['homeTeam']['score'] - live_data['awayTeam']['score'],
            'seconds_remaining': total_seconds,
            'home_efg': tracker.orch.feature_engine._calc_efg(tracker.orch.feature_engine.home_stats),
            'away_efg': tracker.orch.feature_engine._calc_efg(tracker.orch.feature_engine.away_stats),
            'turnover_diff': tracker.orch.feature_engine.home_stats['to'] - tracker.orch.feature_engine.away_stats['to'],
            'home_rebound_rate': tracker.orch.feature_engine._calc_reb_rate(
                tracker.orch.feature_engine.home_stats['reb'], 
                tracker.orch.feature_engine.away_stats['reb']
            ),
            'game_id': game['gameId'],
            'home_team_id': game['homeTeam']['teamId'],
            'away_team_id': game['awayTeam']['teamId']
        }
    except (KeyError, TypeError):
        return None
    
    full_feats = {**live_features, **tracker.orch.prediction_engine.current_game_context}
    model_result = tracker.orch.prediction_engine.predict_with_confidence(full_feats)
    
    # Get Kalshi odds
    kalshi_prob = 0.0
    yes_price = 0
    
    try:
        details = tracker.kalshi.get_market_details(ticker)
        if details:
            yes_price = details.get('yes_ask', 0)
            kalshi_prob = yes_price / 100.0
    except Exception:
        pass
    
    clock_display = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
    
    return {
        'live_data': live_data,
        'model_result': model_result,
        'kalshi_prob': kalshi_prob,
        'yes_price': yes_price,
        'clock_display': clock_display,
        'period': period
    }


def display_trade_signal(game, ticker, pred, signal):
    """Display trade signal with reasoning."""
    ld = pred['live_data']
    home_team = game['homeTeam']['teamTricode']
    away_team = game['awayTeam']['teamTricode']
    
    prob = pred['model_result']['probability']
    market_price = pred['yes_price']
    
    print(f"\n┌─ {game['gameCode']} ─ {home_team} vs {away_team}")
    print(f"│  Score: {ld['homeTeam']['score']}-{ld['awayTeam']['score']} │ Q{pred['period']} {pred['clock_display']}")
    print(f"├─ Odds:")
    print(f"│  Model:  {prob:>6.1%}")
    print(f"│  Market: {pred['kalshi_prob']:>6.1%}  ({market_price}¢)")
    print(f"│  Edge:   {signal.edge:>+6.1%}")
    print(f"├─ Position:")
    print(f"│  Current: {signal.current_contracts} contracts")
    print(f"│  Target:  {signal.target_contracts} contracts")
    
    if signal.action != 'HOLD':
        cost = signal.contracts * (signal.price / 100)
        print(f"├─ ➤ {signal.action} SIGNAL")
        print(f"│  Amount: {signal.contracts} contracts")
        print(f"│  Price: {signal.price:.0f}¢")
        print(f"│  Value: ${cost:.2f}")
        if signal.expected_value != 0:
            print(f"│  EV: ${signal.expected_value:+.2f}")
        print(f"└─ {signal.reason}")
    else:
        print(f"├─ ✋ HOLD")
        print(f"└─ {signal.reason}")


def display_portfolio_summary(engine):
    """Display current portfolio status."""
    summary = engine.get_portfolio_summary()
    
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Balance:           ${summary['balance']:>10,.2f}")
    print(f"Total Exposure:    ${summary['total_exposure']:>10,.2f}  ({summary['open_positions']} positions)")
    print(f"Realized P&L:      ${summary['total_pnl']:>+10,.2f}  ({summary['roi_percent']:+.2f}%)")
    
    if summary['closed_trades'] > 0:
        print(f"\nClosed Trades:     {summary['closed_trades']}")
        print(f"Wins/Losses:       {summary['wins']}/{summary['losses']}  ({summary['win_rate_percent']:.1f}% win rate)")
        print(f"Avg Win:           ${summary['avg_win']:>10,.2f}")
        print(f"Avg Loss:          ${summary['avg_loss']:>10,.2f}")
    
    print("="*80)


def check_and_settle_games(engine, db):
    """Check if any games have finished and settle positions."""
    from src.data.database import Game
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=db.engine)
    session = Session()
    
    settled_any = False
    
    try:
        for game_id in list(engine.open_positions.keys()):
            # Query game status
            game = session.query(Game).filter_by(game_id=game_id).first()
            
            if game and game.home_score is not None and game.away_score is not None:
                # Game has finished, check if we have final score
                if game.home_score > game.away_score:
                    outcome = True  # Home win
                elif game.away_score > game.home_score:
                    outcome = False  # Away win
                else:
                    continue  # Game might not be truly final (OT, etc.)
                
                # Settle position
                success, message, pnl = engine.close_position(game_id, outcome)
                
                if success:
                    print(f"\n{'='*80}")
                    print(f"GAME SETTLED: {game_id}")
                    print(message)
                    print(f"{'='*80}")
                    settled_any = True
    
    finally:
        session.close()
    
    return settled_any


def main(model_type='lr', interval=30, starting_balance=10000.0):
    """
    Run active paper trading simulation.
    """
    # Initialize components
    print("="*80)
    print("ACTIVE TRADING ENGINE")
    print("="*80)
    print(f"Model: {model_type.upper()}")
    print(f"Update Interval: {interval}s")
    print(f"Starting Balance: ${starting_balance:,.2f}")
    print("="*80)
    print()
    
    engine = PaperTradingEngine(starting_balance=starting_balance)
    strategy = TradingStrategy(fractional_kelly=1)  # Full Kelly
    
    print(f"Loaded engine state: ${engine.balance:,.2f} balance\n")
    
    # Get credentials
    key_id = os.environ.get('KALSHI_KEY_ID', "a40ff1c6-12ac-4a6c-9669-ffe12f3de235")
    
    try:
        # Setup tracker
        tracker = OddsTracker(key_id, model_type=model_type)
        tracker.setup()
        
        db = DatabaseManager()
        
        print(f"\nTracking {len(tracker.active_matches)} games...")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"Update #{iteration} at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            
            # Check for settled games
            settled_any = check_and_settle_games(engine, db)
            if settled_any:
                engine.save_state()
            
            if not tracker.active_matches:
                print("No active games matched.")
                time.sleep(interval)
                continue
            
            # Process ALL games (Active Management)
            for match in tracker.active_matches:
                game = match['nba_game']
                ticker = match['kalshi_ticker']
                game_id = game['gameId']
                
                # Get prediction
                pred = get_model_prediction(tracker, game, ticker)
                if not pred:
                    print(f"Skipping {game['gameCode']} (no data)")
                    continue
                
                # Get current position if any
                current_pos = engine.open_positions.get(game_id)
                
                # Evaluate Market (Rebalancing Logic)
                signal = strategy.evaluate_market(
                    game_id=game_id,
                    model_result=pred['model_result'],
                    market_price=pred['yes_price'],
                    bankroll=engine.balance + engine.get_portfolio_summary()['total_exposure'], # Use total equity
                    current_position=current_pos
                )
                
                # Display signal
                display_trade_signal(game, ticker, pred, signal)
                
                # Execute Trade if needed
                if signal.action != 'HOLD':
                    success, message = engine.execute_trade(
                        game_id=game_id,
                        ticker=ticker,
                        action=signal.action,
                        contracts=signal.contracts,
                        price=signal.price,
                        reason=signal.reason
                    )
                    
                    if success:
                        print(f"✓ {message}")
                        engine.save_state()
                    else:
                        print(f"✗ Failed: {message}")
            
            # Display portfolio
            display_portfolio_summary(engine)
            
            # Save state
            engine.save_state()
            
            # Sleep
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")
        engine.save_state()
        print("State saved.")
        
        # Final summary
        display_portfolio_summary(engine)
    
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        engine.save_state()


if __name__ == "__main__":
    # Run with LR model by default
    # Adjust parameters as needed
    main(
        model_type='lr',      # or 'xgboost'
        interval=20,          # seconds between updates
        starting_balance=100.0
    )
