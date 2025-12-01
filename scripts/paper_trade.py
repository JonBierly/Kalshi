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
        score_diff = live_data['homeTeam']['score'] - live_data['awayTeam']['score']
        required_catchup_rate = abs(score_diff) / (total_seconds + 1)
        
        live_features = {
            'score_diff': score_diff,
            'seconds_remaining': total_seconds,
            'home_efg': tracker.orch.feature_engine._calc_efg(tracker.orch.feature_engine.home_stats),
            'away_efg': tracker.orch.feature_engine._calc_efg(tracker.orch.feature_engine.away_stats),
            'turnover_diff': tracker.orch.feature_engine.home_stats['to'] - tracker.orch.feature_engine.away_stats['to'],
            'home_rebound_rate': tracker.orch.feature_engine._calc_reb_rate(
                tracker.orch.feature_engine.home_stats['reb'], 
                tracker.orch.feature_engine.away_stats['reb']
            ),
            'required_catchup_rate': required_catchup_rate,
            'game_id': game['gameId'],
            'home_team_id': game['homeTeam']['teamId'],
            'away_team_id': game['awayTeam']['teamId']
        }
    except (KeyError, TypeError):
        return None
    
    full_feats = {**live_features, **tracker.orch.prediction_engine.current_game_context}
    model_result = tracker.orch.prediction_engine.predict_with_confidence(full_feats)
    
    # Add context for strategy decisions
    model_result['seconds_remaining'] = total_seconds
    model_result['score_diff'] = live_features['score_diff']
    model_result['required_catchup_rate'] = required_catchup_rate
    
    # Get Kalshi bid/ask prices
    # BUY at ASK (what sellers want), SELL at BID (what buyers offer)
    yes_bid = 0    # Price you receive when SELLING YES
    yes_ask = 0    # Price you pay when BUYING YES
    no_bid = 0     # Price you receive when SELLING NO
    no_ask = 0     # Price you pay when BUYING NO
    
    try:
        details = tracker.kalshi.get_market_details(ticker)
        if details:
            yes_bid = details.get('yes_bid', 0)
            yes_ask = details.get('yes_ask', 0)
            no_bid = details.get('no_bid', 0)
            no_ask = details.get('no_ask', 0)
    except Exception:
        pass
    
    # Use mid-price for implied probability display
    yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else yes_ask
    kalshi_prob = yes_mid / 100.0 if yes_mid > 0 else 0.0
    
    clock_display = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
    
    return {
        'live_data': live_data,
        'model_result': model_result,
        'kalshi_prob': kalshi_prob,
        'yes_bid': yes_bid,
        'yes_ask': yes_ask,
        'no_bid': no_bid,
        'no_ask': no_ask,
        'clock_display': clock_display,
        'period': period
    }



def display_trade_signal(game, signal, pred):
    """Display trade signal in table format with market vs model comparison."""
    ld = pred['live_data']
    home_team = game['homeTeam']['teamTricode']
    away_team = game['awayTeam']['teamTricode']
    
    home_score = ld['homeTeam']['score']
    away_score = ld['awayTeam']['score']
    
    # Model probabilities
    model_yes_prob = pred['model_result']['probability']
    model_no_prob = 1 - model_yes_prob
    
    # Market bid/ask prices
    yes_bid = pred['yes_bid']
    yes_ask = pred['yes_ask']
    no_bid = pred['no_bid']
    no_ask = pred['no_ask']
    
    # Mid prices for market probability (average of bid/ask)
    yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else yes_ask
    no_mid = (no_bid + no_ask) / 2 if (no_bid and no_ask) else no_ask
    market_yes_prob = yes_mid / 100.0
    market_no_prob = no_mid / 100.0
    
    # Calculate edges (using mid prices)
    yes_edge = model_yes_prob - market_yes_prob
    no_edge = model_no_prob - market_no_prob
    
    # Header with game info
    print(f"\n{'='*70}")
    print(f"{game['gameCode']:^70}")
    print(f"{home_team} {home_score:>3} vs {away_score:<3} {away_team}  │  Q{pred['period']} {pred['clock_display']:>8}")
    print(f"{'='*70}")
    
    # Odds comparison table
    print(f"{'':20} │ {'YES (Home)':^20} │ {'NO (Away)':^20}")
    print(f"{'-'*20}─┼─{'-'*20}─┼─{'-'*20}")
    print(f"{'Model Probability':20} │ {model_yes_prob:^20.1%} │ {model_no_prob:^20.1%}")
    print(f"{'Market Bid':20} │ {yes_bid:^20.0f}¢ │ {no_bid:^20.0f}¢")
    print(f"{'Market Ask':20} │ {yes_ask:^20.0f}¢ │ {no_ask:^20.0f}¢")
    print(f"{'Market Mid (Prob)':20} │ {market_yes_prob:^20.1%} │ {market_no_prob:^20.1%}")
    print(f"{'Edge':20} │ {yes_edge:^+20.1%} │ {no_edge:^+20.1%}")
    print(f"{'-'*70}")
    
    # Position info
    current_side_display = signal.side if signal.current_contracts > 0 else "NONE"
    print(f"Position: {signal.current_contracts} contracts ({current_side_display}) → {signal.target_contracts} contracts ({signal.side})")
    
    # Action
    if signal.action != 'HOLD':
        cost = signal.contracts * (signal.price / 100)
        action_symbol = "📈 BUY" if signal.action == 'BUY' else "📉 SELL"
        print(f"\n{action_symbol} {signal.contracts} {signal.side} @ {signal.price:.0f}¢ = ${cost:.2f}")
        if signal.expected_value != 0:
            print(f"Expected Value: ${signal.expected_value:+.2f}")
        print(f"Reason: {signal.reason}")
    else:
        print(f"\n⏸️  HOLD - {signal.reason}")
    
    print(f"{'='*70}\n")




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


def check_and_settle_games(engine, tracker):
    """Check if any games have finished and settle positions using live data."""
    settled_any = False
    
    # Build a map of game_id -> live_data for active games
    live_games = {}
    for match in tracker.active_matches:
        game = match['nba_game']
        game_id = game['gameId']
        
        try:
            live_data = tracker.orch.live_client.get_live_game_data(game_id)
            if live_data:
                live_games[game_id] = {
                    'home_score': live_data['homeTeam']['score'],
                    'away_score': live_data['awayTeam']['score'],
                    'period': live_data.get('period', 0),
                    'clock': live_data.get('gameClock', '')
                }
        except:
            pass
    
    # Check each open position
    for game_id in list(engine.open_positions.keys()):
        if game_id not in live_games:
            continue
            
        game_data = live_games[game_id]
        
        # Check if game is truly over: Q4/OT ended and score is not tied
        is_regulation_over = game_data['period'] >= 4 and 'PT0M0' in game_data['clock']
        is_not_tied = game_data['home_score'] != game_data['away_score']
        
        if is_regulation_over and is_not_tied:
            # Determine winner
            home_won = game_data['home_score'] > game_data['away_score']
            
            # Settle position
            success, message, pnl = engine.close_position(game_id, home_won)
            
            if success:
                print(f"\n{'='*80}")
                print(f"🏁 GAME SETTLED: {game_id}")
                print(f"Final Score: {game_data['home_score']}-{game_data['away_score']}")
                print(f"Outcome: {'HOME WIN' if home_won else 'AWAY WIN'}")
                print(message)
                print(f"{'='*80}")
                settled_any = True
    
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
            settled_any = check_and_settle_games(engine, tracker)
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
                
                # Log prediction to database
                from src.data.database import OddsHistory
                from sqlalchemy.orm import sessionmaker
                Session = sessionmaker(bind=db.engine)
                session = Session()
                try:
                    record = OddsHistory(
                        game_id=game_id,
                        timestamp=datetime.utcnow(),
                        model_home_win_prob=float(pred['model_result']['probability']),
                        kalshi_home_win_prob=float(pred['kalshi_prob']),
                        kalshi_market_ticker=ticker,
                        kalshi_yes_price=int(pred['yes_ask']),  # Log ask prices (buy prices)
                        kalshi_no_price=int(pred['no_ask']),
                        home_team_id=game['homeTeam']['teamId'],
                        away_team_id=game['awayTeam']['teamId']
                    )
                    session.add(record)
                    session.commit()
                except Exception as e:
                    print(f"Warning: Failed to log prediction to DB: {e}")
                    session.rollback()
                finally:
                    session.close()
                
                # Get current position if any
                current_pos = engine.open_positions.get(game_id)
                
                # Evaluate Market - use ASK prices (what we'd pay to buy)
                signal = strategy.evaluate_market(
                    game_id=game_id,
                    model_result=pred['model_result'],
                    market_price_yes=pred['yes_ask'],  # Use ASK for buying
                    market_price_no=pred['no_ask'],    # Use ASK for buying
                    bankroll=engine.balance + engine.get_portfolio_summary()['total_exposure'],
                    current_position=current_pos
                )
                
                # Display signal
                display_trade_signal(game, signal, pred)
                
                # Execute Trade if needed
                if signal.action != 'HOLD':
                    # Determine correct price based on action
                    if signal.action == 'BUY':
                        # Pay ASK price when buying
                        execution_price = pred['yes_ask'] if signal.side == 'YES' else pred['no_ask']
                    else:  # SELL
                        # Receive BID price when selling
                        execution_price = pred['yes_bid'] if signal.side == 'YES' else pred['no_bid']
                    
                    success, msg = engine.execute_trade(
                        game_id=game_id,
                        ticker=ticker,
                        action=signal.action,
                        contracts=signal.contracts,
                        price=execution_price,  # Use correct bid/ask
                        side=signal.side,
                        reason=signal.reason
                    )
                    
                    if success:
                        print(f"✓ {msg}")
                        engine.save_state()
                    else:
                        print(f"✗ Failed: {msg}")
            
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
