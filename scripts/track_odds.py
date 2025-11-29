import os
import sys
from src.inference.tracker import OddsTracker

def main(model_type='lr', interval=10):
    """
    Track NBA odds and model predictions.
    
    Args:
        model_type: 'lr', 'xgboost', or 'both' for model selection
        interval: Update interval in seconds (default: 10)
    """
    # Get credentials
    key_id = os.environ.get('KALSHI_KEY_ID', "a40ff1c6-12ac-4a6c-9669-ffe12f3de235")
    
    if model_type == 'both':
        print("=" * 80)
        print("Starting odds tracker with BOTH models for comparison...")
        print("=" * 80)
        print()
        
        try:
            # Create two trackers
            tracker_lr = OddsTracker(key_id, model_type='lr')
            tracker_xgb = OddsTracker(key_id, model_type='xgboost')
            
            # Setup both
            print("Setting up LR tracker...")
            tracker_lr.setup()
            
            print("\nSetting up XGBoost tracker...")
            tracker_xgb.setup()
            
            # Run comparison loop
            run_comparison_loop(tracker_lr, tracker_xgb, interval)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Starting odds tracker with {model_type.upper()} model...")
        
        try:
            tracker = OddsTracker(key_id, model_type=model_type)
            tracker.setup()
            tracker.run_loop(interval=interval)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def run_comparison_loop(tracker_lr, tracker_xgb, interval=10):
    """Run tracking loop comparing both models."""
    import time
    from datetime import datetime
    
    print("\nStarting comparison tracking loop...")
    session_lr = tracker_lr.Session()
    session_xgb = tracker_xgb.Session()
    
    try:
        while True:
            print("\n" + "=" * 80)
            print(f"Update at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            
            if not tracker_lr.active_matches:
                print("No active games matched.")
                time.sleep(interval)
                continue
            
            for match in tracker_lr.active_matches:
                game = match['nba_game']
                ticker = match['kalshi_ticker']
                
                # Get predictions from both models
                try:
                    # LR prediction
                    lr_pred = get_model_prediction(tracker_lr, game, ticker)
                    
                    # XGBoost prediction  
                    xgb_pred = get_model_prediction(tracker_xgb, game, ticker)
                    
                    if lr_pred and xgb_pred:
                        display_comparison(game, ticker, lr_pred, xgb_pred)
                        
                        # Log both to database
                        log_prediction(session_lr, game, ticker, lr_pred, 'lr')
                        log_prediction(session_xgb, game, ticker, xgb_pred, 'xgb')
                    
                except Exception as e:
                    print(f"Error processing {game.get('gameCode', 'unknown')}: {e}")
                    continue
            
            session_lr.commit()
            session_xgb.commit()
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopping comparison tracker...")
    finally:
        session_lr.close()
        session_xgb.close()


def get_model_prediction(tracker, game, ticker):
    """Get prediction from a specific tracker's model."""
    # Setup context if needed
    if tracker.orch.prediction_engine.current_game_id != game['gameId']:
        try:
            tracker.orch.setup_game_context(
                game['gameId'], 
                game['homeTeam']['teamId'], 
                game['awayTeam']['teamId']
            )
            tracker.orch.feature_engine.reset()
        except Exception as e:
            # Silently skip games we can't set up context for
            return None
    
    # Get live data
    try:
        live_data = tracker.orch.live_client.get_live_game_data(game['gameId'])
    except Exception as e:
        # Game hasn't started yet or API error - silently skip
        return None
    
    if not live_data:
        return None
    
    # Check if game has actually started (has stats)
    try:
        home_stats = live_data['homeTeam']['statistics']
        away_stats = live_data['awayTeam']['statistics']
        
        # If game hasn't started, stats might be empty or missing
        if not home_stats or not away_stats:
            return None
    except (KeyError, TypeError):
        # Game data not ready yet
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
    except (KeyError, TypeError) as e:
        # Stats not available yet
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
    except (KeyError, TypeError) as e:
        # Can't build features yet
        return None
    
    full_feats = {**live_features, **tracker.orch.prediction_engine.current_game_context}
    model_result = tracker.orch.prediction_engine.predict_with_confidence(full_feats)
    
    # Get Kalshi odds
    kalshi_prob = 0.0
    yes_price = 0
    no_price = 0
    
    try:
        market_book = tracker.kalshi.get_orderbook(ticker)
        if market_book:
            details = tracker.kalshi.get_market_details(ticker)
            if details:
                yes_price = details.get('yes_ask', 0)
                no_price = details.get('no_ask', 0)
                kalshi_prob = yes_price / 100.0
    except Exception:
        # Kalshi API issue, use 0 for market price
        pass
    
    clock_display = live_data.get('gameClock', '').replace('PT', '').replace('M', ':').replace('S', '')
    
    return {
        'live_data': live_data,
        'model_result': model_result,
        'kalshi_prob': kalshi_prob,
        'yes_price': yes_price,
        'no_price': no_price,
        'clock_display': clock_display,
        'period': period
    }


def display_comparison(game, ticker, lr_pred, xgb_pred):
    """Display formatted comparison of both models' predictions."""
    ld = lr_pred['live_data']
    home_score = ld['homeTeam']['score']
    away_score = ld['awayTeam']['score']
    period = lr_pred['period']
    clock = lr_pred['clock_display']
    
    home_team = game['homeTeam']['teamTricode']
    away_team = game['awayTeam']['teamTricode']
    
    lr_prob = lr_pred['model_result']['probability']
    xgb_prob = xgb_pred['model_result']['probability']
    kalshi_prob = lr_pred['kalshi_prob']
    
    # Calculate difference
    diff = lr_prob - xgb_prob
    diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
    
    # Determine agreement
    lr_pick = "HOME" if lr_prob > 0.5 else "AWAY"
    xgb_pick = "HOME" if xgb_prob > 0.5 else "AWAY"
    agree = "✓" if lr_pick == xgb_pick else "✗"
    
    print(f"\n┌─ {game['gameCode']} ─ {home_team} vs {away_team}")
    print(f"│  Score: {home_score}-{away_score} │ Q{period} {clock}")
    print(f"├─ Model Predictions:")
    print(f"│  LR:       {lr_prob:>6.1%}  ({lr_pick})")
    print(f"│  XGBoost:  {xgb_prob:>6.1%}  ({xgb_pick})")
    print(f"│  Diff:     {diff_str:>6s}  {agree} Models {'agree' if agree == '✓' else 'DISAGREE'}")
    print(f"├─ Market:")
    print(f"│  Kalshi:   {kalshi_prob:>6.1%}  (Ticker: {ticker})")
    print(f"└─ Confidence Intervals (LR | XGB):")
    print(f"   50%: {lr_pred['model_result']['ci_50_lower']:.1%}-{lr_pred['model_result']['ci_50_upper']:.1%} │ " +
          f"{xgb_pred['model_result']['ci_50_lower']:.1%}-{xgb_pred['model_result']['ci_50_upper']:.1%}")
    print(f"   95%: {lr_pred['model_result']['ci_95_lower']:.1%}-{lr_pred['model_result']['ci_95_upper']:.1%} │ " +
          f"{xgb_pred['model_result']['ci_95_lower']:.1%}-{xgb_pred['model_result']['ci_95_upper']:.1%}")


def log_prediction(session, game, ticker, pred, model_type):
    """Log prediction to database."""
    from src.data.database import OddsHistory
    from datetime import datetime
    
    record = OddsHistory(
        game_id=game['gameId'],
        timestamp=datetime.utcnow(),
        model_home_win_prob=float(pred['model_result']['probability']),
        kalshi_home_win_prob=float(pred['kalshi_prob']),
        kalshi_market_ticker=ticker,
        kalshi_yes_price=int(pred['yes_price']),
        kalshi_no_price=int(pred['no_price']),
        home_team_id=game['homeTeam']['teamId'],
        away_team_id=game['awayTeam']['teamId']
    )
    session.add(record)


if __name__ == "__main__":
    # Default: LR model
    # To use XGBoost: main(model_type='xgboost')
    # To compare both: main(model_type='both')
    main(model_type='both', interval=10)

