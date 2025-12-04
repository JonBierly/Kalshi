#!/usr/bin/env python
"""
Track spread markets live - similar to paper_trade.py but for spreads.

Shows real-time comparison of:
- Model spread predictions
- Kalshi spread market prices
- Edges and opportunities
- Arbitrage opportunities
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spread_src.inference.spread_tracker import SpreadTracker


def main():
    """Run the spread tracker."""
    print("=" * 80)
    print("SPREAD MARKET TRACKER")
    print("=" * 80)
    
    # Get Kalshi credentials from environment
    kalshi_key_id = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
    kalshi_key_path = "key.key"
    
    if not kalshi_key_id:
        print("\n✗ Missing Kalshi API credentials")
        print("\nPlease set environment variables:")
        print("  export KALSHI_API_KEY_ID='your_key_id'")
        print("  export KALSHI_PRIVATE_KEY_PATH='path/to/key.key'")
        print("\nOr check scripts/paper_trade.py to see how you have it set up there.")
        return
    
    # Initialize tracker
    print(f"\nInitializing with key: {kalshi_key_id[:8]}...")
    tracker = SpreadTracker(kalshi_key_id, kalshi_key_path)
    
    # Setup (match games to markets)
    print("\nMatching NBA games to Kalshi spread markets...")
    tracker.setup()
    
    if not tracker.active_matches:
        print("\n⚠️  No games matched to spread markets")
        print("\nPossible reasons:")
        print("  1. No NBA games today")
        print("  2. Kalshi doesn't have spread markets for today's games")
        print("  3. Games haven't started yet")
        print("\nTry again during live NBA games!")
        return
    
    print(f"\n✓ Tracking {len(tracker.active_matches)} game(s) with spread markets\n")
    
    # Run tracking loop
    tracker.run_loop(interval=30)  # Update every 30 seconds


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
