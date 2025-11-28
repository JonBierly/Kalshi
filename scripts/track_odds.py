import os
import sys
from src.inference.tracker import OddsTracker

def main():
    # Get credentials
    key_id = os.environ.get('KALSHI_KEY_ID', "a40ff1c6-12ac-4a6c-9669-ffe12f3de235")
    
    try:
        tracker = OddsTracker(key_id)
        tracker.setup()
        tracker.run_loop(interval=10)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
