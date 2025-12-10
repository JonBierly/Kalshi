
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.kalshi import KalshiClient
import uuid

API_KEY = "a40ff1c6-12ac-4a6c-9669-ffe12f3de235"
KEY_PATH = "key.key"

def main():
    print("🔌 Connecting to Kalshi...")
    client = KalshiClient(API_KEY, KEY_PATH)
    
    # Use provided ticker
    target_ticker = "KXNBASPREAD-25DEC10PHXOKC-OKC21"
    print(f"🎯 Target Ticker: {target_ticker}")
    
    # 1. Place SELL YES @ 99 (Should be "Sell YES")
    print("\n📝 Placing SELL YES @ 99¢ (Limit)...")
    order_data = client.place_order(
        ticker=target_ticker,
        side='yes',
        action='sell',
        count=1,
        price=99, # Sell high, unlikely to fill
        order_type='limit'
    )
    
    if not order_data:
        print("Failed to place order.")
        return

    order_id = order_data.get('order_id')
    print(f"  Order ID: {order_id}")
    
    # 2. Fetch Order Details
    print("\n🔍 Fetching Order Details...")
    orders = client.get_orders() # Fetch all/recent orders
    
    my_order = next((o for o in orders if o['order_id'] == order_id), None)
    
    if my_order:
        print(f"  API Reports: Side={my_order['side']}, Action={my_order['action']}, Price={my_order.get('yes_price') or my_order.get('no_price')}")
    else:
        print("  Order not found in pending list.")
        
    # 3. Cancel Order
    print("\n🗑️ Canceling Order...")
    client.cancel_order(order_id)
    print("  Done.")

if __name__ == "__main__":
    main()
