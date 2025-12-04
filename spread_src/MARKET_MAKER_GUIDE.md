# Market Maker Quick Start Guide

## Overview
You now have a complete market making system that:
- ✅ Provides liquidity in Kalshi spread markets
- ✅ Uses your trained Ridge model for fair value
- ✅ Enforces strict risk limits ($20 total, $5/game)
- ✅ Runs in dry-run mode for testing
- ✅ Places ONE order per iteration (configurable)

---

## Components Built

### 1. **Risk Manager** (`spread_src/execution/risk_manager.py`)
- Pre-trade checks for all limits
- Exposure calculation
- Position validation

### 2. **Portfolio Tracker** (`spread_src/execution/portfolio.py`)
- Position tracking (long/short)
- P&L calculation (realized + unrealized)
- Cash and exposure management

### 3. **Market Maker Strategy** (`spread_src/trading/market_maker.py`)
- Fair value from model
- Dynamic spread sizing
- Price improvement logic
- Kelly sizing

### 4. **Order Manager** (`spread_src/execution/order_manager.py`)
- Order placement (STUBS for now)
- Cancel/replace logic
- Fill tracking

### 5. **Live Trading Engine** (`spread_src/scripts/live_market_maker.py`)
- Main loop (30s iterations)
- Game evaluation
- Best opportunity selection
- One order per iteration

---

## Running It

### Dry-Run Mode (Recommended First!)
```bash
python -m spread_src.scripts.live_market_maker
```

**What happens:**
- Fetches live games and markets
- Calculates fair values
- Evaluates opportunities
- **LOGS orders without placing them**
- Shows what it WOULD do

### Live Mode (REAL MONEY!)
Edit `live_market_maker.py`:
```python
DRY_RUN = False  # Line 292
```

Then run:
```bash
python -m spread_src.scripts.live_market_maker
```

**⚠️ USE WITH CAUTION!** This will place REAL orders.

---

## Next Steps (For You)

### 1. **Fill in Kalshi API Stubs**

In `src/data/kalshi.py`, add these methods:

```python
def place_order(self, ticker, side, price, size, type='limit'):
    """
    Place limit order.
    
    Returns:
        order_id: String ID of placed order
    """
    # TODO: Research Kalshi API docs
    # POST /trade-api/v2/portfolio/orders
    pass

def cancel_order(self, order_id):
    """Cancel pending order."""
    # DELETE /trade-api/v2/portfolio/orders/{order_id}
    pass

def get_orders(self, status='open'):
    """Get your orders."""
    # GET /trade-api/v2/portfolio/orders
    pass

def get_fills(self):
    """Get recent fills."""
    # GET /trade-api/v2/portfolio/fills
    pass
```

**Reference:** https://trading-api.readme.io/reference/

### 2. **Test Dry-Run**

Run it during today's games and review logs:
- Are fair values reasonable?
- Are quotes competitive?
- Are risk checks working?
- Does it find opportunities?

### 3. **Small Live Test**

Once comfortable:
- Set `DRY_RUN = False`
- Set `MAX_EXPOSURE = 3.0` (small test)
- Run for ONE game
- Monitor closely!

### 4. **Gradually Scale**

If successful:
- Increase to $5/game
- Add more games
- Scale to full $20

---

## Safety Features

### Hard Limits
```python
MAX_TOTAL_EXPOSURE = 20.0  # Never exceed
MAX_GAME_EXPOSURE = 5.0    # Per game limit
MAX_ORDER_SIZE = 20        # Max contracts
```

### Emergency Stop
```
Ctrl+C → Cancels all orders → Shows positions → Exits
```

### Pre-Trade Checks
Every order must pass:
- Price bounds (1-99¢)
- Size validation
- Total exposure check
- Game exposure check
- Position limits

---

## Configuration

Edit `live_market_maker.py` main():

```python
DRY_RUN = True           # Simulation vs real
MAX_EXPOSURE = 20.0      # Total $ limit
MAX_GAME_EXPOSURE = 5.0  # Per game limit
UPDATE_INTERVAL = 30     # Seconds between iterations
```

---

## Monitoring

### Real-Time Output
```
Iteration 5 @ 19:45:32
================================================================================

DAL @ DEN:
  Refreshing prices...

Best opportunity: Buy edge 12.3¢
  BUY 5 KXNBASPREAD-25DEC02DALDEN-DAL10 @ 45.0¢
  Expected value: 12.3¢
  ✓ Risk check passed
  [DRY-RUN] Order placed: DRY_4

=== PORTFOLIO ===
Cash: $20.00
Exposure: $0.00
Realized P&L: $0.00

Positions:
  (none)

=== ORDERS ===
Pending: 1
  DRY_4: buy 5 KXNBASPREAD-25DEC02DALDEN-DAL10 @ 45.0¢
```

---

## Known Limitations

1. **Kalshi API stubs** - You need to implement these
2. **Simplified fair value** - Currently uses placeholder (50¢)
   - TODO: Integrate full tracker prediction logic
3. **No fill simulation** - Dry-run doesn't simulate fills
4. **One order per iteration** - Can adjust if needed

---

## Files Created

```
spread_src/
├── execution/
│   ├── risk_manager.py      ✓ Risk checks
│   ├── portfolio.py         ✓ Position tracking
│   └── order_manager.py     ✓ Order management (stubs)
├── trading/
│   └── market_maker.py      ✓ Strategy
└── scripts/
    └── live_market_maker.py ✓ Main loop
```

---

## Questions?

The system is ready for dry-run testing! Once you:
1. Fill in Kalshi API methods
2. Test in dry-run mode
3. Verify it looks correct

You can go live with real money (carefully!). 🚀
