# Experiments to Test Spread Betting Opportunities

## 3 Quick Experiments

### 1. **Historical Backtest** (SIMULATED)
See if edges would have appeared in past games.

```bash
python -m spread_src.scripts.backtest_spreads
```

**What it does:**
- Uses test set (180 games)
- Simulates market prices (efficient but noisy)
- Counts betting opportunities (edge > 5%)
- Calculates profit/loss

**Output:**
- Opportunities per game
- Win rate
- Expected profit
- ROI

**Limitation:** Market prices are SIMULATED (not real)

---

### 2. **Live Market Check** (REAL MARKETS)
Check actual Kalshi spread markets RIGHT NOW.

```bash
python -m spread_src.scripts.live_market_check
```

**What it does:**
- Connects to Kalshi API
- Fetches today's NBA spread markets
- Shows current prices
- Detects arbitrage opportunities

**Output:**
- List of available games
- Spread markets for each
- Arbitrage alerts (if any)

**Note:** Only works if there are NBA games today!

---

### 3. **Edge Distribution Analysis**
Understand HOW OFTEN and HOW BIG edges are.

```bash
python -m spread_src.scripts.analyze_edges
```

**What it does:**
- Analyzes all test set events
- Shows edge frequency at different thresholds
- Breaks down by game state (close vs blowout, early vs late)
- Estimates ROI

**Output:**
```
Edge > +10%:  15.2% of events
Edge > +5%:   28.7% of events
Edge > +3%:   42.1% of events

Close Games: 32% of events have edge > 5%
Blowouts:    18% of events have edge > 5%

Expected ROI: 12.3% per $1 bet
```

---

## Recommended Order

### Step 1: Edge Analysis
```bash
python -m spread_src.scripts.analyze_edges
```

**Goal:** Understand if edges exist at all  
**Time:** ~30 seconds

---

### Step 2: Live Market Check
```bash
python -m spread_src.scripts.live_market_check
```

**Goal:** See if Kalshi has spread markets today  
**Time:** ~5 seconds

**If no games:** Come back during an NBA game day!

---

### Step 3: Historical Backtest
```bash
python -m spread_src.scripts.backtest_spreads
```

**Goal:** Simulate full betting strategy  
**Time:** ~1 minute

---

## Interpreting Results

### Good Signs ✅
- Edge > 5% appears frequently (>20% of events)
- Win rate > 53% on trades
- Positive ROI
- Arbitrage opportunities exist

### Bad Signs ❌
- Edge > 5% is rare (<5% of events)
- Win rate < 50%
- Negative ROI
- No spread markets available

---

## What's Next?

### If Edges Exist:
1. **Today:** Run live market check to find real arbitrage
2. **This week:** Build spread tracker for live games
3. **Next week:** Paper trade spreads vs binary

### If No Edges:
1. Check if Kalshi spread markets are liquid
2. Consider hybrid strategy (binary + spreads)
3. Focus on binary strategy improvements

---

## Files Created

```
spread_src/scripts/
├── backtest_spreads.py      ✓ Historical simulation
├── live_market_check.py     ✓ Real Kalshi markets
├── analyze_edges.py         ✓ Edge distribution
└── EXPERIMENTS.md           ✓ This guide
```

---

**Start here:**
```bash
python -m spread_src.scripts.analyze_edges
```

This will tell you if pursuing spread betting makes sense!
