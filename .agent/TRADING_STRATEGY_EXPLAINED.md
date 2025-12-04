# Trading Strategy Breakdown

## Overview
Your trading system uses a **Kelly Criterion-based quantitative strategy** with active position management and strict risk controls.

---

## Decision Flow Chart

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEW MARKET OPPORTUNITY                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ① PRE-GAME FILTER (NEW!)                                       │
│   • gameStatus must be 2 (in-progress)                         │
│   • period must be >= 1 (game started)                         │
│   • Stats must exist                                           │
│   ❌ REJECT if pre-game → Skip entirely                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ② HARD CUTOFF CHECK                                            │
│   • If seconds_remaining < 120 (2 minutes):                    │
│     - SELL ALL positions immediately                           │
│     - BLOCK all new buys                                       │
│   Rationale: Market too volatile, data latency too high        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ③ MERCY RULE (Impossible Comebacks)                            │
│   • Calculate required_catchup_rate = |score_diff| / time      │
│   • If catchup_rate > 0.3 points/second:                       │
│     - Override model prob to 99.9% (leader) or 0.1% (trailer)  │
│   Rationale: Prevent "buying the dip" on guaranteed losses     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ④ CONFIDENCE CHECK                                             │
│   • Calculate CI width = ci_95_upper - ci_95_lower             │
│   • If CI width > 25%:                                         │
│     ❌ REJECT → Model too uncertain                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑤ EDGE CALCULATION                                             │
│   • YES Edge = model_prob - (market_yes_price / 100)           │
│   • NO Edge = (1 - model_prob) - (market_no_price / 100)       │
│   • If both edges < 3%:                                        │
│     ❌ REJECT → No profitable opportunity                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑥ KELLY POSITION SIZING                                        │
│   • Calculate Kelly fraction: f = (bp - q) / b                 │
│     where b = (1 - odds) / odds                                │
│   • Apply fractional Kelly (currently 100% - AGGRESSIVE!)      │
│   • Cap at 5% max of bankroll per game                         │
│   → target_contracts = (bankroll * kelly%) / contract_price    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑦ POSITION REBALANCING                                         │
│   • Compare current_contracts vs target_contracts              │
│   • If sides differ (YES ↔ NO):                                │
│     → SELL ALL first, then buy new side next iteration         │
│   • If difference < minimum threshold:                         │
│     → HOLD (avoid overtrading)                                 │
│   • Otherwise:                                                 │
│     → BUY or SELL to reach target                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑧ EXECUTION                                                    │
│   • BUY: Pay ASK price (what sellers want)                     │
│   • SELL: Receive BID price (what buyers offer)                │
│   • Log to database                                            │
│   • Update portfolio                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components Explained

### 1. **Edge Detection**
Your strategy identifies profitable opportunities by comparing model probability vs market price.

**Formula:**
```python
YES_edge = model_prob - (market_yes_price / 100)
NO_edge = (1 - model_prob) - (market_no_price / 100)
```

**Example:**
- Model says: 65% home win probability
- Kalshi YES price: 55¢
- **YES Edge = 0.65 - 0.55 = +10%** ← Good bet!

**Requirement:** Edge must be > 3% to trade.

---

### 2. **Kelly Criterion Position Sizing**
Determines optimal bet size based on edge and bankroll.

**Formula:**
```python
f = (bp - q) / b
where:
  b = net odds = (1 - price) / price
  p = win probability (from model)
  q = loss probability = 1 - p
```

**Example:**
- Model prob: 60%
- Market price: 50¢ (0.50)
- b = (1 - 0.50) / 0.50 = 1.0
- f = (1.0 × 0.60 - 0.40) / 1.0 = 0.20 (20% of bankroll)

**Your Settings:**
- Fractional Kelly: **100%** (very aggressive, consider 25-50%)
- Max per position: **5%** of bankroll
- Effective bet size: `min(kelly%, 5%)`

---

### 3. **Active Position Management**
Unlike "buy and hold", your strategy **rebalances constantly** as:
- Model probability changes
- Market odds shift
- Time remaining decreases

**Example Flow:**
```
t=10min: Model 60%, target 10 contracts YES
  → BUY 10 YES @ 55¢

t=8min: Model 70%, target 15 contracts YES
  → BUY 5 more YES @ 58¢

t=5min: Model 55%, target 8 contracts YES
  → SELL 7 YES @ 60¢ (take profit)

t=2min: Hard cutoff triggered
  → SELL ALL 8 YES @ 62¢ (exit)
```

---

### 4. **Risk Controls**

| Control | Value | Rationale |
|---------|-------|-----------|
| **Minimum Edge** | 3% | Avoid marginal bets, account for bid-ask spread |
| **Max CI Width** | 25% | Don't bet when model is uncertain |
| **Max Position Size** | 5% of bankroll | Prevent ruin from single loss |
| **Max Total Exposure** | 30% of bankroll | Protection across portfolio |
| **Hard Cutoff** | 120 seconds | Exit before data becomes unreliable |
| **Mercy Rule** | 0.3 pts/sec catchup | Don't fight the impossible |
| **Rebalance Threshold** | 2 contracts | Avoid overtrading on small changes |

---

### 5. **Execution Pricing**

**CRITICAL:** You trade at different prices depending on action:

| Action | Price Used | Why |
|--------|-----------|-----|
| **BUY** | **ASK** price | You pay what sellers demand |
| **SELL** | **BID** price | You receive what buyers offer |

**Example:**
- YES BID: 58¢ (what buyers will pay you)
- YES ASK: 60¢ (what sellers charge you)
- **Bid-Ask Spread: 2¢** ← This is "slippage"

When you buy at 60¢ and immediately sell at 58¢, you lose 2¢ per contract just from spread!

---

## Strategy Strengths ✅

1. **Quantitative & Disciplined**: No emotional decisions
2. **Dynamic**: Adapts to changing probabilities in real-time
3. **Risk-Managed**: Multiple layers of protection
4. **Data-Driven**: Uses advanced ML model for edge
5. **Pre-Game Filter**: Now only trades live games!

---

## Potential Improvements 🔧

### ⚠️ **Fractional Kelly is TOO AGGRESSIVE**
- Current: **100% Kelly** (full Kelly)
- Recommended: **25-50% Kelly** (half Kelly is industry standard)
- Reason: Kelly assumes perfect probability estimates. Your model has error!

### 💡 **Consider Bid-Ask Spread Filter**
- Some markets have 5-10¢ spreads, eating your edge
- Suggestion: `if (ask - bid) > 5¢: skip trade`

### 📊 **Track Win Rate by Time Remaining**
- Your model might be better at Q1 vs Q4
- Could adjust confidence thresholds based on period

### 🎯 **Kelly Assumes Independent Bets**
- You're trading correlated events (same game over time)
- True Kelly fraction should be lower for correlated bets

---

## Example Trade Walkthrough

**Scenario:** Warriors vs Lakers, Q3, 5:30 remaining

```
Live Data:
  • Score: GSW 78 - LAL 75 (GSW +3)
  • Time: 330 seconds total remaining
  • gameStatus: 2 (in-progress) ✅
  • period: 3 ✅

Model Output:
  • GSW win prob: 68%
  • Confidence: [58%, 78%] → CI width = 20% ✅

Market Prices:
  • YES (GSW) BID: 60¢, ASK: 62¢
  • NO (LAL) BID: 38¢, ASK: 40¢

Edge Calculation:
  • YES edge = 0.68 - 0.62 = +6% ✅ (exceeds 3% minimum)
  • NO edge = 0.32 - 0.40 = -8% ❌

Kelly Sizing:
  • b = (1 - 0.62) / 0.62 = 0.613
  • f = (0.613 × 0.68 - 0.32) / 0.613 = 0.159 (15.9%)
  • Bankroll: $1000
  • Target: $1000 × 15.9% = $159 / $0.62 = 256 contracts
  • Capped at 5%: min(256, 80) = 80 contracts

Decision:
  → BUY 80 YES @ 62¢ = $49.60
  → Expected Value: 80 × 0.06 = +$4.80
```

**What happens next iteration (2 minutes later)?**

If GSW extends lead to +7:
  - Model prob → 78%
  - YES edge → +16%
  - Target contracts → 120
  - **Action: BUY 40 more**

If LAL ties game:
  - Model prob → 52%
  - YES edge → +2% (below 3% threshold)
  - Target contracts → 0
  - **Action: SELL ALL 80**

---

## Questions to Consider

1. **Is 100% Kelly too risky?** Industry uses 25-50%.
2. **Are you accounting for bid-ask spread?** 2-5¢ eats your edge.
3. **Should you trade more aggressively in Q1-Q2?** Model might be more reliable early.
4. **Does "mercy rule" trigger too late?** Maybe 0.25 pts/sec instead of 0.3?
5. **Should you have different cutoffs per quarter?** Q4 < 2min vs Q1 < 5min?

---

## Configuration Values (Quick Reference)

```python
# In strategy.py → RiskManager()
max_position_pct = 0.05      # 5% max per game
max_total_exposure_pct = 0.30  # 30% total
min_edge = 0.03              # 3% minimum edge
max_ci_width = 0.25          # 25% max CI width
min_contracts = 1            # Minimum trade size
rebalance_threshold = 2      # Min diff to rebalance

# In strategy.py → TradingStrategy()
fractional_kelly = 1.0       # 100% Kelly (AGGRESSIVE!)

# In strategy.py → evaluate_market()
hard_cutoff_seconds = 120    # 2 minutes
mercy_rule_catchup = 0.3     # 0.3 pts/second
```

---

**Last Updated:** 2025-12-01
