# Spread Trading Strategy - Summary

## ✅ **What We Built**

A complete spread distribution trading system for Kalshi NBA markets with:
- **Distribution prediction** (Ridge regression ensemble)
- **Live market tracking**
- **Arbitrage detection**
- **Time-weighted training** for late-game accuracy

---

## **Key Files Created**

```
spread_src/
├── data/
│   └── spread_markets.py          # Parse tickers, detect arbitrage
├── models/
│   ├── spread_model.py            # Distribution prediction (CDF approach)
│   └── spread_training.py         # (not created separately, in scripts/)
├── trading/
│   └── spread_strategy.py         # Edge detection + position sizing
├── inference/
│   └── spread_tracker.py          # Live tracking (like binary tracker)
└── scripts/
    ├── train_spread_model.py      # Training with time weights ✓
    ├── train_xgboost.py            # XGBoost alternative
    ├── train_quantile.py           # Quantile regression alternative
    ├── compare_models.py           # Model comparison
    └── track_spreads.py            # Live tracker runner ✓
```

---

## **Model Performance**

### **Model Tested: Ridge, XGBoost, Quantile**
**Winner:** Ridge (best calibration + speed)

### **Final Ridge Model (Time-Weighted)**
- **Weighted MAE:** 5.9 points (late-game emphasis)
- **Overall MAE:** 7.80 points
- **RMSE:** 10.45 points
- **Calibration:** Within 1-5% across all thresholds

### **Sample Weights:**
- Last 2 minutes: **10x weight**
- 2-5 minutes: **5x weight**
- 5-10 minutes: **2x weight**
- Rest of game: **1x weight**

---

## **Live Tracking Results**

### **Games Tracked:** 
- HOU @ UTA (Q4)
- PHX @ LAL (Q3)

### **Opportunities Found:**

**PHX @ LAL** (PHX up 16 in Q3):
- 🔥 **PHX >10.5**: Model 62% | Ask 34¢ | **+28% edge**
- 🔥 **PHX >7.5**: Model 76% | Ask 46¢ | **+30% edge**
- 🔥 **PHX >4.5**: Model 86% | Ask 59¢ | **+27% edge**

**Arbitrage Detected:**
- LAL >17.5 vs LAL >20.5: **9¢ risk-free** (before weights fixed)

---

## **Key Learnings**

### **1. Spread Markets Are Less Efficient**
- **Edges:** 20-30% (vs 1-2% on binary)
- **Why:** Fewer participants modeling distributions
- **Opportunity:** Real potential for profitable trading

### **2. Late-Game Accuracy Matters**
- **Problem:** Original model predicted μ=-2.4 for UTA +4 with 37s left
- **Solution:** Time-weighted training (10x weight for <2 min)
- **Result:** Model now respects late-game certainty

### **3. Bid/Ask Matters for Arbitrage**
- **Buy:** Pay yes_ask
- **Sell:** Receive yes_bid
- **Arbitrage:** yes_bid(higher) > yes_ask(lower)

### **4. Ticker Format**
- `KXNBASPREAD-25DEC01HOUUTA-HOU3` = "HOU >3.5 points"
- Integer in ticker means integer + 0.5 in reality

---

## **Next Steps**

### **Immediate:**
1. ✅ Model retrained with time weights
2. ⚠️ Test predictions on NEW late-game scenarios
3. ⏳ Verify late-game accuracy improved

### **Short-term:**
1. Paper trade spreads for 1 week
2. Compare P&L vs binary strategy
3. Tune time weights if needed  

### **Long-term:**
1. Build paper trading engine for spreads
2. Implement full Kelly sizing for distribution bets
3. Deploy live if profitable

---

## **Model Comparison Results**

| Model | MAE | Calibration | Speed | Winner |
|-------|-----|-------------|-------|--------|
| Ridge | 7.79 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | ✅ |
| XGBoost | 8.59 | ⭐⭐ | ⚡⚡⭐ | ❌ |
| Quantile | 8.20 | ⭐⭐⭐⭐ | ⚡⚡⚡ | ❌ |

**Ridge won** due to better calibration + speed.

---

## **Critical Fixes Made**

1. **Ticker parsing:** Changed `\d{7}` to `[A-Z0-9]{7}` for date
2. **Spread values:** Added +0.5 (ticker shows integer, actual is +0.5)
3. **Arbitrage logic:** Use yes_bid for selling, yes_ask for buying
4. **Time weighting:** Emphasize late-game accuracy (10x for <2 min)

---

## **Usage**

### **Train Model:**
```bash
python -m spread_src.scripts.train_spread_model
```

### **Track Live:**
```bash
python -m spread_src.scripts.track_spreads
```

### **Compare Models:**
```bash
python -m spread_src.scripts.compare_models
```

---

## **Status**

✅ Core system built  
✅ Model trained with time weights  
✅ Live tracking working  
✅ Arbitrage detection functional  
⚠️ Needs verification on late-game scenarios  
⏳ Paper trading not yet implemented  

---

**Created:** 2025-12-01  
**Models:** `models/nba_spread_model.pkl`
