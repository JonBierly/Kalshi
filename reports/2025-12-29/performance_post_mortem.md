# Post-Mortem: Trading Performance Analysis (2025-12-29)

## Executive Summary
On 2025-12-29, the NBA spread model realized a total loss of **-$47.16** ($52.14 gross). The primary driver of this loss was **extreme overconfidence in 1st-quarter leads**, particularly in the **GSW @ BKN** and **DEN @ MIA** games. While the model had access to time-remaining and team-stat features, structural bottlenecks in the feature engineering and model architecture prevented it from correctly pricing early-game risk.

---

## 1. Technical Root Causes

### **A. The "Ghost Roster" Mapping Bug (RosterEngine)**
Even though rosters are fetched live via the NBA API, the statistical lookup for those players had a critical dependency on the database state.
*   **The Bug**: `RosterEngine` identifies a player's team based on their *last recorded game in the database*.
*   **The Impact**: Players who changed teams during the offseason (e.g., Buddy Hield to GSW) were mapped to their old teams in the database. When the live engine tried to "project the GSW roster," it filtered for players whose DB `team_id` was GSW. 
*   **The Failure**: These players were **silently excluded** from their current team's projection. This caused the model to see GSW as significantly weaker than BKN, justifying 60-70% confidence in BKN's early lead when the true probability was closer to 50%.

### **B. Variance Underestimation (The Linear Model Ceiling)**
Score volatility in basketball follows a square-root curve ($\sigma \propto \sqrt{t}$), but the current `Ridge` model is linear.
*   **The Failure**: A linear model tries to fit a straight line to this curve. This resulted in a measured **~40% underestimation of variance** in the first quarter (Predicting ~8.5 points of $\sigma$ when actual game movement was ~14.9).
*   **The Result**: The model consistently output "tight" 90% confidence intervals early in the game, making it trade heavily on leads that were naturally volatile.

### **C. Lead "Stickiness" & The Weighting Trap**
To improve crunch-time performance, the model is trained with **10x weighting** for the final 2 minutes.
*   **The Conflict**: Linear models use a single set of coefficients for the entire game. By optimizing for the 10x weighted late-game data (where a 10-point lead is practically permanent), the model's weights became "sticky."
*   **The Outcome**: The model applied "Crunch Time Logic" to the "First Quarter." It ignored the `required_catchup_rate`'s implication for mean-reversion because its primary optimization goal was to be correct in the 4th quarter.

---

## 2. Evidence from Data Analysis
Testing the ensemble on the historical test set revealed the following bias:

| Game Phase | Measured MAE | Model Predicted $\sigma$ | Actual Game Volatility |
| :--- | :--- | :--- | :--- |
| **Q1 (>36m)** | **11.28 pts** | **~8.5 pts** | **~14.88 pts** |
| Q2 (24-36m) | 9.70 pts | ~8.0 pts | 12.77 pts |
| Q4 (2-12m) | 4.99 pts | ~6.0 pts | 6.48 pts |
| Clutch (<2m) | 2.52 pts | ~4.0 pts | 2.83 pts |

**Observation**: The model is well-calibrated in the 4th quarter but dangerously overconfident (underestimating risk by nearly half) in the 1st quarter.

---

## 3. Recommended Structural Fixes

### **Immediate Code Improvements**
1.  **Roster Logic Overhaul**: Modify `RosterEngine.get_projected_roster_features` to perform stat lookups by `player_id` only, removing the requirement that the database's `team_id` match the current team.
2.  **Interaction Features**: Implement `score_diff * (seconds_remaining / 2880)` as a base feature. This forces the model to treat the value of a point differently based on the clock.
3.  **Non-Linear Uncertainty**: Transition the `variance_model` to an XGBoost regressor or a log-transformed Ridge model to better capture the square-root nature of scoring risk.

### **Operational Improvements**
*   **Pre-Flight Backfill**: Ensure `etl_pipeline` and `backfill_stats` are run every morning to update player-team mappings and season-to-date averages before live trading begins.
