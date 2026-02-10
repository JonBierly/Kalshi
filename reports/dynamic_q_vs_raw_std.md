# Why Time-Conditional $q$ is better than Global $q$

This document answers the core question: **"If the model's predicted standard deviation ($\hat{\sigma}$) already changes as the game progresses, why do we need a time-based multiplier $q(t)$?"**


## 1. The "Shrinking Uncertainty" Trap

You are 100% correct that the model's predicted standard deviation ($\hat{\sigma}$) is dynamic. It starts high (around 12-15 points) and shrinks toward zero as the game ends.

However, our research into the **S-Score** ($|error| / \hat{\sigma}$) revealed a hidden problem:

| Game Phase | Model's $\hat{\sigma}$ | Actual Typical Error | S-Score ($error/\hat{\sigma}$) | Model State |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Tip-off)** | 14.0 | 11.2 | **0.8** | Under-confident (Safe) |
| **Q4 (Mid)** | 4.0 | 4.8 | **1.2** | Slightly Over-confident |
| **Q4 (2 min left)** | 1.5 | 3.3 | **2.2** | **Bravely Wrong** |

The model's $\hat{\sigma}$ is shrinking **faster** than the actual score volatility is. In the final minutes, the model thinks the game is "over" and narrows its uncertainty to a tiny sliver (e.g., 1.5 points), but the "intentional fouling game" means the score can still swing by 4-5 points.

## 2. Why Global $q$ fails both ends

If we used a single global multiplier (e.g., the average $q = 1.6$):

1.  **Early Game**: We take the model's already safe 14.0 $\hat{\sigma}$ and multiply it by 1.6 $\to$ **22.4**. The model is now *massively* over-conservative. You will never find an edge because the "safety bars" are so wide they cover the entire stadium.
2.  **Late Game**: We take the model's 1.5 $\hat{\sigma}$ and multiply it by 1.6 $\to$ **2.4**. But the historical data says we actually need an interval of **3.3+** to be 90% sure. We are still under-protected and will lose money on "bad beats."

## 3. The Solution: $q(t)$ as a "Correction Curve"

The multiplier $q(t)$ is not a substitute for standard deviation; it is a **variable lens** that corrects the model's specific biases at specific times.

- At **$t=48$**: $q \approx 0.8$. It says: "Model, you are being too scared, I'm going to tighten your bars slightly so we can actually trade."
- At **$t=2$**: $q \approx 2.2$. It says: "Model, I know you think this game is over, but history shows things get wild here. I'm going to double your uncertainty to protect our capital."

**Conclusion**: We use `score/time` because the **error in the model's uncertainty guess** is tightly correlated with the game clock. Using time as the "anchor" for calibration allows us to be aggressive when the model is shy, and defensive when the model is over-confident.
