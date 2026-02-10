# Mitigating Tail Hallucination

## 1. The Problem: "Fattening the Tails"
When we increase a distribution's standard deviation ($\sigma$) to be "safe," we inadvertently increase the probability of extreme events.

**Example: A +15.5 Point Underdog Blowout**
- **Base Model**: Mean = 0, Std = 5. Probability of winning by >15.5 is **~0.1%**.
- **Widened Model**: Mean = 0, Std = 10. Probability of winning by >15.5 becomes **~6%**.

The trader sees the jump from **0.1% to 6%** as a massive "edge" and places an order, even though the model is only wider because we are *less sure*. Widening should make us *less* likely to trade, not more.

## 2. The Solution: Conservative Probability Bounds
Instead of widening the PDF, we should calculate a **Confidence Interval for the Probability ($P \pm \Delta P$)** and always trade on the **unfavorable bound**.

### New Logic:
1.  Calculate the **Raw Probability ($P_{raw}$)** from the ensemble.
2.  Calculate the **Uncertainty in that Probability ($\Delta P$)** based on ensemble variance and Conformal $q$.
3.  **Conservative Prob ($P_{cons}$)**:
    - If $P_{raw} > 0.5$: $P_{cons} = P_{raw} - \Delta P$ (Subtract uncertainty from the favorite)
    - If $P_{raw} < 0.5$: $P_{cons} = P_{raw} - \Delta P$ (Also subtract uncertainty from the long-shot!)
    - **Crucially**: If we are buying a "YES" contract, we use the **Lower Bound** of the probability.

## 3. Implementation Plan
- [ ] Modify `SpreadDistributionModel` to return both "Raw" and "Conservative" probabilities.
- [ ] Ensure that "Higher Uncertainty" always results in a **smaller** trading edge.
- [ ] Test on extreme spreads to verify the "Hallucination" is gone.
