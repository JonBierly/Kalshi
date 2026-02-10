# Predicted Uncertainty vs. Conformal Calibration

This document explains why we use a conformal multiplier ($q$) instead of relying solely on the model's predicted standard deviation ($\hat{\sigma}$).

## 1. The Core Difference

| Concept | What it is | Analogy |
| :--- | :--- | :--- |
| **Predicted Std ($\hat{\sigma}$)** | The model's **Internal Guess** about how much noise is in the current game state. | A weatherman saying "There's a 10% chance of rain based on my map." |
| **Conformal Multiplier ($q$)** | A **Reality Check** based on how often that guess was actually right in the past. | Observing that "Every time this weatherman says 10% rain, it actually rains 40% of the time." |

## 2. Why the "Guess" isn't enough

Our NGBoost model is powerful, but it's not perfect. It can be "mis-calibrated" for several reasons:

1.  **Model Misspecification**: The model might assume the score follows a Student-T distribution, but real-world "late-game chaos" (intentional fouls, rapid-fire 3s) is "fatter-tailed" than the model can learn.
2.  **Training vs. Reality**: The model was trained on play-by-play events, but we trade on sampled boxscores. The "missing information" between samples adds a layer of uncertainty the model didn't see during training.
3.  **Temporal Bias**: Our data shows the model's "internal map" is very accurate at tip-off but becomes overly optimistic in the 4th quarter.

## 3. The Math of $q(t)$

We don't replace $\hat{\sigma}$; we **scale** it.

$$ \text{Calibrated Boundary} = \text{Prediction} \pm (q(t) \times \hat{\sigma}) $$

- If $q=1.0$, the model is perfectly calibrated (the guess matches reality).
- If $q=1.7$, the model is **over-confident** (reality is 70% noisier than the model thinks).
- If $q=0.8$, the model is **too conservative** (the model is actually better than it thinks it is).

## 4. Why $t$ (Time) matters

The "error in the guess" isn't constant. Late in the game, the incentives change (teams stop playing "normal" basketball and start playing "clock management" or "intentional fouling" basketball). 

NGBoost sees the *features* of this, but it doesn't fully internalize how much it *doesn't know* about those high-leverage moments. The $q(t)$ curve is our way of saying: *"Model, I know you think the score is decided, but historically, this is where things get weird, so I'm inflating your uncertainty by 1.6x just to be safe."*
