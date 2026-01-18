# Day 125: Monitoring & A/B Testing

## 1. The Silent Killer: Drift
A web server typically crashes loud (500 Error).
A ML model **fails silently**. It returns predictions, but they are wrong.

### 1.1 Types of Drift
1.  **Data Drift** ($P(X)$ changes): Input distributions change.
    - *Example*: You trained on sunny images, now it's raining.
2.  **Concept Drift** ($P(Y|X)$ changes): The relationship changes.
    - *Example*: Spam emails look different today than 5 years ago. The definition of "Spam" evolved.

### 1.2 Detection Tools
- **Evidently AI**: Open-source library to visualize drift.
- **Arize / Fiddler**: Enterprise monitoring platforms.
- **KS-Test (Kolmogorov-Smirnov)**: Statistical test to check if two distributions differ.

---

## 2. Deployment Strategies

### 2.1 A/B Testing
Running two Models in parallel to see which converts better.
- **Model A (Control)**: Current Champion. (Traffic: 90%)
- **Model B (Challenger)**: New Model. (Traffic: 10%)
- Compare conversion rates. If B > A, promote B.

### 2.2 Canary Deployment
Roll out Model B to a small subset of users (e.g., internal employees first, then 1%, then 5%).
If errors spike, **Rollback** instantly.

### 2.3 Shadow Mode (Safest)
Deploy Model B alongside Model A.
- Model A gives the result to the user.
- Model B makes a prediction **silently** (logged, but ignored).
- We compare Model B's silent predictions vs Ground Truth later to verify safety before going live.

---

## 3. Retraining Strategies
1.  **Manual**: Data Scientist triggers it.
2.  **Schedule**: Every Sunday night (Dangerous if drift happens on Monday).
3.  **Trigger-based**: If Drift > Threshold, automatically retrain.

---

## 4. Summary
- **Drift**: The enemy of deployed models.
- **Evidently AI**: Tool to visualize drift.
- **Shadow Mode**: The safest way to test a new model in production.

**Next Up:** **Final Capstone**—The 15-Day Challenge.

