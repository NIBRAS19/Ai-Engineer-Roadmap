# Day 28: Loss Functions

## 1. Introduction
The **Loss Function** (or Cost Function) is the compass that guides Gradient Descent.
It boils down "How wrong is the model?" into a single number.
Different tasks require different loss functions.

---

## 2. Regression Losses (Predicting Numbers)
Calculating error for continuous values (Price, Temperature).

### 2.1 Mean Squared Error (MSE)
$$ MSE = \frac{1}{N} \sum (y_{true} - y_{pred})^2 $$
- Penalizes large errors heavily (due to square).
- Sensitive to outliers.

### 2.2 Mean Absolute Error (MAE)
$$ MAE = \frac{1}{N} \sum |y_{true} - y_{pred}| $$
- Robust to outliers.
- Gradients are constant (can be unstable near zero).

---

## 3. Classification Losses (Predicting Categories)

### 3.1 First: Understanding Softmax (Converts Scores to Probabilities)

Before we discuss cross-entropy loss, we need to understand **Softmax**.

Neural networks output **raw scores** (called logits). For a 3-class problem:
- Logits: $[2.0, 1.0, 0.1]$ — "Cat is most likely, then Dog, then Bird"

But these aren't probabilities (they don't sum to 1). **Softmax** converts them:

$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
exp_logits = np.exp(logits)            # [7.39, 2.72, 1.10]
softmax = exp_logits / exp_logits.sum() # [0.659, 0.243, 0.098]
print(softmax)  # Sums to 1.0!
```

### 🎯 Real-World Analogy: The Voting Booth
> Imagine an election. Each candidate has a raw "popularity score." Softmax is like converting those scores into **vote percentages**. A candidate with double the score doesn't get double the votes—the exponential exaggerates differences, making the winner much more confident.

### Temperature in Softmax
The **temperature** parameter ($T$) controls "confidence":

$$ \text{Softmax}(z_i, T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} $$

| Temperature | Effect | Use Case |
|:------------|:-------|:---------|
| $T = 1.0$ | Normal | Standard training |
| $T < 1.0$ | Sharper (more confident) | Greedy decoding |
| $T > 1.0$ | Softer (more uniform) | Knowledge distillation, creative text |

```python
# Temperature = 0.5 (sharper)
softmax_cold = np.exp(logits/0.5) / np.exp(logits/0.5).sum()  # [0.88, 0.11, 0.01]

# Temperature = 2.0 (softer)  
softmax_hot = np.exp(logits/2.0) / np.exp(logits/2.0).sum()   # [0.51, 0.33, 0.16]
```

---

### 3.2 Binary Cross-Entropy (Log Loss)
Used for Binary Classification (Yes/No, Spam/Ham).

$$ Loss = - (y \log(p) + (1-y) \log(1-p)) $$

- If $y=1$, we want $p=1$ (Loss -> 0).
- If $y=1$ but $p=0.01$, Loss is Huge (-log(0.01) ≈ 4.6).

### Why Logarithm?
The log creates **asymmetric penalties**:
- Being 90% confident and correct: small loss
- Being 90% confident and WRONG: HUGE loss

This forces the model to be well-calibrated.

### 3.3 Categorical Cross-Entropy (Multi-class)
Used with Softmax for multi-class problems (Cat vs Dog vs Bird).

$$ Loss = -\sum_c y_c \log(p_c) $$

But since $y$ is one-hot (only one class is 1), this simplifies to:
$$ Loss = -\log(p_{\text{correct class}}) $$

**Example:**
- True label: Cat (class 0)
- Predicted probabilities: $[0.7, 0.2, 0.1]$
- Loss: $-\log(0.7) = 0.36$

If we were less confident:
- Predicted: $[0.4, 0.3, 0.3]$
- Loss: $-\log(0.4) = 0.92$ (higher penalty!)

---

## 4. Practical Exercises

### Exercise 1: Manual MSE
Write a function `mse_loss(y_true, y_pred)` using NumPy.
Test it: `y_true = [10, 20]`, `y_pred = [12, 18]`.

### Exercise 2: Manual Entropy
Calculate the Binary Cross Entropy for:
- True Label $y = 1$
- Prediction $p = 0.9$ (Confident Correct)
- Prediction $p = 0.1$ (Confident Wrong)
Compare the loss values.

---

## 5. Summary
- **MSE**: Default for Regression.
- **Cross-Entropy**: Default for Classification.
- **Choice matters**: Using MSE for classification works technically, but performs poorly.

**CONGRATULATIONS!** You have finished **Week 4: Mathematics for AI**.
You now speak the language of gradients, matrices, and probabilities.
**Next Week:** We start **Machine Learning** proper—Scikit-Learn, Linear Regression, and Decision Trees!
