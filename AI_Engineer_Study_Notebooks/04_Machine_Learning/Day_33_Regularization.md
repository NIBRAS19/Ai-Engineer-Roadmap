# Day 33: Regularization (Ridge, Lasso, and Elastic Net)

## 1. Introduction
Complex models (high degree Polynomials, Deep Neural Nets) are prone to **Overfitting**.
They develop large weights ($w$) to fit noise in the training data.
**Regularization** punishes large weights, forcing the model to stay simple.

$$Loss_{regularized} = Loss_{original} + \lambda \cdot Penalty(weights)$$

Where $\lambda$ (alpha) controls how strictly we penalize.

### 🎯 Real-World Analogy: The Strict Teacher
> Think of **regularization** as a strict teacher. Without it, your model is a student who **memorizes every word of the textbook** (overfitting). 
> 
> - With **L1 regularization (Lasso)**, the teacher says: "Only remember the chapter titles" — some weights become **exactly zero**.
> - With **L2 regularization (Ridge)**, the teacher says: "You can remember everything, but don't obsess over any single sentence" — weights **shrink but don't disappear**.

---

## 2. Ridge Regression (L2 Regularization)

### The Penalty
$$Penalty_{L2} = \lambda \sum_{i} w_i^2$$

We add the **sum of squared weights** to the loss.

### Effect on Weights
- Shrinks all weights towards zero, but **never exactly zero**.
- Large weights are penalized more (due to squaring).
- Good for keeping all features but reducing their overall impact.

### 🎯 Geometric Intuition
> Imagine all your weights as a vector. L2 regularization says: "Keep the **length** of this vector small." It creates a circular constraint—all directions are penalized equally.

```python
from sklearn.linear_model import Ridge

# alpha = regularization strength (higher = simpler model)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print(model.coef_)  # All coefficients are small, but non-zero
```

---

## 3. Lasso Regression (L1 Regularization)

### The Penalty
$$Penalty_{L1} = \lambda \sum_{i} |w_i|$$

We add the **sum of absolute weights** to the loss.

### Effect on Weights
- Can force weights to be **EXACTLY 0**.
- Acts as **automatic feature selection** (removes useless features).
- Produces **sparse models** (few non-zero weights).

### 🎯 Geometric Intuition
> L1 creates a **diamond-shaped** constraint. The corners of the diamond lie on the axes, so the optimal solution often hits a corner—meaning some weights are exactly zero.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

print(model.coef_)  # Some coefficients are exactly 0.0!
```

---

## 4. L1 vs L2: Visual Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|:-------|:-----------|:-----------|
| Penalty | $\sum |w|$ | $\sum w^2$ |
| Weight Effect | Some → **exactly 0** | All → **small but non-zero** |
| Feature Selection | ✅ Yes (automatic) | ❌ No (keeps all) |
| Constraint Shape | Diamond (corners) | Circle (smooth) |
| Best For | High-dimensional sparse data | Correlated features |

### When to Use Which?
```
Do you have many features, and suspect only a few are useful?
├── Yes → Use LASSO (L1)
└── No → Do your features have high correlation (multicollinearity)?
    ├── Yes → Use RIDGE (L2)
    └── Unsure → Use ELASTIC NET (Both)
```

---

## 5. Elastic Net (The Best of Both Worlds)

What if you want BOTH feature selection AND handling of correlated features?

$$Penalty_{ElasticNet} = \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

Elastic Net combines L1 and L2 regularization.

```python
from sklearn.linear_model import ElasticNet

# l1_ratio: 0 = pure Ridge, 1 = pure Lasso, 0.5 = equal mix
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)

print(model.coef_)
```

### When Elastic Net Shines
- You have groups of correlated features (Lasso would randomly pick one).
- You want some feature selection but also stability.
- Default choice for many Kaggle competitions.

---

## 6. Choosing Alpha (Regularization Strength)

| Alpha Value | Effect |
|:------------|:-------|
| $\alpha = 0$ | No regularization (standard Linear Regression) |
| $\alpha$ small (0.01) | Mild regularization, complex model |
| $\alpha$ large (10+) | Strong regularization, very simple model |

### Finding the Best Alpha: Cross-Validation
```python
from sklearn.linear_model import RidgeCV, LassoCV

# Automatically finds best alpha via cross-validation
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")  # e.g., 1.0
```

---

## 7. Regularization in Deep Learning (Preview)

In neural networks, we don't call it "Ridge" or "Lasso"—we call it:
- **Weight Decay**: Same as L2 regularization
- **Dropout**: Randomly "turns off" neurons during training (different form of regularization)
- **Early Stopping**: Stop training before overfitting sets in

```python
# PyTorch weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 8. Practical Exercises

### Exercise 1: Lasso for Feature Selection
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np

# Create dataset: 10 features, but only first 2 are informative
X, y = make_regression(n_samples=100, n_features=10, n_informative=2, random_state=42)

# Standard Linear Regression
lr = LinearRegression().fit(X, y)
print("Linear Regression Coefs:", np.round(lr.coef_, 2))

# Lasso
lasso = Lasso(alpha=1.0).fit(X, y)
print("Lasso Coefs:", np.round(lasso.coef_, 2))  # Most should be ~0
```

### Exercise 2: Alpha Exploration
1. Train Ridge with `alpha=0.001, 0.1, 10, 1000`.
2. Plot coefficient magnitudes vs alpha.
3. Observe how coefficients shrink as alpha increases.

### Exercise 3: The Overfitting Polynomial
1. Generate noisy quadratic data: `y = x^2 + noise`.
2. Fit a degree-10 polynomial without regularization (overfits terribly).
3. Fit with Ridge regularization. Observe the smoothing effect.

---

## 9. Summary
- **Regularization**: Adds penalty to prevent overfitting.
- **Ridge (L2)**: Shrinks weights, keeps all features.
- **Lasso (L1)**: Sets some weights to zero, performs feature selection.
- **Elastic Net**: Combines both, best default choice.
- **Alpha**: Controls regularization strength—tune via cross-validation.

**Connection Note**: In Deep Learning (Day 63), this becomes `weight_decay` in optimizers.

**Next Up:** **Logistic Regression**—Predicting Yes or No.

