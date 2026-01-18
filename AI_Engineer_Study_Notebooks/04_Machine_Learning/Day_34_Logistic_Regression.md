# Day 34: Logistic Regression

## 1. Introduction
Despite the name, **Logistic Regression** is for **Classification**, not Regression.
It predicts probabilities (0 to 1) using the **Sigmoid** function.
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

---

## 2. Implementation
Predicting if a tumor is Malignant (1) or Benign (0).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load Data
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)         # [0, 1, 1...]
probs = model.predict_proba(X_test)   # [[0.1, 0.9], ...] (Probabilities)
```

---

## 3. Decision Boundary
The model learns a hyper-plane (line) that separates the two classes.
- $P > 0.5 \rightarrow 1$
- $P < 0.5 \rightarrow 0$

---

## 4. Practical Exercises

### Exercise 1: Custom Threshhold
Normally we split at 0.5.
For cancer detection, we hate False Negatives.
Change the logic: If probability > 0.1, predict Cancer(1).
Check how this changes the number of positive predictions.

---

## 5. Summary
- **Logistic Regression**: Linear model for binary classification.
- **Sigmoid**: Squashes output to 0-1.
- **Probabilities**: More useful than just hard labels.

**Next Up:** **Decision Trees**—Learning human-like rules.
