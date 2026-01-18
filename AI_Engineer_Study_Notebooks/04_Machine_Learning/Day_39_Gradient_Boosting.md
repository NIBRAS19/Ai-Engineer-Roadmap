# Day 39: Gradient Boosting

## 1. Introduction
Random Forest builds trees **in parallel** (independently).
**Gradient Boosting** builds trees **sequentially**.
- Tree 1 makes errors.
- Tree 2 tries to predict the *errors* of Tree 1.
- Tree 3 tries to predict the *errors* of Tree 2.

It focuses intensely on the "hard" cases.

---

## 2. XGBoost and LightGBM
While Sklearn has `GradientBoostingClassifier`, pros use **XGBoost** or **LightGBM**.
They are optimized, faster, and win almost every tabular data competition.

```python
# pip install xgboost
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)
```

---

## 3. The Risk
Because it focuses on errors, it can try to fit noise (Outliers).
Requires careful tuning of:
- `learning_rate` (how much each tree contributes).
- `n_estimators` (number of trees).

---

## 4. Practical Exercises

### Exercise 1: Boosting vs Bagging
Compare Random Forest and Gradient Boosting on a clean dataset.
Usually Boosting wins by a small margin but takes longer to tune.

---

## 5. Summary
- **Bagging (RF)**: Reduces Variance (Overfitting).
- **Boosting**: Reduces Bias (Underfitting).
- **XGBoost**: The industry standard for Tabular ML.

**Next Up:** **SVM**—Finding the widest street.
