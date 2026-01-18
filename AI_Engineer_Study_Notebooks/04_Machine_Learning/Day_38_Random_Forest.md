# Day 38: Random Forests

## 1. Introduction
A single Decision Tree is smart but unstable (prone to overfitting).
A **Random Forest** is a collection of hundreds of trees. They vote on the answer.
- "Wisdom of the Crowd".
- Reduces Variance (Overfitting).

---

## 2. Bagging (Bootstrap Aggregating)
How do we make the trees different?
1.  **Row Sampling**: Each tree gets a random subset of data (with replacement).
2.  **Feature Sampling**: Each split only looks at a random subset of features.

This ensures trees are uncorrelated.

---

## 3. Implementation

```python
from sklearn.ensemble import RandomForestClassifier

# n_estimators = Number of trees (100 is standard)
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Feature Importance (Bonus!)
print(model.feature_importances_)
# Tells you which columns mattered most.
```

---

## 4. Practical Exercises

### Exercise 1: RF vs Single Tree
Train a Decision Tree and a Random Forest on the same dataset.
Compare their Test Accuracy. The Forest almost always wins.

---

## 5. Summary
- **Ensemble**: Combining multiple models.
- **Random Forest**: Many trees, random data, random features.
- **Robust**: Works well out-of-the-box with very little tuning.

**Next Up:** **Gradient Boosting**—The Kaggle Champion.
