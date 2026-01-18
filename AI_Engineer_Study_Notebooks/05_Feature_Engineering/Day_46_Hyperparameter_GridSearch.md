# Day 46: Hyperparameter Tuning - Grid Search

## 1. Introduction
Models have:
- **Parameters**: Learned from data (Weights).
- **Hyperparameters**: Set by you (Learning Rate, Tree Depth, K in KNN).
How do you pick the best ones?

---

## 2. Grid Search
Brute-force testing every combination.
- `K = [1, 3, 5]`
- `Metric = ['euclidean', 'manhattan']`
Total runs: $3 \times 2 = 6$.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# cv=5 means 5-Fold Cross Validation (Robust scoring)
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

---

## 3. Pros and Cons
- **Pros**: Guaranteed to find the best combo in the grid.
- **Cons**: Slow. Exponential explosion if you have many parameters.

---

## 4. Summary
- **GridSearch**: Try everything properly.
- **CV**: Cross-Validation ensures you don't adjust params to fit just one specific test split.

**Next Up:** **Random Search**—The smarter, faster way.
