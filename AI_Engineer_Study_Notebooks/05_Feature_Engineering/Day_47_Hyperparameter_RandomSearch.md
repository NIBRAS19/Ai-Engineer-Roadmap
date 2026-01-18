# Day 47: Hyperparameter Tuning - Random Search

## 1. Introduction
With Grid Search, if you check 100 values for "Useless Parameter A" and 1 value for "Important Parameter B", you waste time.
**Random Search** samples random combinations.
Research shows Random Search is often **more efficient** than Grid Search for finding good models in the same amount of time.

---

## 2. Implementation

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_neighbors': randint(1, 50),     # Any int between 1 and 50
    'weights': ['uniform', 'distance']
}

# n_iter=10 -> Try 10 random combos
rand_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5)
rand_search.fit(X_train, y_train)
```

---

## 3. Bayesian Optimization (Advanced)
Random search is blind.
**Bayesian Optimization** (libraries like `Optuna`) learns from previous results.
"Trying Learning Rate=0.01 was bad, so I won't try 0.02. I'll try 0.0001."

---

## 4. Summary
- **Grid**: Comprehensive but slow.
- **Random**: Fast and effective.
- **Optuna**: State of the art.

**Next Up:** **Pipelines**—Writing professional production code.
