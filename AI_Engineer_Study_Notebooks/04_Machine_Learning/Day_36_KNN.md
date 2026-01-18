# Day 36: K-Nearest Neighbors (KNN)

## 1. Introduction
KNN is "Lazy Learning".
It doesn't learn a formula. It just memorizes the data.
To predict a new point:
1.  Find the **K** closest points in the training set.
2.  Vote (Classification) or Average (Regression).

---

## 2. Choosing K
- **K=1**: Very sensitive to noise (Outliers dictate result).
- **K=100**: Too smooth (Ignores local patterns).
- **K is typically odd** (to avoid tie votes).

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train) # Just stores data
```

---

## 3. Distance Metrics
How do we define "Closest"?
- **Euclidean**: Standard straight line.
- **Manhattan**: Grid-like (City blocks).

**Scaling Issue**:
If one feature is "Salary" (100000) and another is "Age" (50), Salary dominates the distance.
**You MUST normalize data before using KNN.**

---

## 4. Practical Exercises

### Exercise 1: The Effect of Scaling
1.  Create dataset with Age (0-100) and Income (0-100000).
2.  Train KNN without scaling.
3.  Scale data (StandardScaler).
4.  Train KNN again. Compare results.

---

## 5. Summary
- **KNN**: Distance-based prediction.
- **Scaling**: Mandatory.
- **Computation**: Slow at prediction time (needs to measure distance to everyone).

**Next Up:** **Evaluation Metrics**—You trained models, but are they good?
