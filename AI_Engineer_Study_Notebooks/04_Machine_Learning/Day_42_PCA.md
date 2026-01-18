# Day 42: Principal Component Analysis (PCA)

## 1. Introduction
High dimensions are bad.
- **Curse of Dimensionality**: Data becomes sparse. Distance meaningless.
- **Visualization**: We can't see 100D.
- **Speed**: Training is slow.

**PCA** compresses data by finding the "Principal Components" (directions of max variance) and projecting data onto them.

---

## 2. Implementation
Reduce 100 features to 10.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

print(sum(pca.explained_variance_ratio_))
# e.g., 0.95 (We kept 95% of the information with only 10% of features!)
```

---

## 3. Visualization
We often use PCA to reduce data to 2D or 3D just to plot it.

```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:,0], X_2d[:,1])
```

---

## 4. Practical Exercises

### Exercise 1: MNIST Compression
Load MNIST (784 features per image).
Run PCA to keep 95% variance.
How many components did you need? (Usually ~150).
The 784 -> 150 compression speeds up training massively.

---

## 5. Summary
- **PCA**: Rotates and projects data.
- **Variance**: Information.
- **Preprocessing**: Often done before ML models.

**CONGRATULATIONS!** You have finished **Weeks 5-6: Machine Learning**.
You have mastered the Classical ML Toolkit.
**Next Week:** **Feature Engineering**—Deep diving into Pipelines and handling tough real-world data issues.
