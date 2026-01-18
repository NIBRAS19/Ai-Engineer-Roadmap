# Day 43: Feature Scaling

## 1. Introduction
Distance-based algorithms (KNN, SVM, K-Means) and Gradient Descent algorithms (Linear Regression, Neural Networks) differ in performance significantly based on the **Scale** of the data.
If "Salary" is 0-100,000 and "Age" is 0-100, the model thinks Salary is 1000x more important.
**Scaling** brings them to the same range.

---

## 2. Standardization (Z-Score Normalization)
$$ x_{new} = \frac{x - \mu}{\sigma} $$
- Mean becomes 0.
- Std Dev becomes 1.
- **Best for**: Algorithms assuming Normal Distribution (Logistic Regression, SVM, Neural Nets).
- **Outliers**: Does not crush outliers (they remain large, e.g., 5.0).

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 3. Min-Max Scaling (Normalization)
$$ x_{new} = \frac{x - min}{max - min} $$
- Range becomes [0, 1].
- **Best for**: Algorithms that need bounded input (Image pixel values 0-255 -> 0-1).
- **Outliers**: Sensitive. An outlier of 1,000,000 squashes "normal" data to 0.0001.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4. Practical Exercises

### Exercise 1: Impact on KNN
1.  Create dataset with two features: A (Range 0-1), B (Range 0-1000).
2.  Train KNN. It will ignore Feature A.
3.  Scale them using StandardScaler.
4.  Train KNN. It should use both.

---

## 5. Summary
- **Scale your data**: Unless you use Tree-based models (Random Forest, XGBoost), which are invariant to scale.
- **Fit vs Transform**: call `fit()` ONLY on Train set. call `transform()` on Test set. **Never fit on Test set**.

**Next Up:** **Categorical Features**—Handling text labels.
