# Day 32: Polynomial Regression

## 1. Introduction
Linear Regression ($y=mx+b$) fails when the data is curved (Parabolic, Exponential).
**Polynomial Regression** fits a curve by adding powers of X as new features.
$y = w_0 + w_1 x + w_2 x^2 + ...$

---

## 2. Feature Transformation
We don't need a new algorithm! analyzing $x^2$ is just Linear Regression on a new feature.
We use `PolynomialFeatures` to generate these powers.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Curved Data
X = np.arange(5).reshape(-1, 1) # [[0], [1], [2], [3], [4]]
y = X**2                        # [[0], [1], [4], [9], [16]]

# 1. Transform Features (Degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print(X_poly)
# 1   x   x^2
# [[1, 0,  0],
#  [1, 1,  1],
#  [1, 2,  4], ...]

# 2. Train Linear Regression on transformed data
model = LinearRegression()
model.fit(X_poly, y)
```

---

## 3. The Danger of Overfitting
If you choose `degree=50`, the curve will wiggle through every single data point.
- **Train Accuracy**: 100%.
- **Test Accuracy**: Terrible.
This is **High Variance**. Only use the lowest degree that fits the pattern.

---

## 4. Practical Exercises

### Exercise 1: Quadratic Fit
Generate data $y = 2x^2 + 5 + noise$.
Fit a Linear Regression (Degree 1) and a Polynomial Regression (Degree 2).
Compare their R2 scores. Degree 2 should be much higher.

---

## 5. Summary
- **Polynomial Regression**: Linear Regression on engineered features ($x^2, x^3$).
- **Degree**: Hyperparameter controlling curve complexity.

**Next Up:** **Regularization**—How to stop models from overfitting.
