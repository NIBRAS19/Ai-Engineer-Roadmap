# Day 31: Linear Regression

## 1. Introduction
The "Hello World" of ML algorithms.
It fits a straight line through data to predict a continuous value.
Equation: $y = mx + b$ (or $y = w \cdot x + b$)

- **Parameters**: Weights ($w$) and Bias ($b$).
- **Objective**: Minimize MSE (Mean Squared Error).

---

## 2. Implementation with Scikit-Learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Data (Height vs Weight)
X = np.array([[150], [160], [170], [180]]) # Height (must be 2D array!)
y = np.array([50, 55, 65, 75])             # Weight

# 2. Model
model = LinearRegression()

# 3. Train
model.fit(X, y)

# 4. Interpret
print(f"Intercept (b): {model.intercept_}")
print(f"Coefficient (w): {model.coef_}")
# E.g., coef=0.8 means "For every 1 unit of Height, Weight increases by 0.8"

# 5. Predict
new_height = [[175]]
pred = model.predict(new_height)
print(f"Predicted Weight for 175cm: {pred[0]}")
```

---

## 3. Evaluation
How good is the line?
- **MSE**: Average squared error (Hard to interpret).
- **R-squared ($R^2$)**: Percentage of variance explained. 1.0 is perfect, 0.0 is useless.

```python
from sklearn.metrics import r2_score

y_pred = model.predict(X)
print("R2 Score:", r2_score(y, y_pred))
```

---

## 4. Assumptions of Linear Regression
It works best if:
1.  Relationship is actually linear.
2.  Errors are normally distributed.
3.  Features don't have high correlation with each other (Multicollinearity).

---

## 5. Practical Exercises

### Exercise 1: Sales Predictor
Data: `Advertising Spend ($)` vs `Sales (Units)`.
X = [[10], [20], [30], [40], [50]]
y = [15, 25, 35, 45, 55]
1.  Train a Linear Regression.
2.  Predict sales for Spend = 100.
3.  Does it perfectly match? (It should, this data is perfect $y=x+5$).

---

## 6. Summary
- **Linear Regression**: Fits a line to minimize error.
- **Fitting**: Learning `coef_` and `intercept_`.
- **R2 Score**: Score from 0 to 1.

**Next Up:** **Multivariate & Polynomial Regression**—What if the data is curved?
