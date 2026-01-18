# Day 13: Statistical Functions

## 1. Introduction
One of the main jobs of an AI Engineer is to understand the **distribution** of data.
- What is the average house price?
- What is the maximum pixel value?
- Is the data spread out (high variance) or clustered?

NumPy has highly optimized statistical functions.

---

## 2. Basic Statistics
```python
import numpy as np

data = np.array([10, 20, 30, 40, 50, 600]) # Note the outlier '600'

print(np.mean(data))   # 125.0 (Sensitive to outliers)
print(np.median(data)) # 35.0 (Robust to outliers)
print(np.std(data))    # 213.6... (Standard Deviation)
print(np.var(data))    # Variance (Std^2)
```

---

## 3. Min, Max, and Arg-functions
Often in classification, you don't care about the *value* of the probability, but *which class* has the highest probability.

```python
probs = np.array([0.1, 0.7, 0.2])

print(np.max(probs))    # 0.7 (Highest value)
print(np.argmax(probs)) # 1 (Index of the highest value)
```
*In this case, Index 1 might correspond to class "Dog". `argmax` is used in almost every classification pipeline.*

---

## 4. Axis Argument
Statistics on matrices. Usually, you want stats per column (feature) or per row (sample), not on the whole matrix.

- `axis=0`: Down the columns (Collapses rows). "Average of each feature".
- `axis=1`: Across the rows (Collapses columns). "Average of each sample".

```python
matrix = np.array([
    [10, 20],
    [30, 40]
])

print(np.sum(matrix))          # 100 (Sum of everything)
print(np.sum(matrix, axis=0))  # [40, 60] (Sum of columns)
print(np.sum(matrix, axis=1))  # [30, 70] (Sum of rows)
```

---

## 5. Unique and Counting
Useful for analyzing classification labels.

```python
labels = np.array([0, 1, 1, 0, 2, 1, 0])

# Get unique classes
classes = np.unique(labels)
print(classes) # [0 1 2]

# Get counts
values, counts = np.unique(labels, return_counts=True)
print(dict(zip(values, counts)))
# {0: 3, 1: 3, 2: 1} -> Class 2 is underrepresented (Imbalanced Data warning!)
```

---

## 6. Practical Exercises

### Exercise 1: Accuracy Calculation
You have predictions and truth labels.
`preds = np.array([0, 1, 1, 0, 1])`
`truth = np.array([0, 1, 0, 0, 1])`
1.  Create a boolean array checking where they match.
2.  Take the `mean()` of that array to find the accuracy percentage.
*(Hint: True is 1, False is 0).*

### Exercise 2: Column Normalization
Given a dataset `X` of shape `(100, 5)`.
Compute the mean of each column (`axis=0`).
Subtract this mean from the original data (Broadcasting!).
Verify the new mean of each column is practically 0.

---

## 7. Summary
- **Mean/Median**: Central tendency.
- **Std/Var**: Spread of data.
- **Argmax**: Index of max value (Class prediction).
- **Axis**: 0 for columns, 1 for rows.

**Next Up:** **Practical NumPy**—Putting it all together to simulate a simple Neutron Network layer from scratch.
