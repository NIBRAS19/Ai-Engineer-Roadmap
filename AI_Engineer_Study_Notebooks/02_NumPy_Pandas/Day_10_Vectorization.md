# Day 10: Vectorization and Broadcasting (Speed Math)

## 1. Introduction
In Python, we use loops to process lists. In NumPy, **we NEVER use loops**.
why?
1.  Python loops are slow (type checking, overhead).
2.  NumPy **Vectorization** pushes the loop checking to compiled C code. It's instant.

---

## 2. Vectorized Operations (Element-wise)
You can treat arrays like single numbers.

### Basic Math
```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20, 30])

# Addition
print(arr1 + arr2) # [11 22 33]

# Subtraction
print(arr2 - arr1) # [9 18 27]

# Multiplication
print(arr1 * arr2) # [10 40 90]

# Division
print(arr2 / 2)    # [5 10 15]

# Power
print(arr1 ** 2)   # [1 4 9]
```

### Logical Operations
```python
a = np.array([True, True, False])
b = np.array([True, False, False])

print(a & b) # [True False False] (AND)
print(a | b) # [True True False] (OR)
```

---

## 3. Broadcasting
What happens if you try to add arrays of **different shapes**?
Standard Linear Algebra says "Error".
NumPy says "I got you". It **Broadcasts** (stretches) the smaller array to fit the larger one.

### 🎯 Real-World Analogy: The Radio Tower
> **Broadcasting** is like a radio tower. If you have one speaker (a scalar like `5`) and one house (an array like `[1, 2, 3]`), the speaker "broadcasts" to every room. If you have a row of speakers and a column of houses, they match up where dimensions align—just like how one song on the radio can be heard in every car simultaneously without making copies of the DJ.

This is why it's called "broadcasting"—NumPy doesn't actually copy the data; it just *pretends* the smaller array exists everywhere it's needed.

### Rule 1: Scalar to Array
Adding a single number to a matrix.
```python
arr = np.array([1, 2, 3])
# NumPy "stretches" 5 to [5, 5, 5]
print(arr + 5) # [6 7 8]
```

### Rule 2: Matrix + Row
Adding a row Vector to every row in a Matrix.
```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # Shape (2, 3)

row = np.array([10, 20, 30]) # Shape (3,)

# NumPy adds the row to EACH row of the matrix
print(matrix + row)
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Rule 3: Column + Row (Outer Product logic)
```python
col = np.array([[1], [2], [3]]) # Shape (3, 1)
row = np.array([10, 20])        # Shape (2,)

# Result Shape (3, 2)
print(col + row)
# [[11 21],
#  [12 22],
#  [13 23]]
```

### The General Broadcasting Rule
Two dimensions are compatible if:
1.  They are equal, OR
2.  One of them is 1.

NumPy compares shapes **right-to-left** (trailing dimensions first).

### 🚨 Common Broadcasting Errors (Causes 50% of NumPy Bugs!)
```python
# ERROR CASE: Incompatible shapes
a = np.array([[1, 2, 3]])    # Shape (1, 3)
b = np.array([[1], [2]])     # Shape (2, 1)

# This WORKS: (1,3) + (2,1) broadcasts to (2,3)
print((a + b).shape)  # (2, 3)

# DANGEROUS CASE: Accidental dimension mismatch
weights = np.array([0.1, 0.2, 0.3])  # Shape (3,) - 1D
data = np.random.rand(5, 4)           # Shape (5, 4)

# This FAILS! Trailing dimensions: 3 vs 4 → Not compatible!
# result = data + weights  # ValueError!

# FIX: Ensure last dimension matches
weights_fixed = np.array([0.1, 0.2, 0.3, 0.4])  # Shape (4,)
result = data + weights_fixed  # Works: (5,4) + (4,) → (5,4)
```

### 🔍 Debugging Tip: Always Print Shapes
```python
# Before any operation, check shapes
print(f"a.shape: {a.shape}, b.shape: {b.shape}")
# This one habit will save you hours of debugging.
```

---

## 4. Universal Functions (ufuncs)
Fast element-wise functions.
```python
arr = np.array([0, 90, 180])

print(np.sin(arr))    # Sine
print(np.sqrt([4, 9, 16])) # [2. 3. 4.]
print(np.exp([1, 2])) # e^1, e^2 (Exponential)
print(np.log([1, 10])) # Natural Log data
```

---

## 5. Practical Exercises

### Exercise 1: RMSE Implementation
Compute the **Root Mean Squared Error** between two arrays without loops.
`predictions = np.array([2.5, 0.0, 2.1, 7.8])`
`targets = np.array([3.0, -0.5, 2.0, 7.0])`

Formula: $\sqrt{\frac{1}{N} \sum (pred - target)^2}$
*Hint: Subtract -> Square -> Mean -> Sqrt*

### Exercise 2: Normalization
Given a dataset row `[10, 20, 30]`. Use broadcasting to:
1.  Subtract the mean.
2.  Divide by the standard deviation.
Result should be Z-scores.

---

## 6. Summary
- **Vectorization**: Doing math on entire arrays at once. Fast & Clean.
- **Broadcasting**: Auto-stretching smaller arrays to match larger ones.
- **Rules**: Dimensions match or one is 1.

**Next Up:** **Matrix Operations**—The heart of Neural Networks (Dot Product).
