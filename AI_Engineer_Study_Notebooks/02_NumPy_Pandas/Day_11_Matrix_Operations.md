# Day 11: Matrix Operations (The Dot Product)

## 1. Introduction
If you are learning AI, **this is the most important day of the week**.
Neural Networks are literally layers of **Dot Products**.
$Output = Input \cdot Weights + Bias$

If you understand `np.dot`, you understand 80% of deep learning math.

---

## 2. Element-wise vs Dot Product

### Element-wise Multiplication (`*`)
Multiplies matching positions.
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])

print(A * B) 
# [[1 0]
#  [0 4]]
```

### Dot Product (`@` or `np.dot`)
Row of First $\times$ Column of Second.
```python
# Matrix Multiplication
print(A @ B)
# [[1*1+2*0, 1*0+2*1],
#  [3*1+4*0, 3*0+4*1]]
# Result: 
# [[1 2]
#  [3 4]]
```

---

## 3. The Rules of Dot Product
For `A @ B` to work, **Inner Dimensions must match**.

- Shape of A: `(Rows_A, Cols_A)`
- Shape of B: `(Rows_B, Cols_B)`
- **Condition**: `Cols_A == Rows_B`
- **Result Shape**: `(Rows_A, Cols_B)`

### Example: Valid
A: `(2, 3)`
B: `(3, 4)`
Match: `3 == 3` (Yes)
Result: `(2, 4)`

### Example: Invalid
A: `(2, 3)`
B: `(2, 3)`
Match: `3 == 2` (No!) -> `ValueError: shapes (2,3) and (2,3) not aligned`

---

## 4. Transpose (`.T`)
Flips rows and columns. This is your "Fix Shape Errors" tool.

```python
A = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)

print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
# Shape (3, 2)
```

**Case Study**:
If you have A `(2, 3)` and B `(2, 3)`, you can't multiply them.
But you can multiply `A @ B.T`!
`(2, 3) @ (3, 2) -> (2, 2)`

---

## 5. Other Linear Algebra Operations
NumPy has a sub-library `np.linalg`.

```python
A = np.array([[1, 2], [3, 4]])

# 1. Determinant
print(np.linalg.det(A)) # -2.0

# 2. Inverse (A^-1)
print(np.linalg.inv(A))

# 3. Norm (Magnitude / Length of vector)
v = np.array([3, 4])
print(np.linalg.norm(v)) # 5.0 (Pythagoras 3-4-5)
```

---

## 6. Practical Exercises

### Exercise 1: The Neural Net Layer
Simulate a single layer of a neural network.
1.  `inputs` shape (1, 3): `[1, 2, 3]`
2.  `weights` shape (3, 2): Random numbers.
3.  `bias` shape (2,): `[1, 1]`
4.  Compute: $Output = inputs \cdot weights + bias$
Check the final shape. It should be (1, 2).

### Exercise 2: Similarity Search
Dot product measures similarity!
Given 3 vectors (user ratings for movies):
- Me: `[5, 1, 5]` (Action lover)
- User A: `[1, 5, 1]` (Romance lover)
- User B: `[4, 2, 5]` (Action lover)

Calculate dot product of Me vs A, and Me vs B. Who is more similar to me? (Higher score = More similar).

---

## 7. Summary
- **Dot Product (`@`)**: Matrix multiplication.
- **Rule**: Inner dimensions must match `(M, N) @ (N, K) -> (M, K)`.
- **Transpose (`.T`)**: Flips dimensions, essential for aligning shapes.

**Next Up:** **Reshaping Arrays**—how to flatten images and rearrange data dimensions.
