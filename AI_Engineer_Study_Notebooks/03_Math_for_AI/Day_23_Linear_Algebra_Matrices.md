# Day 23: Linear Algebra - Matrices

## 1. Introduction
A Matrix is a grid of numbers.
But mathematically, **A Matrix is a Function**.
It takes a vector and **transforms** it (rotates, scales, shears).

$$ A \cdot x = b $$
Input $x$, Apply Transformation $A$, Get Output $b$.

---

## 2. Matrix Multiplication
(Covered in NumPy, but here's the math).
Dimensions: $(M \times N) \cdot (N \times K) = (M \times K)$.
This is the mechanism of passing data through layers.
Layer 1 ($N$ neurons) -> Layer 2 ($K$ neurons).

---

## 3. Special Matrices

### 3.1 Identity Matrix ($I$)
The "1" of matrices.
$A \cdot I = A$
```python
I = np.eye(3)
```

### 3.2 Inverse Matrix ($A^{-1}$)
Reverses the transformation of $A$.
$A \cdot A^{-1} = I$
**AI Context:** Used in some analytical solutions (Ordinary Least Squares), but rarely in Deep Learning because calculating it is computationally expensive ($O(N^3)$).

## 4. Eigenvalues and Eigenvectors (The DNA of Matrices)

When a matrix transforms a vector, usually the vector changes direction.
**Eigenvectors** are special vectors that **do not change direction**, only length.
**Eigenvalues** are how much they stretch.

$$ A \cdot v = \lambda \cdot v $$
- $A$: Transformation Matrix
- $v$: Eigenvector (direction preserved)
- $\lambda$: Eigenvalue (stretch factor)

### 🎯 Real-World Analogy: The Spinning Top
> Imagine a spinning top on a table. No matter how the table shakes (transformation), the top spins around its **axis** (eigenvector). The axis direction doesn't change—only the spin speed might (eigenvalue).

### Why Eigenvectors Matter in AI

#### Use Case 1: Principal Component Analysis (PCA) — Day 42 Preview
PCA finds the "main axes" of your data's variance.

```python
import numpy as np

# Scatter plot of correlated data
data = np.array([
    [1, 2], [2, 4], [3, 5], [4, 4], [5, 6]
])

# Step 1: Covariance Matrix
cov_matrix = np.cov(data.T)

# Step 2: Find Eigenvectors and Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eigenvalues)      # [3.7, 0.3] <- First is bigger = more variance
print("Eigenvectors:\n", eigenvectors)  # Directions of maximum spread
```

**Interpretation:**
- The **largest eigenvalue** corresponds to the direction of **maximum variance**.
- This is the first "principal component"—the most important feature direction.
- You can **compress data** by keeping only the top k eigenvectors.

### Eigenvalue Decomposition Formula
Any symmetric matrix can be decomposed:
$$ A = Q \Lambda Q^T $$
- $Q$: Matrix of eigenvectors (columns)
- $\Lambda$: Diagonal matrix of eigenvalues

This is how PCA reduces 1000 dimensions to 50 while keeping most information.

### When Eigenvalues Help in Deep Learning
| Application | How Eigenvalues Help |
|:------------|:--------------------|
| Weight Initialization | Eigenvalues of weight matrix affect signal propagation |
| Hessian Analysis | Eigenvalues of Hessian predict optimization difficulty |
| Graph Neural Networks | Eigenvalues of adjacency matrix encode graph structure |

---

## 5. Practical Exercises

### Exercise 1: Inverse
1.  Create a 2x2 matrix `A = [[2, 1], [1, 2]]`.
2.  Compute `A_inv`.
3.  Compute `np.dot(A, A_inv)`. Is it equal to Identity?

---

## 6. Summary
- **Matrix**: A linear transformation map.
- **Inverse**: The "Undo" button for matrices (if it exists).
- **Eigenvectors**: The axes of rotation/stability.

**Next Up:** **Calculus**—The study of change, and the backbone of training AI.
