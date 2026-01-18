# Day 22: Linear Algebra - Vectors

## 1. Introduction
Linear Algebra is the language of AI.
- An image is a Matrix.
- A prediction is a Vector.
- Training is Matrix Multiplication.

Today, we revisit **Vectors** but from a mathematical perspective, not just a coding one.
**Goal:** Understand Magnitude (Norm) and Direction (Dot Product).

---

## 2. Scalars vs Vectors
- **Scalar**: A single number (e.g., Speed = 50 km/h). Magnitude only.
- **Vector**: A list of numbers (e.g., Velocity = [50, 10]). Magnitude AND Direction.

In AI, a vector represents a single **Data Point**.
Example: A House [Price, Size, Bedrooms] -> $x = [200000, 1500, 3]$.

---

## 3. Vector Operations

### 3.1 Addition
Adding two vectors moves you in the combined direction.
$v_1 = [1, 2]$, $v_2 = [3, 1]$
$v_{sum} = [4, 3]$

### 3.2 Scalar Multiplication
Scaling a vector.
$2 \cdot [1, 2] = [2, 4]$ (Same direction, double the length).

---

## 4. The Norm (Magnitude/Length)
How "big" is the vector?
In Euclidean space (L2 Norm), it's the Pythagorean theorem.

$$ ||v||_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2} $$

**Why important?**
Regularization (L1/L2) penalizes large weights (large norms) to prevent overfitting.

```python
import numpy as np
v = np.array([3, 4])
norm = np.linalg.norm(v)
print(norm) # 5.0
```

---

## 5. The Dot Product (Revisited)
$$ A \cdot B = ||A|| ||B|| \cos(\theta) $$

This formula tells us about the **Angle** ($\theta$) between vectors.
- If $A \cdot B = 0$: Vectors are **Orthogonal** (90 degrees, Unrelated).
- If $A \cdot B > 0$: Vectors point in similar direction (Correlated).
- If $A \cdot B < 0$: Vectors point in opposite direction (Negatively Correlated).

**AI Context:** Recommendation Systems.
If "User Vector" is close to "Movie Vector" (High dot product), recommend it.

---

## 6. Practical Exercises

### Exercise 1: Cosine Similarity
Cosine Similarity ignores magnitude and focuses only on direction.
$$ \text{Similarity} = \frac{A \cdot B}{||A|| ||B||} $$

Calculate the similarity between:
- Apple: `[1, 1]`
- Banana: `[1, 1.1]` (Should be high)
- Car: `[-1, 0]` (Should be low/negative)

Implementation:
```python
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

---

## 7. Summary
- **Vector**: An arrow in space representing data.
- **Norm**: Length of the arrow (Importance).
- **Dot Product**: Similarity between arrows.

**Next Up:** **Matrices**—Transformations, rotations, and solving systems of equations.
