# Day 40: Support Vector Machines (SVM)

## 1. Introduction
SVM tries to find a **hyperplane** (a line in 2D, a plane in 3D, a hyperplane in higher dimensions) that separates classes with the **widest margin** possible.

### 🎯 Real-World Analogy: The Fence Between Farms
> Imagine two farms—one grows apples, one grows oranges. You need to build a fence between them. You could put the fence anywhere, but the **smartest** location is exactly in the middle, as far as possible from both farms. That way, if someone new buys land near the boundary, it's clear which farm they belong to.
>
> SVM builds this optimal fence.

---

## 2. Margins and Support Vectors

### What is the Margin?
The **margin** is the distance between the hyperplane and the nearest data points from each class.

```
        Apple Farm          Fence          Orange Farm
            ●                 |                 ○
          ● ●              <----->            ○ ○
            ●       margin    |    margin       ○
                              |
```

### Support Vectors
The **support vectors** are the data points closest to the decision boundary. They "support" the hyperplane—if they moved, the boundary would change.

### 🎯 Intuition: Maximum Margin = Maximum Confidence
> A wider margin means the model is more confident. If someone stands exactly on the fence, we're unsure. But if there's a huge gap, anyone within that gap clearly belongs to one side.

```python
from sklearn.svm import SVC

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# See the support vectors
print(f"Support vectors shape: {model.support_vectors_.shape}")
# These are the critical data points
```

---

## 3. The C Parameter (Regularization)

**C** controls the trade-off between:
- **High C**: Strict—classify ALL training points correctly. Risk: Overfitting.
- **Low C**: Tolerant—allow some mistakes for a wider margin. Risk: Underfitting.

| C Value | Margin Width | Training Errors | Generalization |
|:--------|:-------------|:----------------|:---------------|
| C = 0.01 | Wide | Many allowed | Good (if data is noisy) |
| C = 1.0 | Medium | Few allowed | Balanced |
| C = 100 | Narrow | Almost none | Risk of overfitting |

```python
# Loose boundary (tolerates mistakes)
svm_soft = SVC(kernel='linear', C=0.01)

# Strict boundary (no mistakes allowed)
svm_hard = SVC(kernel='linear', C=100)
```

---

## 4. The Kernel Trick (Non-Linear Boundaries)

### The Problem
What if data isn't separable by a straight line?

```
    ○ ○ ○ ○          
  ●   ●   ●          <- Blue dots surrounded by red ring
    ○ ○ ○ ○               No straight line can separate them!
```

### The Kernel Trick: Project to Higher Dimensions

#### 🎯 Real-World Analogy: The Party Floor
> Imagine you're at a party where introverts sit in the center and extroverts stand around the edges. Looking from above (2D), you can't draw a straight line to separate them.
>
> But what if everyone **jumps**? The introverts, being shy, barely leave the ground. The extroverts leap high. Now in 3D (x, y, jump_height), you can separate them with a horizontal plane!
>
> The **kernel trick** does this mathematically without actually computing the higher dimensions.

### Common Kernels

| Kernel | Formula | Use Case |
|:-------|:--------|:---------|
| **Linear** | $K(x,y) = x \cdot y$ | Linearly separable data |
| **RBF (Gaussian)** | $K(x,y) = e^{-\gamma ||x-y||^2}$ | Most common, handles curves |
| **Polynomial** | $K(x,y) = (x \cdot y + c)^d$ | Polynomial decision boundaries |

```python
# Linear kernel (straight lines only)
svm_linear = SVC(kernel='linear')

# RBF kernel (curved boundaries, the default)
svm_rbf = SVC(kernel='rbf', gamma='scale')  # gamma controls "wiggliness"

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)
```

---

## 5. The Gamma Parameter (RBF Kernel)

**Gamma** controls how much influence a single training point has.

| Gamma | Effect | Analogy |
|:------|:-------|:--------|
| Low gamma | Far reach, smooth boundaries | A lighthouse beam (shines far) |
| High gamma | Close reach, wiggly boundaries | A flashlight (focused, local) |

```python
# Smooth decision boundary
svm_smooth = SVC(kernel='rbf', gamma=0.1)

# Wiggly decision boundary (may overfit)
svm_wiggly = SVC(kernel='rbf', gamma=10)
```

---

## 6. SVM Pros and Cons

| Pros | Cons |
|:-----|:-----|
| ✅ Effective in high dimensions | ❌ Slow for large datasets (>100k samples) |
| ✅ Memory efficient (uses only support vectors) | ❌ Doesn't give probability estimates by default |
| ✅ Works well with clear margins | ❌ Sensitive to feature scaling |
| ✅ Versatile kernels | ❌ Hard to interpret (black box) |

### Important: Always Scale Your Features!
SVMs are **very sensitive** to feature scales.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Always use a pipeline to scale!
svm_pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0)
)
svm_pipeline.fit(X_train, y_train)
```

---

## 7. Practical Exercises

### Exercise 1: The Donut Problem
```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Generate concentric circles (not linearly separable)
X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)

# Try linear kernel (will fail)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)
print(f"Linear kernel accuracy: {svm_linear.score(X, y):.2f}")  # ~50%

# Try RBF kernel (will succeed)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X, y)
print(f"RBF kernel accuracy: {svm_rbf.score(X, y):.2f}")  # ~100%
```

### Exercise 2: C Parameter Exploration
1. Generate linearly separable data with some noise.
2. Train SVMs with C = 0.001, 1.0, 1000.
3. Visualize the decision boundaries. How does C affect the margin?

### Exercise 3: Gamma Exploration
1. Use `make_moons` dataset.
2. Train RBF SVMs with gamma = 0.1, 1.0, 10.0.
3. Observe how gamma affects boundary complexity.

---

## 8. Summary
- **SVM**: Finds the hyperplane with maximum margin between classes.
- **Support Vectors**: The critical points that define the boundary.
- **C Parameter**: Trade-off between margin width and training errors.
- **Kernel Trick**: Projects data to higher dimensions for non-linear separation.
- **Gamma**: Controls RBF kernel smoothness.
- **Always Scale**: SVMs are sensitive to feature magnitudes!

**When to Use SVM**:
- Small to medium datasets (<100k samples)
- Clear margin of separation expected
- High-dimensional data (text classification, genomics)

**Next Up:** **Unsupervised Learning**—Clustering.

