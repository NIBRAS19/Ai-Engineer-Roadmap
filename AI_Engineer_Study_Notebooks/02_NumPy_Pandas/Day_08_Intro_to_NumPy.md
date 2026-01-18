# Day 8: Introduction to NumPy (The Language of Matrices)

## 1. Description
Welcome to Week 2! We are leaving pure Python behind and entering the world of **High-Performance Computing**.

**NumPy** (Numerical Python) is the library that powers almost every AI tool (Pandas, Scikit-Learn, PyTorch, TensorFlow).
If you understand NumPy, you understand how machines "see" data (images, sound, text) as **Grids of Numbers**.

### Why use NumPy over Lists?
1.  **Speed**: 50x to 100x faster than Python lists.
2.  **Memory**: Uses less memory.
3.  **Convenience**: Advanced math operations (Matrix Multiplication) are built-in.

---

## 2. Installation & Setup
You likely installed it with standard AI environments. If not:
```bash
pip install numpy
```

### Importing
The standard convention is `np`.
```python
import numpy as np

print(np.__version__)
```

---

## 3. The `ndarray` (N-Dimensional Array)
The core object of NumPy. Unlike lists, arrays **must contain elements of the same type** (usually numbers). This limitation allows optimizations.

### Creating Arrays
```python
# 1. From a Python List
arr = np.array([1, 2, 3])
print(arr)      # [1 2 3]
print(type(arr)) # <class 'numpy.ndarray'>

# 2. 2D Array (Matrix) - Think Excel Sheet
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

### Essential Attributes
Understanding "Shape" is vital. **Shape Mismatch** is the #1 error in Deep Learning.
```python
print(matrix.ndim)   # 2 (It has rows and columns, so 2 dimensions)
print(matrix.shape)  # (2, 3) -> (2 Rows, 3 Columns)
print(matrix.size)   # 6 (Total elements)
print(matrix.dtype)  # int32 or int64 (Data type)
```

---

## 4. Built-in Creation Functions
Useful for initializing weights in Neural Networks.

```python
# Array of Zeros (Initializing biases)
zeros = np.zeros((3, 4)) # 3x4 matrix of 0.0

# Array of Ones
ones = np.ones((2, 2))   # 2x2 matrix of 1.0

# Range of numbers (like Python's range)
nums = np.arange(0, 10, 2) # [0, 2, 4, 6, 8]

# Linear Space (Graphing/Plotting)
# 5 numbers evenly spaced between 0 and 1
intervals = np.linspace(0, 1, 5) # [0., 0.25, 0.5, 0.75, 1.]

# Identity Matrix (Linear Algebra)
identity = np.eye(3) 
```

---

## 5. Random Numbers
AI models start with random "weights" and learn by adjusting them.

```python
np.random.seed(42)  # Makes random numbers reproducible (Crucial for debugging!)

# Random numbers between 0 and 1 (Uniform Distribution)
rand_uniform = np.random.rand(3, 3)

# Random numbers from Normal Distribution (Gaussian - Bell Curve)
# Mean=0, Std=1. Standard initialization for many models.
rand_normal = np.random.randn(3, 3) 

# Random Integers
rand_int = np.random.randint(0, 10, size=(2, 5)) # 0 to 9, shape (2,5)
```

---

## 6. Practical Exercises

### Exercise 1: The Image Placeholder
An image is just a 3D matrix (Height, Width, Color Channels).
Create a "fake image" utilizing `np.random.randint`:
- Dimensions: 28x28 pixels.
- Channels: 1 (Grayscale).
- Values: 0 to 255.
Print its shape and data type.

### Exercise 2: Weight Initialization
Create a "Weight Matrix" of shape (10, 5) initialized with random numbers from a standard normal distribution. Verify the shape.

---

## 7. Summary
- **NumPy** is the foundation of AI math.
- **Arrays** are faster/efficient lists of the same data type.
- **Attributes**: `shape`, `ndim`, and `dtype` are the first things you check when debugging data.
- **Creation**: Use `zeros`, `ones`, `arange`, and `random` to generate data.

**Next Up:** **Indexing and Slicing**—how to grab specific data from these matrices (like cropping an image).
