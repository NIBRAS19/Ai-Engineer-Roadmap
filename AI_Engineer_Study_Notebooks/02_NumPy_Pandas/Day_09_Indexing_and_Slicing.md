# Day 9: Indexing and Slicing Arrays

## 1. Introduction
Accessing data in NumPy is similar to Python lists but far more powerful. You can access rows, columns, or specific sub-regions.
In AI, this is how you:
- **Crop** an image.
- **Split** a dataset into Features (X) and Target (y).
- **Filter** data based on conditions.

---

## 2. Basic Indexing (1D Arrays)
Works exactly like lists.
```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print(arr[0])       # 10 (First)
print(arr[-1])      # 50 (Last)
print(arr[1:4])     # [20 30 40] (Slice)
print(arr[::2])     # [10 30 50] (Step of 2)
```

**Key Difference:** Slicing a NumPy array returns a **View**, not a Copy. Modifying the slice modifies the original array!
```python
slice_arr = arr[0:2]
slice_arr[0] = 999
print(arr)  # [999, 20, 30, 40, 50] -> Original Changed!
```
*To avoid this, use `.copy()`.*

### 🔴 Critical: Views vs Copies (Memory Behavior)

This concept causes **countless bugs** and is essential to understand before working with AI data pipelines.

#### 🎯 Real-World Analogy: Shared vs. Personal Documents
> Think of a **View** like a shared Google Doc. If you edit it, everyone sees the changes. A **Copy** is like downloading that Google Doc to your computer—your changes are yours alone.

#### When NumPy Creates a View:
- **Basic slicing**: `arr[0:5]`, `arr[::2]`, `arr[:, 1]`
- **Transposing**: `arr.T`
- **Reshaping** (usually): `arr.reshape(2, 3)`

#### When NumPy Creates a Copy:
- **Fancy indexing**: `arr[[0, 2, 4]]` (passing a list of indices)
- **Boolean indexing**: `arr[arr > 0]`
- **Explicit copy**: `arr.copy()`

#### Practical Example: The Train/Test Split Trap
```python
# TRAP: If you normalize training data, test data changes too!
all_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

train = all_data[:3]    # This is a VIEW!
test = all_data[3:]     # This is a VIEW!

# Normalize training data (subtract mean)
train -= train.mean()   # Modifying VIEW modifies original!

print(all_data)  # [-1, 0, 1, 4, 5] - Original is messed up!
print(test)      # [4, 5] - Oops, test is still pointing to corrupted data

# CORRECT WAY: Use .copy()
train = all_data[:3].copy()
test = all_data[3:].copy()
```

#### 🔍 Checking if Array is a View
```python
# .base is None -> It's an independent copy
# .base is not None -> It's a view of something else
slice_arr = arr[0:2]
print(slice_arr.base is arr)  # True -> It's a view!

copied = arr[0:2].copy()
print(copied.base)  # None -> Independent copy
```

---

## 3. 2D Indexing (Matrices)
Syntax: `arr[row_index, col_index]`
This is much cleaner than Python's list-of-lists `arr[row][col]`.

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Single Element
print(matrix[0, 0])  # 1 (Top-Left)
print(matrix[1, 2])  # 6 (Row 1, Col 2)

# Specific Row
print(matrix[0, :])  # [1, 2, 3] (Row 0, All columns)

# Specific Column
print(matrix[:, 1])  # [2, 5, 8] (All rows, Col 1)

# Sub-matrix (Cropping)
# Rows 0-1, Cols 1-2
crop = matrix[0:2, 1:3]
# [[2, 3],
#  [5, 6]]
```

---

## 4. Fancy Indexing (Integer Indexing)
Passing a list of indices to select specific items in random order.
```python
arr = np.array([10, 20, 30, 40, 50])

indices = [0, 3, 4]
print(arr[indices])  # [10, 40, 50]
```

---

## 5. Boolean Indexing (Filtering)
The most extensively used feature in data cleaning. You can select elements that satisfy a condition.

```python
data = np.array([1, -5, 10, -2, 20])

# 1. Create a Mask (Boolean Array)
mask = data > 0
print(mask) # [True, False, True, False, True]

# 2. Apply Mask
positive_data = data[mask] 
# OR shorthand: data[data > 0]

print(positive_data) # [1, 10, 20]
```

### AI Context: ReLU Activation
ReLU (Rectified Linear Unit) turns negative numbers into 0.
```python
activations = np.array([1.5, -0.8, 3.2, -1.1])
activations[activations < 0] = 0
print(activations)  # [1.5, 0.0, 3.2, 0.0]
```

---

## 6. Practical Exercises

### Exercise 1: Train/Test Split
Manually split a dataset.
Given `data = np.arange(100)` (numbers 0-99):
1.  Assign the first 80 numbers to `train_data`.
2.  Assign the last 20 numbers to `test_data`.
3.  Print their shapes.

### Exercise 2: Image Patch Extraction
Create a 10x10 matrix with random integers.
Extract the 3x3 square from the very center of this matrix.
*(Hint: Rows 4,5,6 and Cols 4,5,6)*

---

## 7. Summary
- **Slicing returns Views**: Be careful about overwriting data.
- **`[row, col]`**: The syntax for 2D access.
- **`:`**: Stands for "all". `arr[:, 0]` is "all rows, 0th column".
- **Boolean Indexing**: `arr[arr > 0]` is the standard way to filter data.

**Next Up:** **Vectorization**—why loops are banned in NumPy, and how to do math instanly.
