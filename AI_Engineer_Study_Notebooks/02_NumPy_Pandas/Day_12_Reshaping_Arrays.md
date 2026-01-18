# Day 12: Reshaping Arrays

## 1. Introduction
Data rarely comes in the shape your model expects.
- Images come as (Height, Width). Models want (Batch, Height, Width).
- Data comes as a 3D volume. You need to "flatten" it to a 1D vector for a linear classifier.

NumPy gives you tools to massage data shape without copying the underlying data.

---

## 2. `.reshape()`
Changes dimensions as long as the **total number of elements** remains the same.

```python
import numpy as np

# A list of 12 numbers (1D)
arr = np.arange(12) 
# [0, 1, 2, ..., 11]

# Reshape to Matrix (3x4)
grid = arr.reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Reshape to Cube (2x2x3)
cube = arr.reshape(2, 2, 3)
```

### The Magic `-1` Dimension
If you know one dimension but are too lazy to calculate the other, use `-1`. NumPy figures it out.

```python
arr = np.arange(12)
reshaped = arr.reshape(4, -1) 
# We asked for 4 rows. NumPy calculates cols = 12/4 = 3.
# Result shape is (4, 3)
```

---

## 3. Flattening Data
Converting a multi-dimensional matrix into a 1D vector.
Essential before feeding images into a Dense (Fully Connected) Neural Network layer.

```python
matrix = np.array([[1, 2], [3, 4]])

# Method 1: .flatten() (Returns a COPY)
flat_copy = matrix.flatten()

# Method 2: .ravel() (Returns a VIEW - changes affect original)
# Faster, memory efficient. Use this unless you need a copy.
flat_view = matrix.ravel()
```

---

## 4. Expanding Dimensions
Sometimes a model expects a 4D input `(Batch, Channel, Height, Width)`, but you only have a single image `(Channel, Height, Width)`.
You need to add a "fake" batch dimension.

```python
img = np.zeros((3, 224, 224)) # Shape (3, 224, 224)

# Method 1: newaxis
img_batch = img[np.newaxis, :, :, :]
print(img_batch.shape) # (1, 3, 224, 224)

# Method 2: expand_dims
img_batch_2 = np.expand_dims(img, axis=0) # Add dimension at index 0
```

---

## 5. Squeeze
Removes dimensions of size 1.
```python
# Prediction result might be [[0.8]] (Shape (1, 1))
pred = np.array([[0.8]])

scalar = pred.squeeze() # Shape () -> just 0.8
print(scalar)
```

---

## 6. Practical Exercises

### Exercise 1: Image Flattening
You have a batch of 10 grayscale images, each 28x28. Shape: `(10, 28, 28)`.
Reshape this array into `(10, 784)`. This flattens each image into a row vector (28*28=784), but keeps the batch size intact.

### Exercise 2: Dimension Juggling
Create an array of shape `(2, 3)`.
1.  Add a dimension to make it `(2, 3, 1)`.
2.  Add a dimension to make it `(1, 2, 3)`.

---

## 7. Summary
- **`.reshape(rows, cols)`**: Change shape. Elements count must match.
- **`-1`**: "Calculate this dimension for me".
- **`.ravel()`**: Flatten to 1D efficiently.
- **`newaxis / expand_dims`**: Add dimensions (e.g., for batching).

**Next Up:** **Statistical Functions**—Min, Max, Mean, and Standard Deviation.
