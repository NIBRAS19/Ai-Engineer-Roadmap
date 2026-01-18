# Day 58: Tensors

## 1. Introduction
A **Tensor** is a multi-dimensional matrix containing elements of a single data type.
It is almost identical to a NumPy array, but with two superpowers:
1.  **Runs on GPU**.
2.  **Tracks Gradients** (for backpropagation).

---

## 2. Creating Tensors
```python
import torch
import numpy as np

# From list
x = torch.tensor([[1, 2], [3, 4]])

# From NumPy
np_arr = np.array([1, 2, 3])
x_np = torch.from_numpy(np_arr)

# Random / Zeros / Ones
x_rand = torch.rand(2, 2)
x_zeros = torch.zeros(2, 2)
```

---

## 3. Operations
Standard math applies.

```python
x = torch.ones(2, 2)
y = torch.ones(2, 2) * 2

print(x + y) # Addition
print(x * y) # Element-wise multiplication
print(x @ y) # Matrix Multiplication (Dot Product)
```

---

## 4. Reshaping (`view` vs `reshape`)
In PyTorch, we prefer `.view()`.

```python
x = torch.rand(4, 4)
y = x.view(16)      # Flatten
z = x.view(-1, 8)   # (2, 8)
```

---

## 5. Summary
- **Tensor**: GPU-ready NumPy array.
- **`.to(device)`**: Moves tensor to GPU/CPU.
- **`.item()`**: Extracts the value from a single-element tensor.

**Next Up:** **Autograd**—Magic Calculus.
