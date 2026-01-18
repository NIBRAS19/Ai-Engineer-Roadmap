# Day 61: Activation Functions

## 1. Introduction
Without Activation Functions, deep neural networks are just a single Linear Regression.
Non-linearity allows models to learn complex patterns (Curves, Shapes).

---

## 2. Common Functions

### 2.1 ReLU (Rectified Linear Unit)
$$ f(x) = max(0, x) $$
- **Default check**: Use this for hidden layers.
- Fast, simple, solves vanishing gradient problem.

### 2.2 Sigmoid
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
- Output: (0, 1).
- Use case: **Binary Classification** (Output Layer).

### 2.3 Softmax
- Output: Probabilities summing to 1.
- Use case: **Multi-Class Classification** (Output Layer).

---

## 3. Implementation

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-1.0, 0.0, 1.0])

# ReLU
print(F.relu(x)) # [0., 0., 1.]

# Sigmoid
print(torch.sigmoid(x)) 

# Softmax (Across dimension)
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(probs) # Sums to 1.0
```

---

## 4. Summary
- **Hidden Layers**: Use ReLU.
- **Output Layer**: Use Sigmoid (Binary) or Softmax (Multi).

**Next Up:** **Loss Functions**—Which error metric to use.
