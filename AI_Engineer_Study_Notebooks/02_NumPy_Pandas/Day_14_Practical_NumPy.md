# Day 14: Practical NumPy for AI

## 1. Introduction
Congratulations on finishing the NumPy week!
Today, we combine everything (Arrays, Random, Dot Product, Broadcasting, Indexing, Stats) to build something real.
We will **simulate the Forward Pass of a Neural Network** from scratch. No PyTorch, no TensorFlow. Just math.

---

## 2. Theoretical Setup
A simple Neural Network layer does this:
$$ \text{Output} = \text{ReLU}(\text{Input} \cdot \text{Weights} + \text{Bias}) $$

- **Input**: A batch of data (e.g., 5 samples, 3 features each).
- **Weights**: Connections between neurons.
- **Bias**: An offset.
- **ReLU**: Activation function (Output 0 if negative).

---

## 3. The Implementation

### Step 1: Initialize Data
```python
import numpy as np

# Reproducibility
np.random.seed(42)

# Batch of 5 samples, 3 features (e.g., House Price, Bedrooms, Age)
X = np.random.randn(5, 3) 
print("Input Shape:", X.shape) # (5, 3)
```

### Step 2: Initialize Weights and Biases
We want to transform this into 4 features (Hidden neurons).
- Weights shape must typically be `(Input_Features, Output_Features)` -> `(3, 4)`.
- Bias shape corresponds to `(Output_Features,)` -> `(4,)`.

```python
W = np.random.randn(3, 4) 
b = np.zeros((4,))

print("Weights Shape:", W.shape) #(3, 4)
print("Bias Shape:", b.shape)    #(4,)
```

### Step 3: Linear Transformation (Dot Product)
$$ Z = X \cdot W + b $$

```python
# Dot Product (5,3) @ (3,4) -> (5,4)
# Broadcasting adds bias (4,) to every row of (5,4)
Z = np.dot(X, W) + b

print("Linear Output Shape:", Z.shape) # (5, 4)
print("First sample linear output:", Z[0])
```

### Step 4: Activation Function (ReLU)
Turn negatives to zero.

```python
# Boolean masking / np.maximum
A = np.maximum(0, Z)

print("\nFinal Activity (after ReLU):")
print(A)
```

### Step 5: Pooling (Simulated)
Let's say we want to take the 'max' activation for each sample to get a single score (Global Max Pooling).

```python
scores = np.max(A, axis=1) # Max across columns
print("\nFinal Scores per sample:", scores)
```

---

## 4. Advanced: Data Normalization
AI models fail if inputs differ wildly in range (e.g., Price=100000 vs Age=50). We normalize them.

```python
raw_data = np.array([
    [100, 50000],
    [150, 75000],
    [120, 60000]
])

# Standardization: (X - Mean) / Std
mean = np.mean(raw_data, axis=0)
std = np.std(raw_data, axis=0)

normalized_data = (raw_data - mean) / std

print("Means (should be ~0):", np.mean(normalized_data, axis=0))
print("Stds (should be ~1):", np.std(normalized_data, axis=0))
```

---

## 5. Weekly Challenge: Softmax Function
The Softmax function converts raw scores into probabilities.
Formula: 
$$ \sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}} $$

**Task:**
1.  Create a vector `logits = [2.0, 1.0, 0.1]`.
2.  Compute `exp_logits = np.exp(logits)`.
3.  Compute `sum_exp_logits = np.sum(exp_logits)`.
4.  Compute probabilities: `probs = exp_logits / sum_exp_logits`.
5.  Verify they sum to 1.

---

## 6. Conclusion
You have effectively built the mathematical engine of a neuron layer.
- **PyTorch `nn.Linear`** just wraps `X @ W + b`.
- **PyTorch `nn.ReLU`** just wraps `np.maximum(0, Z)`.
- **Normalization** is `StandardScaler` in Scikit-Learn.

Next week, we move from **Arrays** to **Tables**. We will handle real-world labeled datasets using **Pandas**.
