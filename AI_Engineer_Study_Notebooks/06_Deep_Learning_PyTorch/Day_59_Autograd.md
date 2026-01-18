# Day 59: Autograd (Automatic Differentiation)

## 1. Introduction

Do you remember calculating derivatives by hand in Calculus (Day 24)?
PyTorch does it for you—**automatically**.

It records every operation performed on a tensor to build a **Computation Graph**.
Then, calling `.backward()` calculates gradients using the **Chain Rule** (Day 24).

### 🔗 Connection to Day 25: Gradient Descent

Remember the gradient descent formula?
$$ W_{new} = W_{old} - \alpha \cdot \nabla Loss(W) $$

In PyTorch:
- `loss.backward()` computes $\nabla Loss(W)$ (the gradient)
- `optimizer.step()` performs the update

**This is gradient descent in action!** Every training step you've seen is this formula.

---

## 2. The Computation Graph

### 🎯 Real-World Analogy: The Domino Trail
> Imagine setting up a trail of dominoes. Each domino (operation) is placed one after another: input → linear → activation → loss. When you call `.backward()`, it's like tipping the last domino—the chain reaction travels **backward**, calculating how each domino (weight) affected the final result.

```
Forward Pass (Left to Right):
   x → [Linear] → z → [ReLU] → a → [Loss] → L

Backward Pass (Right to Left):
   dL/dx ← dL/dz ← dL/da ← dL/dL (=1)
```

### Visualizing the Graph

```python
import torch

# Create a tensor requiring gradient
w = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0])  # Data (no grad needed)

# Forward pass builds the graph
y = w * x            # MulBackward
loss = (y - 5)**2    # PowBackward, SubBackward

print(loss)  # tensor([9.], grad_fn=<PowBackward0>)
# The grad_fn shows the graph is tracking operations!
```

---

## 3. `requires_grad=True`

This flag tells PyTorch: "Track everything that happens to this variable".

| Tensor Type | `requires_grad` | Why |
|:------------|:----------------|:----|
| Model weights | `True` | We need to update them |
| Input data (X) | `False` | We don't train the data |
| Targets (y) | `False` | Labels are fixed |

```python
import torch

w = torch.tensor([1.0], requires_grad=True)  # Weight
x = torch.tensor([2.0])                       # Data (no grad)

# Forward pass (y = w * x)
y = w * x 

# Loss function (L = (y - 5)^2)
loss = (y - 5)**2

print(loss)  # tensor([9.], grad_fn=<PowBackward0>)
```

---

## 4. Backward Pass

Compute gradients ($\frac{\partial Loss}{\partial w}$).

```python
loss.backward()

# The gradient is stored in .grad attribute
print(w.grad) 
# dL/dw = 2 * (y - 5) * x
#       = 2 * (1*2 - 5) * 2 = 2 * (-3) * 2 = -12
```

### Step-by-Step: How `.backward()` Works

Using Chain Rule from Day 24:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$

| Step | Expression | Value |
|:-----|:-----------|:------|
| $L = (y-5)^2$ | $\frac{\partial L}{\partial y} = 2(y-5)$ | $2(2-5) = -6$ |
| $y = w \cdot x$ | $\frac{\partial y}{\partial w} = x$ | $2$ |
| Chain Rule | $\frac{\partial L}{\partial w} = -6 \times 2$ | $-12$ ✓ |

---

## 5. Putting It Together: Manual Gradient Descent

```python
import torch

# Initialize weight
w = torch.tensor([10.0], requires_grad=True)
x = torch.tensor([2.0])
target = 5.0
lr = 0.1

# Training loop (manual gradient descent)
for i in range(20):
    # Forward
    y = w * x
    loss = (y - target)**2
    
    # Backward (compute gradient)
    loss.backward()
    
    # Update weight (gradient descent!)
    with torch.no_grad():  # Don't track this operation
        w -= lr * w.grad
    
    # Clear gradient for next iteration
    w.grad.zero_()
    
    print(f"Iter {i+1}: w={w.item():.4f}, loss={loss.item():.4f}")

# w should converge to 2.5 (because 2.5 * 2 = 5)
```

---

## 6. `torch.no_grad()`

When testing/evaluating, we don't need gradients (saves RAM and computation).

```python
model.eval()  # Put model in evaluation mode

with torch.no_grad():
    y_pred = model(x_test)
    # No computation graph is built
    # Memory efficient for inference
```

### When to Use:
| Situation | Use `no_grad()`? |
|:----------|:-----------------|
| Training | ❌ No |
| Validation | ✅ Yes |
| Inference/Prediction | ✅ Yes |
| Updating weights manually | ✅ Yes |

---

## 7. Common Mistakes

### Mistake 1: Forgetting to Zero Gradients
```python
# WRONG: Gradients accumulate!
for epoch in range(10):
    loss.backward()  # Gradient keeps adding up!

# CORRECT:
for epoch in range(10):
    optimizer.zero_grad()  # Clear first
    loss.backward()
```

### Mistake 2: Calling `.backward()` Twice
```python
loss.backward()
loss.backward()  # ERROR! Graph was freed

# FIX: If you need to call twice, retain the graph
loss.backward(retain_graph=True)
loss.backward()
```

---

## 8. Preview: Gradient Problems (Day 60-61)

### Vanishing Gradients
When gradients become too small (→0), early layers don't learn.
- **Cause**: Deep networks + saturating activations (Sigmoid)
- **Solution**: ReLU, proper initialization, ResNets

### Exploding Gradients
When gradients become too large (→∞), training explodes.
- **Cause**: Deep networks + bad initialization
- **Solution**: Gradient clipping, proper initialization

```python
# Gradient Clipping (prevents explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9. Practical Exercises

### Exercise 1: Manual Gradient Check
1. Create a simple function $f(x) = x^3$ at $x=2$.
2. Calculate the gradient manually: $f'(x) = 3x^2 = 12$.
3. Verify with PyTorch autograd.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**3
y.backward()
print(x.grad)  # Should be 12.0
```

### Exercise 2: Visualize the Computation Graph
Use `torchviz` to visualize a simple network's computation graph.

```python
# pip install torchviz
from torchviz import make_dot

output = model(input_tensor)
make_dot(output, params=dict(model.named_parameters())).render("graph", format="png")
```

---

## 10. Summary

| Concept | What It Does | Day Connection |
|:--------|:-------------|:---------------|
| **requires_grad** | Tells PyTorch to track operations | - |
| **backward()** | Computes gradients via chain rule | Day 24 (Chain Rule) |
| **optimizer.step()** | Updates weights using gradients | Day 25 (Gradient Descent) |
| **no_grad()** | Disables tracking for inference | - |

**The Training Loop Formula:**
```
forward() → loss() → backward() → step() → zero_grad()
    ↓          ↓          ↓           ↓           ↓
  Predict   Compare   Gradients   Update    Clear
```

**Next Up:** **Neural Network Basics**—Building layers with proper initialization.

