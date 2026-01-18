# Day 25: Gradient Descent

## 1. Introduction
**Gradient Descent** is the optimization algorithm used to train 99% of modern AI models.
It answers: "How should I change the weights to reduce the error?"

Formula:
$$ W_{new} = W_{old} - \alpha \cdot \nabla Loss(W) $$
- $\alpha$: **Learning Rate** (Step size).
- $\nabla Loss$: Gradient (Slope).

### 🎯 Real-World Analogy: The Blindfolded Golfer
> Imagine you're blindfolded on a hilly golf course and need to reach the lowest point (the hole). You can only **feel the slope under your feet**. **Gradient Descent** is taking tiny steps *downhill*. 
> 
> - If you take **huge steps** (high learning rate), you might overshoot and end up on the other side of the hill.
> - If you take **tiny steps** (low learning rate), you'll get there... eventually... maybe after 1000 years.
> - The gradient tells you which direction is "downhill" and how steep it is.

---

## 2. The Algorithm Step-by-Step
1.  **Initialize** weights randomly.
2.  **Forward Pass**: Compute prediction and Loss.
3.  **Backward Pass**: Calculate Gradient of Loss w.r.t weights (using Chain Rule from Day 24).
4.  **Update**: Move weights opposite to gradient.
5.  **Repeat** until convergence.

### Visualization: What Happens Each Step
```
Iteration 1:  Loss = 10.0   →  Gradient = -2.5   →  Weights shift right
Iteration 2:  Loss = 7.2    →  Gradient = -1.8   →  Weights shift right (smaller)
Iteration 3:  Loss = 5.1    →  Gradient = -1.1   →  Weights shift right (smaller)
...
Iteration 50: Loss = 0.01   →  Gradient ≈ 0      →  CONVERGED!
```

---

## 3. Learning Rate ($\alpha$)
The **most important hyperparameter** in deep learning.

| Learning Rate | Effect | Symptom |
|:--------------|:-------|:--------|
| Too Small (0.0001) | Takes forever to converge | Loss decreases very slowly |
| Too Large (1.0) | Overshoots, never converges | Loss oscillates or explodes (NaN) |
| Just Right (0.001-0.01) | Smooth convergence | Loss steadily decreases |

### 🚨 Common Mistake: "My Loss is NaN!"
This almost always means your learning rate is too high. The gradient update overshoots so badly that the numbers explode to infinity.

**Fix**: Reduce learning rate by 10x (e.g., 0.01 → 0.001).

---

## 4. Types of Gradient Descent

### 4.1 Batch Gradient Descent
Use **ALL** data to calculate one step.
- Pros: Accurate gradient, stable.
- Cons: Too slow for big data. Memory intensive. (Can't fit 1M images in RAM)

### 4.2 Stochastic Gradient Descent (SGD)
Use **ONE** random sample to calculate step.
- Pros: Fast updates, can escape local minima due to noise.
- Cons: Noisy (zig-zag path, may never converge exactly).

### 4.3 Mini-Batch Gradient Descent (The Standard)
Use a **Batch** (e.g., 32 or 64 samples).
- Best of both worlds: Stable enough, fast enough.
- This is what PyTorch/TensorFlow actually do.

---

## 5. Local Minima vs Global Minima

### 🎯 Analogy: Valleys in a Mountain Range
> The golf course isn't smooth—it has multiple small dips (local minima) and one deepest hole (global minimum). Gradient descent can get "stuck" in a local dip, thinking it's found the lowest point.

### Why This Matters in Deep Learning
- Neural network loss landscapes have **millions of dimensions**.
- Good news: In high dimensions, most "dips" are saddle points, not true minima.
- SGD's noise actually helps escape saddle points.

```
                 Local           Global
                 Minimum         Minimum
        \       /       \       /
         \     /         \     /
          \   /           \   /
           \ /             \_/
            ▼               ▼
         (stuck!)        (goal!)
```

---

## 6. Beyond Basic Gradient Descent: Momentum

### Problem with Vanilla SGD
Imagine a long, narrow valley. The gradient points mostly side-to-side (high curvature) and only slightly down the valley (low curvature). SGD zig-zags back and forth slowly.

### Solution: Add Momentum
Like a ball rolling downhill—it builds up speed and ignores small bumps.

$$ v_t = \beta \cdot v_{t-1} + \nabla Loss $$
$$ W_{new} = W_{old} - \alpha \cdot v_t $$

Where $\beta \approx 0.9$ is the momentum coefficient.

### Modern Optimizers (Preview for Day 63)
| Optimizer | Key Idea |
|:----------|:---------|
| **SGD + Momentum** | Accumulates velocity |
| **Adam** | Adaptive learning rate per parameter |
| **AdamW** | Adam + weight decay (best default today) |

---

## 7. Connection to PyTorch (Day 59-64 Preview)

Here's what gradient descent looks like in actual PyTorch code:

```python
import torch
import torch.nn as nn

# Model, Loss, Optimizer
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # <- Learning Rate here!

# Training Loop (This IS Gradient Descent!)
for epoch in range(100):
    # 1. Forward Pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # 2. Backward Pass (Compute Gradients via Chain Rule)
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute gradients
    
    # 3. Update Weights (The actual Gradient Descent step!)
    optimizer.step()       # W = W - lr * gradient
```

Every `optimizer.step()` is one iteration of gradient descent.

---

## 8. Practical Exercises

### Exercise 1: Gradient Descent Simulation
Find the minimum of $f(x) = (x - 3)^2$ starting at $x=10$.
- Derivative $f'(x) = 2(x-3)$.
- Learning rate $\alpha = 0.1$.
- Run a loop for 20 iterations: `x = x - lr * gradient`.
- Does x converge to 3?

```python
x = 10
lr = 0.1
for i in range(20):
    gradient = 2 * (x - 3)
    x = x - lr * gradient
    print(f"Iter {i+1}: x = {x:.4f}, f(x) = {(x-3)**2:.4f}")
```

### Exercise 2: Learning Rate Exploration
Modify Exercise 1:
1. Try `lr = 0.01` — Does it converge? How many iterations?
2. Try `lr = 1.0` — What happens? (Hint: It should oscillate or diverge)
3. Try `lr = 1.1` — Does it explode?

### Exercise 3: Momentum Intuition
Add momentum to Exercise 1:
```python
x = 10
lr = 0.1
beta = 0.9
velocity = 0

for i in range(20):
    gradient = 2 * (x - 3)
    velocity = beta * velocity + gradient
    x = x - lr * velocity
    print(f"Iter {i+1}: x = {x:.4f}")
```
Does it converge faster than without momentum?

---

## 9. Summary
- **Gradient Descent**: Iteratively improving parameters by walking downhill.
- **Learning Rate**: The most important hyperparameter. Too high = explosion, too low = slow.
- **Mini-Batch**: The practical implementation used in PyTorch/TensorFlow.
- **Momentum**: Helps escape saddle points and smooth out zig-zags.
- **PyTorch**: `optimizer.step()` IS gradient descent.

**Connection Note**: This is the algorithm. Day 59 (Autograd) shows how PyTorch computes the gradients automatically using the Chain Rule from Day 24.

**Next Up:** **Statistics Basics**—Describing data distributions.

