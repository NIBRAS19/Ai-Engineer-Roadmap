# Day 24: Calculus - Derivatives

## 1. Introduction
To train an AI, we need to minimize Valid Loss (Error).
Imagine the Loss is a mountain valley. We are at the top (High Error). We want to go to the bottom (Low Error).
**Calculus tells us which way is "Down".**

---

## 2. The Derivative (Slope)
The derivative of a function $f(x)$ tells us the **Rate of Change** at a specific point.
- Positive Derivative: Function is going up.
- Negative Derivative: Function is going down.
- Zero Derivative: We are at a peak or a valley (Flat).

Notation: $f'(x)$ or $\frac{dy}{dx}$.

### Example
$f(x) = x^2$ (Parabola).
Derivative $f'(x) = 2x$.
- At $x=3$, Slope = 6 (Steep slope up).
- At $x=-3$, Slope = -6 (Steep slope down).
- At $x=0$, Slope = 0 (Valley bottom).

---

## 3. Partial Derivatives
AI functions have millions of inputs (weights), not just $x$.
$f(x, y) = x^2 + y^2$.
We can find the slope with respect to **just x**, holding y constant. This is a **Partial Derivative** ($\frac{\partial f}{\partial x}$).

The collection of all partial derivatives is called the **Gradient** ($\nabla$).
$$ \nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] $$

The Gradient vector always points **Uphill** (steepest ascent).
To minimize loss, we go opposite the gradient (**Downhill**).

---

## 4. Chain Rule (The Heart of Backpropagation)

How do we find the derivative of a complex function like $Loss(ReLU(x \cdot w + b))$?
We break it into small chains.

### The Formula
If $y = g(u)$ and $u = h(x)$, then:
$$ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} $$

### 🎯 Real-World Analogy: The Domino Effect
> Imagine a chain of dominoes: **x** knocks over **u**, and **u** knocks over **y**.
> 
> - How hard does **y** fall when **x** falls? 
> - It depends on (1) how hard **u** falls when **x** falls, AND (2) how hard **y** falls when **u** falls.
> - Multiply these effects together = Chain Rule!

### Step-by-Step Example
Let's trace through a simple neuron:
$$L = (ReLU(w \cdot x + b) - y_{true})^2$$

**Layer by layer:**
1. $z = w \cdot x + b$ (Linear)
2. $a = ReLU(z)$ (Activation)  
3. $L = (a - y_{true})^2$ (Loss)

**Backward pass (Chain Rule in action):**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

| Term | Derivative | Meaning |
|:-----|:-----------|:--------|
| $\frac{\partial L}{\partial a}$ | $2(a - y_{true})$ | "How does loss change with activation?" |
| $\frac{\partial a}{\partial z}$ | $1$ if $z > 0$, else $0$ | "ReLU gradient" |
| $\frac{\partial z}{\partial w}$ | $x$ | "How does z change with weight?" |

**Final answer:**
$$\frac{\partial L}{\partial w} = 2(a - y_{true}) \cdot \mathbb{1}_{z>0} \cdot x$$

This is **Backpropagation**. We multiply local derivatives to find the global impact of a weight on the loss.

### Why This Matters for Day 59 (Autograd)
PyTorch's `loss.backward()` does exactly this—it walks backward through the computation graph, applying the chain rule at each step automatically.

```python
# PyTorch does this for you:
loss.backward()  # Computes ALL gradients using Chain Rule
print(w.grad)    # The result: dL/dw
```

---

## 5. Practical Exercises

### Exercise 1: Numerical Derivative
Calculus is smooth, but computers are discrete. approximating slope:
$$ f'(x) \approx \frac{f(x + h) - f(x)}{h} $$
Write a python function to estimate the derivative of $f(x) = x^3$ at $x=4$ with $h=0.0001$. Compare it to the exact answer ($3x^2 = 3(16) = 48$).

---

## 6. Summary
- **Derivative**: Use to find the slope.
- **Gradient**: Vector of partial derivatives (Direction of steepest ascent).
- **Minimization**: Walking opposite to the gradient.

**Next Up:** **Gradient Descent**—The algorithm that uses derivatives to train models.
