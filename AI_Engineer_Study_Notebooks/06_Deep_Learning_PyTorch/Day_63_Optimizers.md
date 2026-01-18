# Day 63: Optimizers (SGD & Adam)

## 1. Introduction
The Optimizer performs the weight update:
$$ W = W - \alpha \nabla L $$
PyTorch handles this in `torch.optim`.

---

## 2. SGD (Stochastic Gradient Descent)
The classic. Simple, robust, but sometimes slow convergence.
Momentum helps it speed through flat areas.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## 3. Adam (Adaptive Moment Estimation)
The standard choice for most Deep Learning tasks.
It adjusts the learning rate for each parameter individually.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 4. The Step Function
Once per batch, you call:
1.  `optimizer.zero_grad()`: Clear old gradients.
2.  `loss.backward()`: Calculate new gradients.
3.  `optimizer.step()`: Update weights.

---

## 5. Summary
- **Start with Adam**: Learning rate 3e-4 (0.0003) is a legendary default ("Karpathy Constant").
- **SGD**: Use if Adam fails or for simple problems.

**Next Up:** **The Training Loop**—Putting it all together.
