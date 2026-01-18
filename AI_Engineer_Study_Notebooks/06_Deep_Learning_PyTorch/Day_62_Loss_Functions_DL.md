# Day 62: Loss Functions in PyTorch

## 1. Introduction
PyTorch optimizes models by minimizing the Loss.
The loss function takes (prediction, target) and returns a scalar error.

---

## 2. Regression Losses
Predicting a number.

### MSELoss (Mean Squared Error)
```python
criterion = nn.MSELoss()
loss = criterion(pred, target)
```

## 3. Classification Losses

### CrossEntropyLoss
Used for **Multi-Class** classification.
**CRITICAL NOTE**: In PyTorch, `nn.CrossEntropyLoss` combines `LogSoftmax` + `NLLLoss`.
This means: **Do NOT put a Softmax layer at the end of your network if you use this loss.** Pass raw logits.

```python
criterion = nn.CrossEntropyLoss()
# pred: (Batch, Num_Classes), target: (Batch, ) Class Indices
loss = criterion(pred_logits, target_indices)
```

### BCELoss (Binary Cross Entropy)
Used for Binary Classification.
Requires Sigmoid activation beforehand.

---

## 4. Summary
- **Regression**: `MSELoss`.
- **Classification**: `CrossEntropyLoss` (Raw logits).
- **Binary**: `BCEWithLogitsLoss` (Stable) or `BCELoss`.

**Next Up:** **Optimizers**—SGD and Adam.
