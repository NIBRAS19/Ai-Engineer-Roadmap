# Day 64: The Training Loop

## 1. Introduction
Unlike Scikit-Learn (`model.fit()`), PyTorch requires you to write the training loop manually.
This gives you total control.

---

## 2. The Blueprint
For every epoch:
  For every batch of data:
    1. **Forward Pass**: Predict.
    2. **Calculate Loss**: Compare with target.
    3. **Backward Pass**: Calculate gradients.
    4. **Optimizer Step**: Update weights.
    5. **Zero Grads**: Reset for next batch.

---

## 3. Implementation

```python
epochs = 5

for epoch in range(epochs):
    # Training Mode (Important for Dropout/BatchNorm)
    model.train()
    
    for X_batch, y_batch in train_loader:
        # 1. Forward
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        
        # 2. Backward
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Compute new gradients
        optimizer.step()      # Update weights
        
    print(f"Epoch {epoch+1} Complete. Loss: {loss.item()}")
```

---

## 4. Summary
- **Zero Grad**: Crucial! Otherwise gradients accumulate.
- **Model State**: `model.train()` vs `model.eval()`.

**Next Up:** **DataLoaders**—Feeding data efficiently.
