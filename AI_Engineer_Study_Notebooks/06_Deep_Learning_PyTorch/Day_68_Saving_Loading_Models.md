# Day 68: Saving and Loading Models

## 1. Introduction
After training for 48 hours, you MUST save your model.
In PyTorch, we usually save the **State Dictionary** (weights only), not the whole object (code + weights).

---

## 2. Saving (`state_dict`)

```python
# Best Practice: Save only weights
torch.save(model.state_dict(), "model.pth")
```

## 3. Loading
You must first re-create the architecture (code), then load the weights into it.

```python
# 1. Re-create architecture
model = SimpleNet() 

# 2. Load weights
model.load_state_dict(torch.load("model.pth"))

# 3. Set mode
model.eval() # Essential for inference!
```

---

## 4. Checkpointing
Saving more than just the model (e.g., Optimizer state, Epoch number) so you can resume training later.

```python
checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.45
}
torch.save(checkpoint, "checkpoint.pth")
```

---

## 5. Summary
- **`.pth` or `.pt`**: Standard extensions.
- **`state_dict`**: A dictionary mapping layer names to weight tensors.

**CONGRATULATIONS!** You have finished **Weeks 9-10: Deep Learning with PyTorch**.
You now possess the power to build Neural Networks.
**Next Week:** **Computer Vision (CNNs)**—Teaching computers to see images.
