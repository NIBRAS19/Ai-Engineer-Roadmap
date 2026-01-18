# Day 71: Pooling and Batch Normalization

## 1. Pooling Layers (`nn.MaxPool2d`)
We need to reduce image size to (1) reduce computation and (2) force the model to look at "the big picture" (Translation Invariance).

### Max Pooling
Takes the **Maximum** value in a window.
- Size $2 \times 2$, Stride 2.
- Halves the Height and Width.
- **Concept**: Keeps the strongest feature (e.g., "There is a beak here"), discards location precision.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input (16, 64, 64) -> Output (16, 32, 32)
```

### 🧠 Modern Debate: Pooling vs. Stride=2
In older nets (VGG, AlexNet), we used MaxPool to downsample.
In modern nets (ResNet, YOLO), we often use **Convolution with Stride=2** instead.

| Method | Pros | Cons |
|:-------|:-----|:-----|
| **Max Pooling** | Simple, no parameters to learn | Discards information completely |
| **Conv Stride=2** | Learns *how* to downsample | Adds parameters (weights) |

**Verdict**: Most modern architectures prefer Strided Convolutions or Adaptive Pooling.

---

## 2. Batch Normalization (`nn.BatchNorm2d`)
Neural Networks hate it when the scale of data changes deep inside the network (Internal Covariate Shift).
**BatchNorm** re-centers the data to Mean=0, Std=1 inside the network.
- **Effect**: Converges much faster, allows higher learning rates.
- **Placement**: Usually `Conv -> BatchNorm -> ReLU`.

```python
bn = nn.BatchNorm2d(num_features=16) # Must match matches output channels of previous Conv
```

---

## 3. Practical Exercises

### Exercise 1: Calculate Dimensions
Input: $32 \times 32$.
1.  Apply Conv2d(kernel=3, stride=1, padding=1). Size?
2.  Apply MaxPool2d(kernel=2). Size?
*(Ans: 32x32 -> 32x32 -> 16x16)*

---

## 4. Summary
- **Pooling**: Downsampling.
- **BatchNorm**: Stabilizing training.

**Next Up:** **MNIST CNN**—Building our first Vision Model.
