# Day 70: Convolutional Layers (`nn.Conv2d`)

## 1. Introduction: How Computers See
We don't feed the whole image at once. We scan it for features.

### 🎯 Real-World Analogy: The Magnifying Glass
> A CNN filter is like looking at a photo, but through a small **magnifying glass** that only shows one small square at a time. You slide this glass across the entire image.
> 
> At each position, you're checking: "**Does this small patch look like an edge? A curve? A texture?**"
> 
> - The filter is trained to recognize specific patterns (e.g., vertical lines).
> - Just like how your brain automatically spots faces in random patterns (pareidolia), the CNN spots features everywhere.

---

## 2. The Math: Cross-Correlation
We take a $3 \times 3$ Filter (Kernel) and slide it over the Input Image.
At every step, we multiply overlapping pixels and sum them up.
$$ Output[i,j] = \sum (Input\_Patch \times Filter) $$

---

## 3. Key Hyperparameters
1.  **Kernel Size**: Size of the filter (Usually $3 \times 3$ or $5 \times 5$).
2.  **Stride**: How many pixels we move per step. (Stride 1 = smooth, Stride 2 = halves the size).
3.  **Padding**: Adding zeros around the border to keep output size same as input size.

### 🧠 Concept: Receptive Field
The **Receptive Field** is the part of the original image that a specific neuron "sees".
- In the first layer, a neuron sees only $3 \times 3$ pixels.
- In deeper layers, acceptable field grows! A neuron in Layer 5 might "see" a $50 \times 50$ patch because it combines info from previous layers.
- **Why it matters**: To detect large objects (like a car), you need a deep network with a large receptive field.

---

## 4. Implementation in PyTorch

```python
import torch
import torch.nn as nn

# Input: Batch=1, Channels=3 (RGB), Height=64, Width=64
image = torch.randn(1, 3, 64, 64)

# Layer: 
# in_channels=3 (RGB)
# out_channels=16 (Number of filters/features to learn)
# kernel_size=3
# padding=1 (keeps size 64x64)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

output = conv(image)
print(output.shape) 
# torch.Size([1, 16, 64, 64]) -> 16 Feature Maps
```

---

## 5. Summary
- **Filters**: Learnable weights that act like magnifying glass patterns.
- **Feature Maps**: The output "activations" showing where features were found.
- **Receptive Field**: How much closer context a neuron has.

**Next Up:** **Pooling and Normalization**—Shrinking the data.

