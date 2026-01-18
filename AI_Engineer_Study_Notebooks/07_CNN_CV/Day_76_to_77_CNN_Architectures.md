# Days 76-77: Modern CNN Architectures

## 1. Evolution of CNNs
AI moves fast. Knowing the history helps you choose the right model.

### 1.1 AlexNet (2012)
The one that started the Deep Learning boom.
- 5 Conv Layers.
- Introduced ReLU and Dropout.

### 1.2 VGG (2014)
"Deeper is Better".
- 16 or 19 layers.
- Uses only $3 \times 3$ convolutions.
- **Cons**: Huge number of parameters (138 Million). Slow.

### 1.3 ResNet (2015)
"Skip Connections".
- Solved Vanishing Gradient for very deep networks.
- Allows gradients to flow through a "highway" skipping layers.
- 18, 34, 50, 101, 152 layers.
- **Standard**: ResNet50 is the default choice for most projects today.

### 1.4 EfficientNet (2019)
"Smarter Scaling".
- Optimizes Depth, Width, and Resolution simultaneously.
- Extremely efficient (Fast and Accurate).

---

## 2. Choosing a Model
- **Mobile/Edge**: MobileNet, ShuffleNet (Low latency).
- **General Purpose**: ResNet50.
- **High Accuracy**: EfficientNet-B7, Vision Transformers (ViT).

---

## 3. Practical Exercises

### Exercise 1: Model Surgery
Load `resnet50`. Print `model`.
 Identify the children layers (`layer1`, `layer2`...).
 Notice how the skip connections are implemented (`BasicBlock`).

---

## 4. Summary
- **ResNet**: The workhorse. Skip connections are key.
- **EfficientNet**: The modern state-of-the-art for ConvNets.

**Next Up:** **The CNN Project**—Classifying real-world images.
