# Day 72: Building a CNN for MNIST

## 1. Goal
Classify handwritten digits (0-9).
Images are grayscale $28 \times 28$.

---

## 2. The Architecture
Pattern: `Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC`.

```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classifier
        # Input: 28x28 -> Pool -> 14x14 -> Pool -> 7x7
        # Channels: 1 -> 32 -> 64
        # Flatten Size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Check shape before flattening!
        # print(x.shape) 
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 3. Training
Use `CrossEntropyLoss` and `Adam`.
You should achieve >99% accuracy easily.

---

## 4. Summary
- **Architecture**: Stacking Conv blocks increases depth and abstraction.
- **Flattening**: The bridge between 2D Conv world and 1D Dense world.

**Next Up:** **Data Augmentation**—Getting more data for free.
