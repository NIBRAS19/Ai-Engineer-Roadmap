# Day 60: Neural Network Layers

## 1. Introduction
We don't manually do `w * x + b` anymore.
PyTorch provides `torch.nn`, which contains pre-built layers.

---

## 2. `nn.Module`
Every Neural Network in PyTorch is a **Class** that inherits from `nn.Module`.
It must have:
1.  `__init__`: Define the layers.
2.  `forward()`: Define how data flows through layers.

## 3. The Linear Layer (`nn.Linear`)
Fully Connected Layer (Dense).
$$ y = xA^T + b $$

```python
import torch.nn as nn

# Input: 10 features, Output: 5 features
fc = nn.Linear(in_features=10, out_features=5)

# Checking weights
print(fc.weight.shape) # (5, 10)
print(fc.bias.shape)   # (5,)
```

---

## 4. Building a Simple Net

```python
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128) # Hidden Layer
        self.fc2 = nn.Linear(128, 10)  # Output Layer (10 classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # Activation
        x = self.fc2(x)
        return x

model = SimpleNet()
print(model)
```

---

## 5. Summary
- **Subclass `nn.Module`**.
- **Define layers in `__init__`**.
- **Connect them in `forward`**.

**Next Up:** **Activation Functions**—Adding non-linearity.
