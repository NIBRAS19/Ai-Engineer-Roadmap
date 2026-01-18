# Days 66-67: Complete Training Pipeline

## 1. Goal
Combine everything: Data -> Model -> Loss -> Optimizer -> Loop -> Evaluation.

---

## 2. Full Code Template (Save this!)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Data Setup
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,)) # Binary labels

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2) # 2 Output classes
)

# 3. Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Loop
for epoch in range(10):
    for X, y in loader:
        # Forward
        preds = model(X)
        loss = criterion(preds, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
print("Training Complete!")
```

---

## 3. Evaluation Loop
```python
model.eval() # Stop Dropout, BatchNorm updates
with torch.no_grad(): # Disable Gradient tracking (Saving Memory)
    dataset_test = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
    test_loader = DataLoader(dataset_test, batch_size=10)
    
    correct = 0
    total = 0
    for X, y in test_loader:
        preds = model(X)
        predicted_classes = torch.argmax(preds, dim=1)
        correct += (predicted_classes == y).sum().item()
        total += y.size(0)
        
print(f"Accuracy: {correct/total}")
```

---

## 4. Summary
This template works for 90% of Deep Learning tasks.
Only the **Data** and the **Model Architecture** change. The loop stays the same.

**Next Up:** **Saving & Loading**—Keeping your work.
