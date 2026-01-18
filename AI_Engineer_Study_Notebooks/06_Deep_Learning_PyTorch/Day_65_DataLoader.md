# Day 65: Dataset and DataLoader

## 1. Introduction
How do you handle 100GB of images? You can't load them all into RAM.
PyTorch splits data handling into two parts:
1.  **Dataset**: Knows how to read 1 item (e.g., from disk).
2.  **DataLoader**: Batches, Shuffles, and loads items in parallel using CPU workers.

---

## 2. Custom Dataset
Must implement `__len__` and `__getitem__`.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

## 3. DataLoader
The Magic Manager.
```python
dataset = CustomDataset(X_train, y_train)

loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,  # Shuffle training data!
    num_workers=2  # Parallel loading
)

# Usage
for batch_X, batch_y in loader:
    pass
```

---

## 4. Summary
- **Dataset**: Stores samples.
- **DataLoader**: Batches samples efficiently.

**Next Up:** **Full Pipeline**—Building a complete model.
