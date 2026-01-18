# Day 73: Data Augmentation

## 1. Introduction
Neural Networks are data hungry. If you only have 1,000 images of cats, the model will memorize them.
**Augmentation** creates fake new data by modifying existing images.
- Rotate 15 degrees.
- Flip horizontally.
- Change brightness.

This forces the model to learn **Invariant Features** (A cat is still a cat if it's upside down... mostly).

---

## 2. Using `torchvision.transforms`

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(), # Look! Augmentation happens BEFORE converting to Tensor
    transforms.Normalize((0.5,), (0.5,))
])

# Validation/Test set should NOT have random augmentations (only Resize/Normalize)
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 3. Practical Exercises

### Exercise 1: Visualize Augmentations
1.  Load an image using PIL.
2.  Apply `transforms.RandomAffine(30)`.
3.  Show the result. It should be shifted/rotated.

---

## 4. Summary
- **Augmentation**: Occurs on the CPU while the GPU is training. Zero-cost regularization.
- **Rule**: Apply heavy augmentation to Train, but none to Test.

**Next Up:** **Transfer Learning**—Standing on the shoulders of giants.
