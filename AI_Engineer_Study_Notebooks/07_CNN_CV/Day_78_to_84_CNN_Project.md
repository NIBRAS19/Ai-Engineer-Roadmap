# Days 78-84: Image Classification Project

## 1. The Challenge
You will build a system to classify images from the **CIFAR-10** dataset (Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck).

---

## 2. Checklist

### Phase 1: Data Prep
- [ ] Load CIFAR-10 using `torchvision.datasets`.
- [ ] Create DataLoaders (Train/Test).
- [ ] Visualize a batch of images (`matplotlib`).
- [ ] Implement Augmentation (Flip, Crop, Normalize).

### Phase 2: Model Architecture
- [ ] **Approach A**: Build a custom CNN (3 blocks of Conv-BatchNorm-ReLU-Pool).
- [ ] **Approach B**: Use Transfer Learning (ResNet18). Modify the last layer for 10 classes.

### Phase 3: Training Loop
- [ ] Write the loop (Forward, Loss, Backward, Step).
- [ ] Track Training Loss and Validation Accuracy per epoch.
- [ ] Save the best model (`best_model.pth`).

### Phase 4: Analysis
- [ ] Plot Learning Curves (Loss vs Epochs).
- [ ] Calculate **Confusion Matrix**. Which classes are confused? (e.g., Cat vs Dog).
- [ ] Visual Inference: Show an image and the top 3 probabilities.

---

## 3. Bonus Challenge
- Use **Learning Rate Scheduler** (`torch.optim.lr_scheduler`) to decay LR when loss plateaus.
- Try **Test Time Augmentation (TTA)**: Average predictions of the original image + a flipped version.

**CONGRATULATIONS!** You have completed **Weeks 11-12: Computer Vision**.
You can now see the world through the eyes of an AI.
**Next Week:** **Natural Language Processing (NLP)**—Teaching machines to read and speak.
