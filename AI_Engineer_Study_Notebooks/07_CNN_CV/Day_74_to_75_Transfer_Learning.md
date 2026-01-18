# Days 74-75: Transfer Learning

## 1. Introduction
Training a CNN from scratch requires:
1.  Millions of images (ImageNet).
2.  Weeks of GPU time.

**Transfer Learning** allows us to take a model trained by Google/Facebook (on ImageNet) and adapt it to our small dataset (e.g., differentiating types of flowers).
- The filters learn "Edges", "Curves", "Textures" which are universal.
- We only need to retrain the **Classifier** (the last layer).

---

## 2. Using Pre-trained Models (`torchvision.models`)

```python
import torchvision.models as models

# 1. Load Pre-trained ResNet18
# weights='DEFAULT' loads the best available weights
model = models.resnet18(weights='DEFAULT')

# 2. Freeze Parameters
# We don't want to mess up the Feature Extractor filters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the Head
# ResNet's final layer is called 'fc' and outputs 1000 classes (ImageNet)
# We change it to output 2 classes (Cat vs Dog)
num_input_features = model.fc.in_features
model.fc = nn.Linear(num_input_features, 2)

# Note: The new layer has requires_grad=True by default
```

---

## 3. Fine-Tuning
If we have more data, we can unfreeze some of the later convolutional layers and train them with a very low learning rate (`1e-5`).

---

## 4. Summary
- **Download**: Get a state-of-the-art model in 1 line.
- **Freeze**: Stop backprop for the "body".
- **Replace Head**: Customize outputs.
- **Train**: Only the head updates. Fast and accurate.

**Next Up:** **CNN Architectures**—Understanding VGG, ResNet, and EfficientNet.
