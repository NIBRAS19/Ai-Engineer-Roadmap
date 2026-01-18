# Day 30: Train-Test Split

## 1. The Golden Rule
**NEVER evaluate your model on the same data used to train it.**
If you do, you are just testing its ability to **memorize** (Overfitting), not its ability to **generalize**.

---

## 2. The Split
We divide our dataset into:
1.  **Training Set (70-80%)**: Used to learn the weights.
2.  **Test Set (20-30%)**: Used to simulate "future data". We only touch this ONCE at the very end.

*(Optional: Validation Set for hyperparameter tuning).*

---

## 3. Implementation

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Fake Data
X = np.arange(100).reshape((50, 2)) # 50 samples, 2 features
y = np.arange(50)

# The Split
# random_state=42 ensures the split is reproducible (same rows every time)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape) # (40, 2)
print("Test Shape:", X_test.shape)   # (10, 2)
```

---

## 4. Stratified Split
What if you have a rare class (e.g., Fraud = 1%)?
A random split might put ALL fraud cases in the test set. The model learns nothing about fraud.
**Stratify** ensures the proportion of classes is preserved in both sets.

```python
# y has 90 'No' and 10 'Yes'
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

---

## 5. Practical Exercises

### Exercise 1: Manual Split vs Sklearn
1.  Create a dataset of 100 rows.
2.  Use `train_test_split` to get an 80/20 split.
3.  Check the size of `X_train`.

---

## 6. Summary
- **Overfitting**: Memorizing data.
- **Train-Test Split**: The defense against overfitting.
- **Random State**: Essential for reproducibility.

**Next Up:** **Linear Regression**—Predicting continuous values.
