# Day 37: Evaluation Metrics

## 1. Introduction
"Accuracy: 95%" means nothing if your dataset has 99% healthy patients. A model that predicts "Healthy" for everyone gets 99% accuracy but kills the 1% of sick people.
We need better metrics.

---

## 2. Confusion Matrix
A table showing:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

---

## 3. The Big Three
1.  **Precision**: "Of all predicted positives, how many were real?" (Don't cry wolf).
    $$ \frac{TP}{TP + FP} $$
2.  **Recall (Sensitivity)**: "Of all real positives, how many did we find?" (Don't miss cancer).
    $$ \frac{TP}{TP + FN} $$
3.  **F1 Score**: Harmonic mean of Precision and Recall. The balanced metric.

---

## 4. Classification Report
Sklearn gives you everything in one command.

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

---

## 5. Practical Exercises

### Exercise 1: The Imbalanced Fyle
Create `y_true = [0]*95 + [1]*5` (95% Class 0).
Create `y_pred = [0]*100` (Model predicts 0 always).
calculate Accuracy (High) vs Recall (Zero).

---

## 6. Summary
- **Accuracy**: Good for balanced classes.
- **Precision/Recall**: Necessary for imbalanced classes.
- **Confusion Matrix**: The full picture.

**Next Up:** **Ensemble Methods**—Combining weak models to create a super-learner.
