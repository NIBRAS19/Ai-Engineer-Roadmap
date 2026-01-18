# Day 45: Imbalanced Data

## 1. Introduction
Scenario: Fraud Detection.
- Legit Transactions: 99,000.
- Fraud Transactions: 1,000.
A model predicting "Legit" effectively gets 99% accuracy.
We need to teach the model to care about the minority class.

---

## 2. Class Weights
Tell the model: "An error on the Fraud class is 99x worse than an error on Legit class".
Most Sklearn models have `class_weight='balanced'`.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

---

## 3. Resampling Techniques

### 3.1 Undersampling
Delete random rows from the Majority class.
- **Pros**: Fast.
- **Cons**: Losses data.

### 3.2 Oversampling (SMOTE)
**Synthetic Minority Over-sampling Technique**.
Creates **fake** new fraud cases by interpolating between existing ones.

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**CRITICAL**: Only Apply SMOTE on **Training Data**. Never on Test Data.

---

## 4. Summary
- **Metrics**: Use F1-Score / Recall, not Accuracy.
- **Algo Level**: Use `class_weight`.
- **Data Level**: Use SMOTE.

**Next Up:** **Hyperparameter Tuning**—Finding the best settings.
