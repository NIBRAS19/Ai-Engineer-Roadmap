# Day 44: Categorical Features

## 1. Introduction
"Red", "Blue", "Green".
Models need numbers.
We covered `pd.get_dummies` in Pandas week, but in ML pipelines, we use Scikit-Learn encoders.

---

## 2. Ordinal Encoding
For categories with an **Order**.
- "Low", "Medium", "High" -> 0, 1, 2.

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
X_enc = enc.fit_transform([['High'], ['Low']])
# [[2.], [0.]]
```

## 3. One-Hot Encoding
For categories **without Order**.
- "Cat", "Dog", "Bird" -> [1,0,0], [0,1,0], [0,0,1].
- **Problem**: Creates huge number of columns if you have 10,000 cities. (High Dimensionality).

```python
from sklearn.preprocessing import OneHotEncoder
# sparse=False returns a dense array (e.g. numpy array) instead of sparse matrix
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_enc = enc.fit_transform([['Cat'], ['Dog']])
```

## 4. Target Encoding (Mean Encoding)
Replace the category with the **Mean Target Value** for that category.
- If "New York" rows have average Price $500k, replace "New York" with 500,000.
- **Risk**: Serious Data Leakage if not done correctly (need Cross-Validation).

---

## 5. Summary
- **Ordinal**: For ranked data.
- **One-Hot**: For low-cardinality nominal data.
- **Target**: For high-cardinality data (zip codes).

**Next Up:** **Imbalanced Data**—When 99% of your data is Class 0.
