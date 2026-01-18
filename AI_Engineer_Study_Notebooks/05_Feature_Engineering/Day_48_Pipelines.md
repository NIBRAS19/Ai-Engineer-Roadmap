# Day 48: Pipelines

## 1. Introduction
The most common bug in ML Deployment:
- You scaled your training data.
- **You forgot** to scale the new incoming data in production.
- The model outputs garbage.

**Pipelines** bundle Preprocessing + Modeling into a single object.
`Pipeline.fit()` automatically scales Train.
`Pipeline.predict()` automatically scales Test/Live data.

---

## 2. Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Steps: List of (Name, Object)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Use it just like a normal model
pipe.fit(X_train, y_train) # Scales X_train, then fits SVM
pipe.score(X_test, y_test) # Scales X_test, then predicts
```

---

## 3. ColumnTransformer
Pipelines apply to the whole table.
What if you want to Scale Column A, but OneHotEncode Column B?
Use `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'salary']),
        ('cat', OneHotEncoder(), ['gender', 'city'])
    ])

full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])
```

---

## 4. Summary
- **Pipeline**: Chains steps. Prevents data leakage.
- **Reliability**: Ensures production data is treated exactly like training data.

**Next Up:** **Model Persistence**—Saving your brain to a file.
