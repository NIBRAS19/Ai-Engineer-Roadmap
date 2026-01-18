# Day 29: Introduction to Machine Learning

## 1. What is Machine Learning?
Traditional Programming: `Data + Rules = Answers`.
Machine Learning: `Data + Answers = Rules`.

We show the computer examples, and it figures out the logic.

---

## 2. Types of Machine Learning

### 2.1 Supervised Learning (The most common)
We have **Labeled Data** (Input X, Output y).
- **Regression**: Predicting a number (Price, Temperature).
- **Classification**: Predicting a category (Spam/Not Spam, Cat/Dog).

### 2.2 Unsupervised Learning
We have **Unlabeled Data** (Only X).
- **Clustering**: Grouping similar items (Customer Segmentation).
- **Dimensionality Reduction**: Compressing data (PCA).

### 2.3 Reinforcement Learning
Learning by trial and error (Robotics, Game playing).

---

## 3. Scikit-Learn (`sklearn`)
The standard library for classical ML in Python.
- Unified API: Nearly every model uses `.fit()` and `.predict()`.
- Built-in Datasets.
- Preprocessing and Metrics included.

### Hello World in Scikit-Learn
```python
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]] # Features
y = [0, 0, 1, 1]         # Labels

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)          # Train the model

print(model.predict([[1.5]])) # Predict for new data
```

---

## 4. The ML Workflow
1.  **Data Collection**: Get X and y.
2.  **Preprocessing**: Clean, Scale, Encode.
3.  **Split**: Train / Validation / Test sets.
4.  **Training**: `model.fit(X_train, y_train)`.
5.  **Evaluation**: Score the model on Test set.
6.  **Deployment**: Save the model.

---

## 5. Summary
- **Supervised**: Labeled data.
- **Unsupervised**: Unlabeled data.
- **Scikit-Learn**: The go-to tool.

**Next Up:** **Train-Test Split**—The Golden Rule of ML (Never test on training data).
