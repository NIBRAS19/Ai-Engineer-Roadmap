# AI Engineer Day-Wise Learning Plan
## Complete Curriculum: Beginner to Advanced




print(f"Your grade is: {grade}")
```

**Output:** `Your grade is: B`

---

### Comparison Operators
```python
x = 10
y = 20

print(x == y)   # Equal: False
print(x != y)   # Not equal: True
print(x > y)    # Greater than: False
print(x < y)    # Less than: True
print(x >= 10)  # Greater or equal: True
print(x <= 10)  # Less or equal: True
```

---

### Logical Operators
```python
a = True
b = False

print(a and b)  # False (both must be True)
print(a or b)   # True (at least one True)
print(not a)    # False (negation)

# Practical example
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive")
```

---


### Lambda Functions
**What it is:** Anonymous, single-expression functions.

**Syntax:**
```python
lambda arguments: expression
```

**Example:**
```python
# Regular function
def square(x):
    return x ** 2

# Lambda equivalent
square = lambda x: x ** 2

print(square(5))  # 25

# Common use with map, filter
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # [2, 4, 6, 8, 10]

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]
```

---

## Day 6: Object-Oriented Programming (OOP)

### Classes and Objects
**What it is:** Classes are blueprints for creating objects. Objects are instances of classes.

**Syntax:**
```python
class ClassName:
    def __init__(self, parameters):
        self.attribute = value
    
    def method(self):
        # code
```

**Example:**
```python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor (initializer)
    def __init__(self, name, age):
        # Instance attributes
        self.name = name
        self.age = age
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    def description(self):
        return f"{self.name} is {self.age} years old"

# Creating objects
buddy = Dog("Buddy", 3)
max_dog = Dog("Max", 5)

print(buddy.name)           # Buddy
print(buddy.bark())         # Buddy says Woof!
print(max_dog.description()) # Max is 5 years old
print(Dog.species)          # Canis familiaris
```

---

### Inheritance
**What it is:** A class can inherit attributes and methods from another class.

**Example:**
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!
```

---

## Day 7: File Handling & Exception Handling

### Reading Files
```python
# Method 1: Basic
file = open("data.txt", "r")
content = file.read()
file.close()

# Method 2: Using 'with' (recommended - auto-closes)
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

### Writing Files
```python
# Write mode (overwrites)
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("Second line")

# Append mode
with open("output.txt", "a") as file:
    file.write("\nAppended line")
```

### Exception Handling
**What it is:** Handles errors gracefully without crashing the program.

**Syntax:**
```python
try:
    # code that might raise an error
except ExceptionType:
    # handle the error
finally:
    # always executes
```

**Example:**
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always runs")

# Practical file example
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found!")
```

---

# Week 2: NumPy - The Language of Matrices

---

## Day 8: Introduction to NumPy Arrays

### What is NumPy?
**What it is:** NumPy (Numerical Python) is the fundamental package for scientific computing in Python.

**Why it is used:** 
- 50x faster than Python lists
- Foundation for all AI/ML libraries
- Enables vectorized operations

**Installation:**
```python
pip install numpy
```

### Creating Arrays
```python
import numpy as np

# 1D Array (Vector)
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)  # [1 2 3 4 5]

# 2D Array (Matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d.shape)  # (2, 3)

# Creation functions
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(3)
arr_range = np.arange(0, 10, 2)  # [0 2 4 6 8]
```

---

## Day 9: Array Indexing and Slicing

### Basic Indexing
```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[0])    # 10
print(arr[-1])   # 50
print(arr[1:4])  # [20 30 40]
```

### 2D Indexing
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[0, 1])     # 2
print(arr_2d[:, 1])     # [2 5 8] (column)
print(arr_2d[0:2, 1:3]) # [[2 3] [5 6]]
```

### Boolean Masking
```python
arr = np.array([1, 5, 10, 15, 20])
print(arr[arr > 10])  # [15 20]
print(arr[(arr > 5) & (arr < 20)])  # [10 15]
```

---

## Day 10: Vectorization and Broadcasting

### Element-wise Operations
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 + arr2)   # [5 7 9]
print(arr1 * arr2)   # [4 10 18]
print(arr1 ** 2)     # [1 4 9]
```

### Broadcasting
```python
arr = np.array([1, 2, 3])
print(arr + 10)  # [11 12 13]

matrix = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
print(matrix + row)  # [[11 22 33] [14 25 36]]
```

---

## Day 11: Matrix Operations

### Dot Product
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)  # Matrix multiplication
# [[19 22]
#  [43 50]]

print(A.T)  # Transpose
```

---

## Day 12: Reshaping Arrays

```python
arr = np.arange(12)
print(arr.reshape(3, 4))
print(arr.reshape(2, -1))  # Auto-calculate
print(arr.flatten())

# Adding dimensions
arr = np.array([1, 2, 3])
print(arr[np.newaxis, :])  # (1, 3)
print(arr[:, np.newaxis])  # (3, 1)
```

---

## Day 13: Statistical Functions

```python
arr = np.array([1, 2, 3, 4, 5])

print(np.mean(arr))   # 3.0
print(np.std(arr))    # 1.41
print(np.sum(arr))    # 15
print(np.min(arr))    # 1
print(np.argmax(arr)) # 4

# Random numbers
np.random.seed(42)
rand = np.random.randn(3, 3)  # Normal distribution
```

---

## Day 14: Practical NumPy for AI

### Simulating Neural Network Layer
```python
np.random.seed(42)
X = np.random.randn(5, 3)  # 5 samples, 3 features
W = np.random.randn(3, 4)  # Weights
b = np.zeros(4)            # Bias

Z = X @ W + b              # Forward pass
A = np.maximum(0, Z)       # ReLU activation
```

### Data Normalization
```python
data = np.array([10, 20, 30, 40, 50])
normalized = (data - data.min()) / (data.max() - data.min())
standardized = (data - data.mean()) / data.std()
```

---

# Week 3: Pandas - Data Manipulation

---

## Day 15: DataFrames and Series

### What is Pandas?
**What it is:** Library for data manipulation and analysis with tabular data structures.

**Why it is used:** Essential for loading, cleaning, and preparing datasets for ML.

```python
import pandas as pd

# Creating DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

print(df.head())     # First 5 rows
print(df.info())     # Data types
print(df.describe()) # Statistics
```

---

## Day 16: Data Selection (loc & iloc)

```python
# iloc: Integer-based location
print(df.iloc[0])       # First row
print(df.iloc[0:2, 1])  # Rows 0-1, column 1

# loc: Label-based location
print(df.loc[df['age'] > 25])
print(df.loc[:, 'name'])

# Boolean filtering
print(df[df['salary'] > 55000])
print(df[(df['age'] > 25) & (df['salary'] < 70000)])
```

---

## Day 17: Handling Missing Data

```python
# Check for missing values
print(df.isnull().sum())

# Drop missing values
df.dropna()                    # Drop rows with any NaN
df.dropna(subset=['age'])      # Drop where age is NaN

# Fill missing values
df.fillna(0)                   # Fill with 0
df.fillna(df.mean())           # Fill with mean
df['age'].fillna(df['age'].median(), inplace=True)
```

---

## Day 18: GroupBy and Aggregations

```python
# GroupBy operations
grouped = df.groupby('department')
print(grouped.mean())
print(grouped.agg({'salary': 'mean', 'age': 'max'}))

# Pivot tables
pivot = pd.pivot_table(df, values='salary', 
                       index='department', 
                       aggfunc='mean')
```

---

## Day 19: Feature Engineering

```python
# Apply custom functions
df['salary_k'] = df['salary'].apply(lambda x: x / 1000)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['department'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 100], 
                         labels=['Young', 'Mid', 'Senior'])

# Date features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

---

## Day 20: Merging and Concatenating

```python
# Concatenate DataFrames
df_combined = pd.concat([df1, df2], axis=0)  # Stack rows
df_combined = pd.concat([df1, df2], axis=1)  # Stack columns

# Merge (SQL-style joins)
merged = pd.merge(df1, df2, on='id', how='left')
merged = pd.merge(df1, df2, on='id', how='inner')
```

---

## Day 21: Reading/Writing Data

```python
# CSV files
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Excel files
df = pd.read_excel('data.xlsx')
df.to_excel('output.xlsx', index=False)

# JSON files
df = pd.read_json('data.json')
df.to_json('output.json')
```

---

# Week 4: Mathematics for AI

---

## Day 22: Linear Algebra - Vectors

### What is a Vector?
**What it is:** One-dimensional array of numbers representing magnitude and direction.

**Why it is used:** Data points, features, and embeddings in ML are vectors.

```python
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Vector norm (magnitude)
norm = np.linalg.norm(v1)  # sqrt(1² + 2² + 3²) = 3.74

# Unit vector (normalized)
unit_v = v1 / np.linalg.norm(v1)
```

---

## Day 23: Linear Algebra - Matrices

```python
# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B

# Inverse
A_inv = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

---

## Day 24: Calculus - Derivatives

### What is a Derivative?
**What it is:** Rate of change of a function with respect to its input.

**Why it is used:** Gradient descent uses derivatives to minimize loss functions.

```python
# Numerical derivative approximation
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

print(derivative(f, 3))  # ~6 (derivative of x² is 2x)
```

---

## Day 25: Calculus - Gradient Descent

### What is Gradient Descent?
**What it is:** Optimization algorithm to find minimum of a function.

**Formula:** `new_weight = old_weight - learning_rate * gradient`

```python
def gradient_descent(f, df, x0, learning_rate=0.1, iterations=100):
    x = x0
    history = [x]
    
    for _ in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example: minimize f(x) = x²
f = lambda x: x ** 2
df = lambda x: 2 * x  # derivative

minimum, history = gradient_descent(f, df, x0=10)
print(f"Minimum at x = {minimum}")  # ~0
```

---

## Day 26: Statistics - Descriptive Statistics

```python
import numpy as np
from scipy import stats

data = np.array([2, 4, 6, 8, 10, 12, 14])

# Central tendency
mean = np.mean(data)      # 8.0
median = np.median(data)  # 8.0
mode = stats.mode(data)   # Most frequent

# Spread
variance = np.var(data)   # 16.0
std_dev = np.std(data)    # 4.0
range_val = np.max(data) - np.min(data)  # 12

# Percentiles
percentile_25 = np.percentile(data, 25)
percentile_75 = np.percentile(data, 75)
```

---

## Day 27: Probability Distributions

```python
import numpy as np
from scipy import stats

# Normal Distribution
normal = np.random.normal(mean=0, scale=1, size=1000)

# Probability density
x = 1.5
pdf = stats.norm.pdf(x, loc=0, scale=1)

# Cumulative probability
cdf = stats.norm.cdf(x, loc=0, scale=1)  # P(X <= 1.5)

# Bernoulli (Binary outcomes)
bernoulli = np.random.binomial(n=1, p=0.5, size=100)

# Uniform Distribution
uniform = np.random.uniform(low=0, high=1, size=100)
```

---

## Day 28: Loss Functions

### Mean Squared Error (MSE)
**What it is:** Average squared difference between predictions and actual values.

**Why it is used:** For regression problems.

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
print(mse(y_true, y_pred))  # 0.024
```

### Cross-Entropy Loss
**What it is:** Measures difference between two probability distributions.

**Why it is used:** For classification problems.

```python
def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
print(cross_entropy(y_true, y_pred))
```

---

# Week 5-6: Machine Learning with Scikit-Learn

---

## Day 29: Introduction to Machine Learning

### What is Machine Learning?
**What it is:** Systems that learn patterns from data without being explicitly programmed.

**Types of ML:**
- **Supervised Learning:** Learning from labeled data (classification, regression)
- **Unsupervised Learning:** Finding patterns in unlabeled data (clustering)
- **Reinforcement Learning:** Learning through trial and rewards

### ML Workflow
```python
# Standard ML Pipeline
# 1. Load data
# 2. Explore and preprocess
# 3. Split into train/test
# 4. Train model
# 5. Evaluate
# 6. Tune hyperparameters
# 7. Deploy
```

---

## Day 30: Train-Test Split and Cross-Validation

```python
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Sample data
X = np.random.randn(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, 100)  # Binary labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}")  # 80
print(f"Test size: {len(X_test)}")    # 20

# Cross-validation
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

---

## Day 31: Linear Regression

### What is Linear Regression?
**What it is:** Predicting continuous values by fitting a straight line.

**Formula:** `y = mx + b` (or `y = w₁x₁ + w₂x₂ + ... + b`)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2 * X.flatten() + 5 + np.random.randn(100) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"Coefficient: {model.coef_[0]:.3f}")   # ~2
print(f"Intercept: {model.intercept_:.3f}")   # ~5
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
```

---

## Day 32: Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Pipeline for cleaner code
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)
```

---

## Day 33: Regularization (Ridge & Lasso)

### What is Regularization?
**What it is:** Adding penalty to prevent overfitting by constraining model weights.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization) - can zero out features
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# Compare coefficients
print(f"Ridge coefficients: {ridge.coef_}")
print(f"Lasso coefficients: {lasso.coef_}")  # Some may be 0
```

---

## Day 34: Logistic Regression (Classification)

### What is Logistic Regression?
**What it is:** Classification algorithm using sigmoid function to predict probabilities.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilities

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

---

## Day 35: Decision Trees

### What is a Decision Tree?
**What it is:** Tree-like model that makes decisions based on feature thresholds.

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Classification
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
print(f"Feature importances: {clf.feature_importances_}")

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.savefig('decision_tree.png')
```

---

## Day 36: K-Nearest Neighbors (KNN)

### What is KNN?
**What it is:** Classifies based on majority vote of K nearest neighbors.

```python
from sklearn.neighbors import KNeighborsClassifier

# Train
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Finding optimal K
k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

optimal_k = k_range[np.argmax(scores)]
print(f"Optimal K: {optimal_k}")
```

---

## Day 37: Evaluation Metrics

### Confusion Matrix
```python
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Metrics for binary classification
# (use average='weighted' for multiclass)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
```

### ROC Curve (Binary Classification)
```python
from sklearn.metrics import roc_curve, auc

# For binary classification
y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

---

## Day 38: Random Forest

### What is Random Forest?
**What it is:** Ensemble of decision trees using bagging (bootstrap aggregating).

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)
rf.fit(X_train, y_train)

print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Feature Importances: {rf.feature_importances_}")

# Out-of-bag score
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.3f}")
```

---

## Day 39: Gradient Boosting (XGBoost, LightGBM)

### What is Gradient Boosting?
**What it is:** Sequential ensemble where each tree corrects errors of previous trees.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Scikit-learn implementation
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Accuracy: {gb.score(X_test, y_test):.3f}")

# XGBoost (install: pip install xgboost)
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_model.fit(X_train, y_train)
print(f"XGBoost Accuracy: {xgb_model.score(X_test, y_test):.3f}")
```

---

## Day 40: Support Vector Machines (SVM)

### What is SVM?
**What it is:** Finds optimal hyperplane that maximizes margin between classes.

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# SVMs are sensitive to feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
print(f"Accuracy: {svm.score(X_test_scaled, y_test):.3f}")

# Different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X_train_scaled, y_train)
    print(f"{kernel}: {svm.score(X_test_scaled, y_test):.3f}")
```

---

## Day 41: K-Means Clustering

### What is K-Means?
**What it is:** Unsupervised algorithm that groups data into K clusters.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_}")  # Sum of squared distances

# Silhouette score (how well-separated clusters are)
sil_score = silhouette_score(X, clusters)
print(f"Silhouette Score: {sil_score:.3f}")

# Elbow method for finding optimal K
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
```

---

## Day 42: Principal Component Analysis (PCA)

### What is PCA?
**What it is:** Dimensionality reduction by projecting data onto principal components.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained: {sum(pca.explained_variance_ratio_):.3f}")

# Scree plot
pca_full = PCA()
pca_full.fit(X)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
         np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
```

---

# Week 7-8: Feature Engineering & Hyperparameter Tuning

---

## Day 43: Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Min-Max Scaler (0-1 range)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train)

# Robust Scaler (handles outliers)
robust = RobustScaler()
X_robust = robust.fit_transform(X_train)

# Important: fit on train, transform on test
X_test_scaled = scaler.transform(X_test)
```

---

## Day 44: Handling Categorical Features

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Label Encoding (ordinal categories)
le = LabelEncoder()
encoded = le.fit_transform(['low', 'medium', 'high', 'medium'])
print(encoded)  # [1, 2, 0, 2]

# One-Hot Encoding (nominal categories)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform([['red'], ['blue'], ['green'], ['red']])

# Column Transformer for mixed data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'salary']),
        ('cat', OneHotEncoder(), ['color', 'size'])
    ]
)
```

---

## Day 45: Handling Imbalanced Data

```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Check class distribution
from collections import Counter
print(Counter(y_train))

# SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights in model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

---

## Day 46: Hyperparameter Tuning - Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

---

## Day 47: Hyperparameter Tuning - Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,  # Number of random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

---

## Day 48: Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simple pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
print(f"Accuracy: {pipe.score(X_test, y_test):.3f}")

# Pipeline with GridSearch
param_grid = {
    'pca__n_components': [5, 10, 15],
    'classifier__n_estimators': [50, 100]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## Day 49: Model Persistence

```python
import joblib
import pickle

# Save model with joblib (recommended for sklearn)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

# Save with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Save pipeline (includes preprocessing)
joblib.dump(pipe, 'pipeline.joblib')
```

---

## Day 50-56: ML Project Practice

### Complete ML Project Workflow
```python
# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load and explore data
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Preprocess
# Handle missing values
df.fillna(df.median(), inplace=True)

# Encode categoricals
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 8. Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 9. Save model
joblib.dump(model, 'final_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
```

---

# Week 9-10: Deep Learning with PyTorch

---

## Day 57: Introduction to PyTorch

### What is PyTorch?
**What it is:** Open-source deep learning framework by Meta with dynamic computation graphs.

**Why it is used:** Industry standard for research and production AI.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Check PyTorch version and GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## Day 58: Tensors

### What is a Tensor?
**What it is:** Multi-dimensional array (like NumPy array) that can run on GPU.

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
z = torch.ones(2, 3)
r = torch.randn(3, 3)  # Normal distribution

print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")

# Tensor operations
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(a + b)         # Element-wise addition
print(a * b)         # Element-wise multiplication
print(a @ b)         # Matrix multiplication
print(torch.matmul(a, b))  # Same as @

# NumPy conversion
import numpy as np
numpy_arr = np.array([1, 2, 3])
tensor = torch.from_numpy(numpy_arr)
back_to_numpy = tensor.numpy()
```

---

## Day 59: Autograd (Automatic Differentiation)

### What is Autograd?
**What it is:** Automatic computation of gradients for backpropagation.

```python
import torch

# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward pass
y = x ** 2
z = y.sum()  # z = x₁² + x₂²

print(f"z = {z}")

# Backward pass (compute gradients)
z.backward()

# Gradients: dz/dx = 2x
print(f"Gradients: {x.grad}")  # [4.0, 6.0]

# Detach from computation graph
x_detached = x.detach()

# No gradient context
with torch.no_grad():
    y = x * 2  # No gradient tracking
```

---

## Day 60: Neural Network Basics (nn.Module)

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN(input_size=10, hidden_size=32, output_size=2)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

---

## Day 61: Activation Functions

```python
import torch.nn as nn

# Common activation functions
relu = nn.ReLU()          # max(0, x)
sigmoid = nn.Sigmoid()     # 1 / (1 + e^(-x))
tanh = nn.Tanh()          # (e^x - e^(-x)) / (e^x + e^(-x))
softmax = nn.Softmax(dim=1)  # For multi-class output
leaky_relu = nn.LeakyReLU(0.01)  # Allows small negative gradient

# Example
x = torch.randn(5)
print(f"ReLU: {relu(x)}")
print(f"Sigmoid: {sigmoid(x)}")
print(f"Tanh: {tanh(x)}")
```

---

## Day 62: Loss Functions

```python
import torch.nn as nn

# Regression loss
mse_loss = nn.MSELoss()      # Mean Squared Error
mae_loss = nn.L1Loss()       # Mean Absolute Error

# Classification loss
ce_loss = nn.CrossEntropyLoss()     # Multi-class
bce_loss = nn.BCEWithLogitsLoss()   # Binary

# Example usage
predictions = torch.randn(5, 3)  # 5 samples, 3 classes
targets = torch.tensor([0, 1, 2, 1, 0])  # Class indices

loss = ce_loss(predictions, targets)
print(f"Cross Entropy Loss: {loss.item():.4f}")
```

---

## Day 63: Optimizers

```python
import torch.optim as optim

model = SimpleNN(10, 32, 2)

# Common optimizers
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = optim.Adam(model.parameters(), lr=0.001)
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.1)
# or
scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, mode='min', patience=5)
```

---

## Day 64: The Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Model, loss, optimizer
model = SimpleNN(10, 32, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

---

## Day 65: DataLoader and Datasets

```python
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=0
)

# Iterate through batches
for batch_X, batch_y in dataloader:
    print(f"Batch shape: {batch_X.shape}")
    break
```

---

## Day 66-67: Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train), 
    torch.LongTensor(y_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val), 
    torch.LongTensor(y_val)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = SimpleNN(10, 64, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with validation
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(val_loader), correct / total

# Training loop
for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
```

---

## Day 68: Saving and Loading Models

```python
import torch

# Save entire model
torch.save(model, 'model_complete.pth')
model = torch.load('model_complete.pth')

# Save only state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load state dict
model = SimpleNN(10, 64, 2)
model.load_state_dict(torch.load('model_weights.pth'))

# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

# Week 11-12: Convolutional Neural Networks (CNNs)

---

## Day 69: Introduction to CNNs

### What is a CNN?
**What it is:** Neural network designed for image processing using convolutions.

**Components:**
- Convolutional layers: Extract features
- Pooling layers: Downsample
- Fully connected layers: Classification

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

---

## Day 70: Convolutional Layers Deep Dive

```python
import torch.nn as nn

# Conv2d parameters
conv = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3x3 filter
    stride=1,           # Step size
    padding=1           # Keep same size
)

# Calculate output size:
# Output = (Input - Kernel + 2*Padding) / Stride + 1

# Input: (batch, 3, 32, 32)
# Output: (batch, 64, 32, 32)

# Different padding modes
same_padding = nn.Conv2d(3, 64, 3, padding='same')
valid_padding = nn.Conv2d(3, 64, 3, padding=0)  # No padding
```

---

## Day 71: Pooling and Batch Normalization

```python
import torch.nn as nn

# Pooling layers
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

# Batch Normalization
# Normalizes activations, speeds up training
batch_norm = nn.BatchNorm2d(64)  # 64 channels

# Typical block
conv_block = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

---

## Day 72: MNIST Classification with CNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Train
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Test accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}")
```

---

## Day 73: Data Augmentation

```python
from torchvision import transforms

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Day 74-75: Transfer Learning

### What is Transfer Learning?
**What it is:** Using pre-trained models as starting point for new tasks.

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train the new layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tuning: Unfreeze some layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Train with lower learning rate for pre-trained layers
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

---

## Day 76-77: Popular CNN Architectures

```python
from torchvision import models

# VGG16 - Simple, deep architecture
vgg16 = models.vgg16(pretrained=True)

# ResNet - Skip connections
resnet50 = models.resnet50(pretrained=True)

# EfficientNet - Efficient scaling
efficientnet = models.efficientnet_b0(pretrained=True)

# MobileNet - Lightweight for mobile
mobilenet = models.mobilenet_v3_small(pretrained=True)

# Vision Transformer
vit = models.vit_b_16(pretrained=True)
```

---

## Day 78-84: CNN Project Practice

### Image Classification Project
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset (ImageFolder expects: root/class_name/images)
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model with transfer learning
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(25):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
          f"Acc={100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), 'image_classifier.pth')
```

---

# Week 13-14: Natural Language Processing (NLP)

---

## Day 85: Introduction to NLP

### What is NLP?
**What it is:** Field of AI focused on understanding and generating human language.

**Key Applications:**
- Text Classification (sentiment, spam)
- Named Entity Recognition (NER)
- Machine Translation
- Question Answering
- Text Generation

```python
# Common NLP libraries
# pip install nltk spacy transformers

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Day 86: Text Preprocessing

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The quick brown foxes are jumping over the lazy dogs!"

# Lowercase
text_lower = text.lower()

# Remove punctuation
text_clean = re.sub(r'[^\w\s]', '', text_lower)

# Tokenization
tokens = word_tokenize(text_clean)
print(f"Tokens: {tokens}")

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]
print(f"Filtered: {filtered}")

# Stemming (crude, rule-based)
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print(f"Stemmed: {stemmed}")

# Lemmatization (dictionary-based, better)
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
print(f"Lemmatized: {lemmatized}")
```

---

## Day 87: Text Representation - Bag of Words & TF-IDF

### Bag of Words (BoW)
**What it is:** Represents text as word frequency counts.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "I love machine learning",
    "Machine learning is great",
    "Deep learning is part of machine learning"
]

# Bag of Words
bow = CountVectorizer()
bow_matrix = bow.fit_transform(documents)
print(f"Vocabulary: {bow.get_feature_names_out()}")
print(f"BoW Matrix:\n{bow_matrix.toarray()}")

# TF-IDF (Term Frequency - Inverse Document Frequency)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
print(f"TF-IDF Matrix:\n{tfidf_matrix.toarray()}")
```

---

## Day 88: Word Embeddings

### What are Word Embeddings?
**What it is:** Dense vector representations where similar words have similar vectors.

```python
import gensim.downloader as api

# Load pre-trained Word2Vec
word2vec = api.load('word2vec-google-news-300')

# Get word vector
vector = word2vec['king']
print(f"Vector shape: {vector.shape}")  # (300,)

# Find similar words
similar = word2vec.most_similar('king', topn=5)
print(f"Similar to 'king': {similar}")

# Famous example: King - Man + Woman = Queen
result = word2vec.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(f"King - Man + Woman = {result}")
```

---

## Day 89: Text Classification with Traditional ML

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data
texts = ["Great product!", "Terrible experience", "Love it", "Waste of money"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Day 90-91: Recurrent Neural Networks (RNN & LSTM)

### What is an RNN?
**What it is:** Neural network with memory for sequential data.

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate last hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

model = LSTMClassifier(vocab_size=10000, embed_dim=128, 
                       hidden_dim=256, num_classes=2)
```

---

## Day 92: Introduction to Transformers

### What is a Transformer?
**What it is:** Architecture using self-attention, replacing RNNs for NLP.

**Key Concepts:**
- **Self-Attention:** Relates all positions in sequence
- **Multi-Head Attention:** Multiple attention patterns
- **Positional Encoding:** Adds position information

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output, attn_weights
```

---

# Week 15-16: Hugging Face Transformers & LLMs

---

## Day 93: Hugging Face Basics

```python
# pip install transformers

from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
result = ner("Apple CEO Tim Cook announced new products in California.")
print(result)

# Question Answering
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is its capital."
)
print(result)

# Text Generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50)
print(result)
```

---

## Day 94: Tokenizers

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you doing today?"

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Encode (to IDs)
encoded = tokenizer.encode(text)
print(f"IDs: {encoded}")

# Full encoding with attention mask
encoding = tokenizer(
    text,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt"
)
print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention Mask: {encoding['attention_mask']}")

# Batch encoding
texts = ["Hello world", "How are you?"]
batch = tokenizer(texts, padding=True, return_tensors="pt")
```

---

## Day 95: Using Pre-trained Models

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Machine learning is fascinating"
inputs = tokenizer(text, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: (batch, seq_len, hidden_dim)
# outputs.pooler_output: (batch, hidden_dim) - [CLS] token representation
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"Pooler output shape: {outputs.pooler_output.shape}")

# Get sentence embedding (mean pooling)
sentence_embedding = outputs.last_hidden_state.mean(dim=1)
```

---

## Day 96-97: Fine-tuning BERT for Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("imdb")

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", 
                    truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),
    eval_dataset=tokenized_datasets["test"].shuffle().select(range(500)),
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
```

---

## Day 98: Text Embeddings for Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn from data",
    "Neural networks are inspired by the brain",
    "The weather is nice today"
]

embeddings = model.encode(documents)
print(f"Embeddings shape: {embeddings.shape}")  # (4, 384)

# Semantic search
query = "How do computers learn?"
query_embedding = model.encode(query)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], embeddings)[0]

# Rank results
for idx in np.argsort(similarities)[::-1]:
    print(f"{similarities[idx]:.3f}: {documents[idx]}")
```

---

## Day 99-100: Working with LLM APIs

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Chat completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Machine learning is fascinating"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

---

## Day 101-102: Prompt Engineering

### Prompt Techniques

```python
# 1. Zero-shot prompting
prompt = "Classify the sentiment: 'I love this movie!'"

# 2. Few-shot prompting
prompt = """Classify the sentiment:

Text: "Great experience!" -> Positive
Text: "Terrible service" -> Negative
Text: "I love this product!" -> """

# 3. Chain-of-Thought (CoT)
prompt = """
Solve this step by step:
Q: If a train travels at 60 mph for 2.5 hours, how far does it travel?

Let's think step by step:
1. Speed = 60 mph
2. Time = 2.5 hours
3. Distance = Speed × Time = 60 × 2.5 = 150 miles

Answer: 150 miles

Q: If a car travels at 45 mph for 3 hours, how far does it travel?

Let's think step by step:
"""

# 4. Role-based prompting
system_prompt = """You are an expert Python developer. 
When explaining code:
- Use simple language
- Provide examples
- Highlight best practices"""
```

---

## Day 103-105: Building a RAG System

### What is RAG?
**What it is:** Retrieval-Augmented Generation - combining search with LLM generation.

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain({"query": "What are the main topics covered?"})
print(result["result"])
print(f"Sources: {result['source_documents']}")
```

---

## Day 106-112: NLP Project Practice

### Complete NLP Pipeline
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
import numpy as np

# Configuration
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3

# Load dataset
dataset = load_dataset("emotion")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS
)

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./emotion_classifier",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save
model.save_pretrained("./emotion_classifier_final")
tokenizer.save_pretrained("./emotion_classifier_final")

# Inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction.item()

# Test
print(predict("I am so happy today!"))
```

---

# Week 17-18: Advanced LLMs & AI Agents

---

## Day 113: Local LLM Deployment

```python
# Using Ollama for local LLMs
# Install: curl https://ollama.ai/install.sh | sh
# Run: ollama run llama2

import ollama

# Simple completion
response = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Explain neural networks'}]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Write a poem'}],
    stream=True
):
    print(chunk['message']['content'], end='')
```

---

## Day 114: Function Calling with LLMs

```python
from openai import OpenAI
import json

client = OpenAI()

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

# Call with function
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Function: {function_name}, Args: {arguments}")
```

---

## Day 115-116: Building AI Agents with LangChain

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()

def calculator(expression: str) -> str:
    """Evaluates a math expression"""
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search the internet for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations"
    )
]

# Create agent
llm = ChatOpenAI(temperature=0)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=5
)

# Run agent
result = agent_executor.invoke({
    "input": "What is the current population of Japan divided by 1000?"
})
print(result["output"])
```

---

## Day 117-118: Multi-Agent Systems

```python
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI

# Agent roles
research_agent_prompt = """You are a research agent.
Your job is to gather information on the given topic."""

writer_agent_prompt = """You are a writing agent.
Your job is to write content based on research provided."""

reviewer_agent_prompt = """You are a review agent.
Your job is to review and improve the content."""

# Simple multi-agent workflow
def multi_agent_workflow(topic):
    llm = ChatOpenAI(temperature=0.7)
    
    # Step 1: Research
    research = llm.predict(f"{research_agent_prompt}\n\nTopic: {topic}")
    print("Research completed...")
    
    # Step 2: Write
    draft = llm.predict(f"{writer_agent_prompt}\n\nResearch:\n{research}")
    print("Draft completed...")
    
    # Step 3: Review
    final = llm.predict(f"{reviewer_agent_prompt}\n\nDraft:\n{draft}")
    print("Review completed...")
    
    return final

result = multi_agent_workflow("Benefits of AI in healthcare")
print(result)
```

---

## Day 119-120: LangGraph for Stateful Agents

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

def research_node(state: AgentState):
    """Research step"""
    return {"messages": ["Research completed"], "next_action": "write"}

def write_node(state: AgentState):
    """Writing step"""
    return {"messages": ["Writing completed"], "next_action": "review"}

def review_node(state: AgentState):
    """Review step"""
    return {"messages": ["Review completed"], "next_action": "end"}

def should_continue(state: AgentState):
    if state["next_action"] == "end":
        return END
    return state["next_action"]

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)

workflow.set_entry_point("research")
workflow.add_conditional_edges("research", should_continue)
workflow.add_conditional_edges("write", should_continue)
workflow.add_conditional_edges("review", should_continue)

app = workflow.compile()

# Run
result = app.invoke({"messages": [], "next_action": "research"})
print(result)
```

---

# Week 19-20: MLOps & Deployment

---

## Day 121-122: Model Serving with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib

app = FastAPI()

# Load model at startup
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Preprocess
    features = scaler.transform([request.features])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability)
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
```

---

## Day 123-124: Docker for ML

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.joblib
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

---

## Day 125-126: Model Monitoring

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
PREDICTIONS = Counter('predictions_total', 'Total predictions', ['model', 'class'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

def predict_with_monitoring(features):
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Record metrics
    PREDICTIONS.labels(model='v1', class=str(prediction)).inc()
    LATENCY.observe(time.time() - start_time)
    
    return prediction

# Start metrics server
start_http_server(8001)
```

### Data Drift Detection
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Reference data (training data)
reference_data = pd.read_csv("training_data.csv")

# Current data (production data)
current_data = pd.read_csv("production_data.csv")

# Create drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save report
report.save_html("drift_report.html")
```

---

## Day 127-128: CI/CD for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run tests
        run: pytest tests/

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Train model
        run: python train.py
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying model to production..."
```

---

## Day 129-130: Experiment Tracking with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
---

# 🎉 Congratulations!

You have completed the **140-Day AI Engineer Learning Plan**!

## Summary of Topics Covered:

| Week | Days | Topic |
|------|------|-------|
| 1 | 1-7 | Python Fundamentals |
| 2 | 8-14 | NumPy |
| 3 | 15-21 | Pandas |
| 4 | 22-28 | Mathematics for AI |
| 5-6 | 29-42 | Machine Learning (Scikit-Learn) |
| 7-8 | 43-56 | Feature Engineering & Hyperparameter Tuning |
| 9-10 | 57-68 | Deep Learning (PyTorch) |
| 11-12 | 69-84 | CNNs & Computer Vision |
| 13-14 | 85-92 | NLP Fundamentals |
| 15-16 | 93-112 | Transformers & LLMs |
| 17-18 | 113-120 | AI Agents |
| 19-20 | 121-140 | MLOps & Deployment |

## Next Steps:

1. **Build Projects** - Apply what you learned to real-world problems
2. **Contribute to Open Source** - Contribute to AI/ML projects on GitHub
3. **Stay Updated** - Follow AI research papers and blogs
4. **Specialize** - Deep dive into areas that interest you most
5. **Network** - Join AI communities and attend meetups

---

**Happy Learning! 🚀**
