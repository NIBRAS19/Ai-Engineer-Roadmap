# Day 35: Decision Trees

## 1. Introduction
A Decision Tree makes predictions by asking a sequence of questions.
"Is it raining?" -> Yes -> "Is it windy?" -> No -> **Go Outside**.

### 🎯 Real-World Analogy: 20 Questions Game
> Decision Trees work exactly like the game "20 Questions." You ask yes/no questions that split possibilities in half. The best questions are the ones that **eliminate the most uncertainty**. "Is it alive?" is better than "Is it a blue elephant?"

**Pros**: Interpretability (Easy to explain to stakeholders).
**Cons**: Prone to Overfitting (Memorizing the tree to training data).

---

## 2. How Trees Learn: Splitting Criteria

The tree chooses questions (splits) that create the **purest** groups.
"Pure" = All samples in a group belong to the same class.

### 2.1 Entropy (Information Theory)

**Entropy** measures the "messiness" or "uncertainty" of a group.

$$ Entropy(S) = -\sum_{c} p_c \log_2(p_c) $$

Where $p_c$ is the proportion of class $c$.

| Distribution | Entropy | Interpretation |
|:-------------|:--------|:---------------|
| 50% Cat, 50% Dog | 1.0 | Maximum uncertainty |
| 90% Cat, 10% Dog | 0.47 | Low uncertainty |
| 100% Cat, 0% Dog | 0.0 | Perfect purity |

```python
import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))  # +1e-10 to avoid log(0)

# Example
y_mixed = [0, 0, 1, 1]      # 50-50 split
y_pure = [1, 1, 1, 1]       # All same class
print(f"Mixed entropy: {entropy(y_mixed):.2f}")   # 1.0
print(f"Pure entropy: {entropy(y_pure):.2f}")     # 0.0
```

### 2.2 Information Gain

**Information Gain** measures how much a split **reduces** entropy.

$$ IG(S, A) = Entropy(S) - \sum_{v} \frac{|S_v|}{|S|} \cdot Entropy(S_v) $$

- $S$: Parent node's samples
- $A$: The feature we're splitting on
- $S_v$: Samples in child node $v$

The tree picks the split with the **highest Information Gain**.

### 2.3 Gini Impurity (The Practical Choice)

Sklearn uses **Gini Impurity** by default (faster to compute, similar results).

$$ Gini(S) = 1 - \sum_{c} p_c^2 $$

| Distribution | Gini | Interpretation |
|:-------------|:-----|:---------------|
| 50% Cat, 50% Dog | 0.5 | Maximum impurity |
| 100% Cat | 0.0 | Perfect purity |

```python
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)
```

---

## 3. Implementation

Works for both Classification and Regression.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train
clf = DecisionTreeClassifier(max_depth=3, criterion='gini')  # or 'entropy'
clf.fit(X, y)

# Visualize
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=['Age', 'Income', 'Student'], 
          class_names=['No', 'Yes'])
plt.show()
```

---

## 4. Overfitting and Pruning

### The Problem: Overfitting
Without constraints, a tree will keep splitting until each leaf has exactly one sample. This achieves 100% training accuracy but terrible test accuracy.

### 🎯 Analogy: The Overachieving Student
> An overfitting tree is like a student who memorizes the exact wording of practice problems instead of understanding the concepts. When exam questions are slightly different, they fail.

### Solution: Pruning

**Pre-Pruning** (Stop Growing Early):
| Hyperparameter | Effect |
|:---------------|:-------|
| `max_depth` | Maximum tree depth (most important!) |
| `min_samples_split` | Minimum samples needed to create a split |
| `min_samples_leaf` | Minimum samples in each leaf |
| `max_features` | Number of features to consider per split |

**Post-Pruning** (Grow Full Tree, Then Cut Back):
- `ccp_alpha`: Cost-Complexity Pruning. Higher = simpler tree.

```python
# Pre-pruning
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

# Post-pruning (Cost-Complexity Pruning)
path = clf.cost_complexity_pruning_path(X_train, y_train)
clf_pruned = DecisionTreeClassifier(ccp_alpha=0.01)
clf_pruned.fit(X_train, y_train)
```

---

## 5. Feature Importance

Trees tell you which features matter most!

```python
clf.fit(X_train, y_train)

# Feature importances (higher = more useful)
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Example output:
# Income: 0.45
# Age: 0.35
# Student: 0.20
```

---

## 6. Trees vs Other Models

| Aspect | Decision Tree | Linear Models |
|:-------|:--------------|:--------------|
| Interpretability | ✅ Excellent | ⚠️ Good (coefficients) |
| Non-linear relationships | ✅ Natural | ❌ Needs feature engineering |
| Robustness | ❌ High variance | ✅ Stable |
| Ensemble potential | ✅ Random Forest, XGBoost | ⚠️ Limited |

**Decision Trees are the building blocks for:** Random Forest (Day 38), Gradient Boosting (Day 39), XGBoost, LightGBM.

---

## 7. Practical Exercises

### Exercise 1: The Overfitting Tree
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Overfit tree
overfit = DecisionTreeClassifier(max_depth=None)
overfit.fit(X_train, y_train)
print(f"Overfit - Train: {overfit.score(X_train, y_train):.2f}, Test: {overfit.score(X_test, y_test):.2f}")

# Pruned tree
pruned = DecisionTreeClassifier(max_depth=3)
pruned.fit(X_train, y_train)
print(f"Pruned - Train: {pruned.score(X_train, y_train):.2f}, Test: {pruned.score(X_test, y_test):.2f}")
```

### Exercise 2: Calculate Information Gain by Hand
Given:
- Parent: 8 samples (5 Yes, 3 No)
- Left child: 4 samples (4 Yes, 0 No)
- Right child: 4 samples (1 Yes, 3 No)

Calculate:
1. Entropy of parent
2. Weighted entropy of children
3. Information Gain

### Exercise 3: Feature Importance Analysis
Train a decision tree on a real dataset (e.g., Iris or Breast Cancer).
Plot the feature importances as a bar chart.

---

## 8. Summary
- **Decision Tree**: A flowchart of if-then-else decisions.
- **Splitting Criteria**: Entropy/Information Gain or Gini Impurity.
- **Overfitting**: Trees love to memorize—use pruning!
- **Feature Importance**: Trees reveal which features matter.
- **Foundation**: Building block for powerful ensembles (Random Forest, XGBoost).

**Next Up:** **KNN**—Learning by similarity.

