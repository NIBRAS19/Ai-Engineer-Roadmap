# Day 18: GroupBy and Aggregations

## 1. Introduction
Sometimes you don't care about individual rows; you care about groups.
- "What is the average sales **per store**?"
- "What is the max accuracy **per model architecture per hyperparameter setting**?"

Pandas `groupby` works exactly like SQL `GROUP BY`.
Pattern: **Split -> Apply -> Combine**.

### 🎯 Real-World Analogy: Sorting Laundry
> Think of GroupBy like sorting laundry. You **split** clothes into piles (whites, colors, delicates), **apply** an action to each pile (wash at different temperatures), and **combine** them back into a clean wardrobe. You don't wash each sock individually—you work in batches.

---

## 2. Basic GroupBy

```python
import pandas as pd
df = pd.DataFrame({
    'Company': ['Google', 'Google', 'MSFT', 'MSFT', 'FB', 'FB'],
    'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
    'Sales': [200, 120, 340, 124, 243, 350]
})

# Group object (lazy evaluation)
by_comp = df.groupby('Company')

# Apply aggregation
print(by_comp.mean(numeric_only=True))
#          Sales
# Company       
# FB       296.5
# Google   160.0
# MSFT     232.0
```

---

## 3. Common Aggregations
- `mean()`: Average
- `sum()`: Total
- `count()`: Frequency
- `std()`: Standard Deviation (Volatility)
- `max()` / `min()`
- `first()` / `last()`: First or last row per group

```python
# Multiple stats at once
print(by_comp.describe())
```

---

## 4. `agg()` Function (Custom Aggregations)
Apply different functions to different columns.

```python
# Sum of sales, but also Count of employees involved
print(by_comp.agg({'Sales': ['sum', 'mean'], 'Person': 'count'}))
```

---

## 5. Multi-Level GroupBy (Hierarchical Grouping)
Group by **multiple columns** for deeper analysis.

### AI Use Case: Model Performance Analysis
```python
# Experiment results: Model architecture + Learning Rate -> Accuracy
experiments = pd.DataFrame({
    'Model': ['ResNet', 'ResNet', 'ResNet', 'ViT', 'ViT', 'ViT'],
    'LearningRate': [0.001, 0.01, 0.001, 0.001, 0.01, 0.001],
    'Epoch': [10, 10, 20, 10, 10, 20],
    'Accuracy': [0.85, 0.78, 0.91, 0.88, 0.82, 0.93]
})

# Group by Model AND LearningRate
grouped = experiments.groupby(['Model', 'LearningRate'])

print(grouped['Accuracy'].mean())
# Model   LearningRate
# ResNet  0.001           0.880   <- Average of epochs 10 & 20
#         0.010           0.780
# ViT     0.001           0.905
#         0.010           0.820
```

### Accessing Multi-Level Groups
```python
# Get specific subgroup
resnet_slow = grouped.get_group(('ResNet', 0.001))
print(resnet_slow)
```

### Unstacking for Readability
```python
# Convert hierarchical index to columns (like a pivot table)
pivot = grouped['Accuracy'].mean().unstack()
print(pivot)
# LearningRate    0.001   0.010
# Model                     
# ResNet         0.880   0.780
# ViT            0.905   0.820
```

---

## 6. `transform()` (Keep Original Shape)
Unlike `agg()` which collapses groups, `transform()` returns a Series with the **same length** as the original DataFrame. Perfect for adding group-level statistics back to rows.

### AI Use Case: Normalize Features Per Group (Z-Score)
```python
# Normalize sales within each company (compare employees fairly)
df['Sales_ZScore'] = df.groupby('Company')['Sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(df)
#   Company   Person  Sales  Sales_ZScore
# 0  Google      Sam    200      0.707107
# 1  Google  Charlie    120     -0.707107  <- Below company average
# 2    MSFT      Amy    340      0.707107
# ...
```

### Common Transform Use Cases:
- **Per-group normalization**: As shown above
- **Per-group ranking**: `transform('rank')`
- **Fill missing with group mean**: `transform(lambda x: x.fillna(x.mean()))`

---

## 7. `apply()` (Custom Operations)
When you need more flexibility than `agg()` or `transform()`.

```python
# Custom function: Return the row with max sales per company
def top_performer(group):
    return group.loc[group['Sales'].idxmax()]

top_employees = df.groupby('Company').apply(top_performer, include_groups=False)
print(top_employees)
```

---

## 8. Practical Exercises

### Exercise 1: Sales Analysis
Using the dataframe above:
1.  Find the **total** sales for each company.
2.  Find the company with the highest single sale (`max`).

### Exercise 2: Multi-Level Model Analysis
Using the `experiments` dataframe:
1.  Find the average accuracy for each **Model + Epoch** combination.
2.  Which hyperparameter setting (Model + LearningRate) has the highest accuracy?

### Exercise 3: Normalization with Transform
Given training data with features from different sensors (each sensor has different scales):
```python
sensor_data = pd.DataFrame({
    'Sensor': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Reading': [100, 150, 120, 5000, 5500, 5200]
})
```
Use `groupby` + `transform` to normalize each sensor's readings to have mean=0 and std=1.

---

## 9. Summary
- **GroupBy**: Splits data into buckets based on a category.
- **Aggregations**: Summarizes those buckets into single numbers.
- **Multi-Level GroupBy**: Use `groupby(['col1', 'col2'])` for deeper analysis.
- **Transform**: Apply group-level operations while keeping original DataFrame shape.
- **Apply**: Maximum flexibility for custom group operations.

Vital for **Business Intelligence**, model experiment tracking, and feature engineering.

**Next Up:** **Feature Engineering**—transforming data into a format that ML models can understand (e.g., converting "Male/Female" to 0/1).

