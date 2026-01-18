# Day 17: Handling Missing Data

## 1. Introduction
Real-world data is dirty. Sensors fail, users skip form fields, logs get corrupted.
Missing data usually appears as `NaN` (Not a Number) or `None`.
**AI Rule:** You cannot feed `NaN` into a mathematical model (it will output `NaN` everywhere). You must clean it.

---

## 2. Detecting Missing Data
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan],
    'B': [5, np.nan, np.nan],
    'C': [1, 2, 3]
})

# Check for nulls
print(df.isnull())       # DataFrame of True/False
print(df.isnull().sum()) # Count nulls per column
# A    1
# B    2
# C    0
```

---

## 3. Strategy 1: Deletion
If you have massive data and only a few missing rows, just drop them.

```python
# Drop rows with ANY missing value
clean_rows = df.dropna(axis=0)

# Drop columns with ANY missing value
clean_cols = df.dropna(axis=1)

# Drop only if specific column is missing
clean_a = df.dropna(subset=['A'])
```

**Pros:** Simple, truthful.
**Cons:** You lose data. If you have little data, this kills your model accuracy.

---

## 4. Strategy 2: Imputation (Filling)
Guessing the missing value based on other data.

```python
# 1. Fill with a constant (Good for categorical "Unknown")
df_filled = df.fillna(0)

# 2. Mean Imputation (Standard for features like Age, Price)
mean_val = df['A'].mean()
df['A'] = df['A'].fillna(mean_val)

# 3. Forward Fill (Good for Time Series)
# Uses the previous day's value to fill today's missing one.
ts_df = df.fillna(method='ffill')
```

---

## 5. Practical Exercises

### Exercise 1: Cleaning a Survey
Data: `survey = pd.DataFrame({'age': [25, 30, np.nan, 22], 'city': ['NY', np.nan, 'SF', 'NY']})`
1.  Check how many values are missing.
2.  Fill missing 'age' with the **average age**.
3.  Drop rows where 'city' is missing.

---

## 6. Summary
- **Detect**: `isnull().sum()`
- **Drop**: `dropna()` (Aggressive)
- **Fill**: `fillna(mean)` (Conservative, best for ML)

**Next Up:** **GroupBy and Aggregations**—Summarizing data to find trends (e.g., getting average salary per department).
