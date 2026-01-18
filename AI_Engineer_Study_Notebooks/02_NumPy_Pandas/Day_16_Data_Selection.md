# Day 16: Data Selection (loc & iloc)

## 1. Introduction
Selecting specific rows and columns is the most frequent task in Data Engineering.
- "Give me all users under age 25."
- "Select the first 100 rows for training."
- "Extract independent variables (X) and target variable (y)."

Pandas uses two powerful indexers: `.loc` (Label-based) and `.iloc` (Index-based).

---

## 2. `.iloc` (Integer Location)
Selects by **position** (like list indices).
**Syntax**: `df.iloc[row_index, col_index]`

```python
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'SF']
})

# 1. Single Element (Row 0, Col 1 -> Age)
print(df.iloc[0, 1])  # 25

# 2. Slice Rows (First 2 rows)
print(df.iloc[0:2]) 

# 3. Slice Columns (All rows, First 2 columns)
print(df.iloc[:, 0:2]) 
```

**Common AI Use Case:** Splitting features.
```python
# Assuming last column is the Target (y)
X = df.iloc[:, :-1] # All columns except last
y = df.iloc[:, -1]  # Only the last column
```

---

## 3. `.loc` (Label Location)
Selects by **name/label** or **boolean condition**.
**Syntax**: `df.loc[row_label, col_name]`

```python
# 1. Select by Column Name
print(df.loc[0, 'Name']) # 'Alice' (Row label is index 0)

# 2. Select multiple columns
print(df.loc[:, ['Name', 'City']])
```

---

## 4. Boolean Indexing (Filtering)
This uses `.loc` implicitly. You filter rows based on a True/False condition.

```python
# Filter: People older than 28
mask = df['Age'] > 28
print(mask) 
# 0    False
# 1    True
# 2    True

adults = df[df['Age'] > 28] 
print(adults)
```

**Complex Filtering:**
Use `&` (AND) and `|` (OR). Parentheses are mandatory.
```python
# Age > 25 AND City is NY
filtered = df[(df['Age'] > 25) & (df['City'] == 'NY')]
```

---

## 5. Practical Exercises

### Exercise 1: Training Set Split
Given a DataFrame with 100 rows.
Use `.iloc` to:
1.  Assign the first 80 rows to `train_df`.
2.  Assign the last 20 rows to `test_df`.

### Exercise 2: Outlier Removal
Data: `df = pd.DataFrame({'val': [10, 20, 500, 30, 40]})`.
Filter the DataFrame to keep only rows where `val` is less than 100.

---

## 6. Summary
- **`.iloc`**: Uses integer positions `[0:5, 1:3]`. Best for slicing arrays.
- **`.loc`**: Uses labels `['Name', 'Age']`. Best for reading specific columns.
- **Boolean Indexing**: `df[df['col'] > 0]`. Best for data cleaning.

**Next Up:** **Handling Missing Data**—Dealing with `NaN`, the archenemy of AI models.
