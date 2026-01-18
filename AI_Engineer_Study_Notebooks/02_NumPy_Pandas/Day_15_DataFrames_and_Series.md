# Day 15: Pandas DataFrames and Series

## 1. Introduction to Pandas
**Pandas** is the "Excel of Python". It is built on top of NumPy but designed for **Tabular Data** with mixed types (text, numbers, dates).
In AI, Pandas is used for:
- **Data Loading**: Reading CSV, Excel, SQL.
- **Data Cleaning**: Handling missing values, typos.
- **Feature Selection**: Preparing X (Features) and y (Labels).

---

## 2. The Core Structures

### 2.1 Series (1D)
A **Series** is like a column in Excel. It has an **Index** (labels) and **Values** (data).
```python
import pandas as pd

# Creating a Series
prices = pd.Series([100, 200, 150], index=['Apple', 'Banana', 'Orange'])

print(prices)
# Apple     100
# Banana    200
# Orange    150
# dtype: int64

print(prices['Apple']) # 100
```

### 2.2 DataFrame (2D)
A **DataFrame** is a table. It is a dictionary of Series.
```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Salary": [50000, 60000, 70000]
}

df = pd.DataFrame(data)
print(df)
#       Name  Age  Salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   70000
```

---

## 3. Inspeting Data
When you open a dataset with 100,000 rows, you can't print the whole thing.

```python
# 1. View top rows
print(df.head(2)) 

# 2. View bottom rows
print(df.tail(1))

# 3. Check data types and missing values (Crucial!)
print(df.info()) 
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3 entries, 0 to 2
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
#  0   Name    3 non-null      object (Text)
#  1   Age     3 non-null      int64 
#  2   Salary  3 non-null      int64 

# 4. Statistical Summary
print(df.describe())
#              Age        Salary
# mean   30.000000  60000.000000
```

---

## 4. Column Access
You can grab a column as a Series.

```python
# Method 1 (Bracket Notation - Recommended)
ages = df['Age'] 

# Method 2 (Dot Notation - Avoid if column name has spaces)
salaries = df.Salary
```

---

## 5. Practical Exercises

### Exercise 1: Dataset Creation
Create a DataFrame representing an AI Model Leaderboard.
- Columns: `Model` (e.g., GPT-4, Llama-2), `Params_Billion` (float), `Score` (float).
- Add 3 rows of data.
- Print the `info()` of your DataFrame.

### Exercise 2: Quick Analysis
Using your DataFrame:
1.  Extract the `Score` column.
2.  Print the average score (`.mean()`).
3.  Print the maximum parameters (`.max()`).

---

## 6. Summary
- **Series**: Single column.
- **DataFrame**: Full table.
- **`head()`, `info()`, `describe()`**: The "Hello World" commands of any data project.

**Next Up:** **Data Selection**—how to slice tables using Loc and Iloc (the SQL of Python).
