# Day 19: Feature Engineering

## 1. Introduction
Models can only read numbers.
If your data is "Red", "Green", "Blue", a Neural Network cannot process it.
**Feature Engineering** is the art of creating new input features from existing data to improve model performance.

---

## 2. Applying Functions (`.apply`)
The most versatile tool. Transform a column using any custom function.

```python
import pandas as pd
df = pd.DataFrame({'text': ['hello world', 'AI is cool', 'pandas rocks']})

# Feature: Text Length
df['char_count'] = df['text'].apply(len)
print(df)
#            text  char_count
# 0   hello world          11
# 1    AI is cool          10
# 2  pandas rocks          12
```

**Lambda Integration:**
```python
# Convert to simple normalized form
df['clean_text'] = df['text'].apply(lambda x: x.lower().strip())
```

---

## 3. Dealing with Categories (One-Hot Encoding)
Converting text labels into binary vectors.
"Red", "Green" -> [1, 0], [0, 1]

```python
data = pd.DataFrame({'color': ['Red', 'Blue', 'Green', 'Red']})

# Get Dummies
encoded = pd.get_dummies(data['color'])
print(encoded)
#    Blue  Green  Red
# 0     0      0    1
# 1     1      0    0
# ...
```

---

## 4. Binning
Converting continuous numbers into discrete buckets.
Age 18, 19, 21 -> "Young".
Age 60, 65 -> "Senior".

```python
ages = pd.DataFrame({'age': [10, 25, 40, 80]})
labels = ['Child', 'Adult', 'Senior']
bins = [0, 18, 60, 100]

ages['category'] = pd.cut(ages['age'], bins=bins, labels=labels)
print(ages)
```

---

## 5. Practical Exercises

### Exercise 1: Salary Normalization
Data: `df = pd.DataFrame({'salary': [50000, 100000, 75000]})`
Create a new column `salary_k` by dividing the salary by 1000. Use `.apply` or simple division.

### Exercise 2: Encoding
Data: `df = pd.DataFrame({'size': ['S', 'M', 'L', 'M']})`
Map these manually: S=1, M=2, L=3.
*Hint: Use `df['size'].map({'S': 1, ...})`*

---

## 6. Summary
- **`.apply()`**: Custom transformations.
- **`get_dummies()`**: One-Hot Encoding for text categories.
- **`.map()`** and **`.cut()`**: Mapping values and binning.

**Next Up:** **Merging Dataframes**—Joining multiple tables (SQL Joins).
