# Day 20: Merging and Concatenating

## 1. Introduction
Data often lives in different places.
- User info in `users.csv`.
- Transaction logs in `sales.csv`.
To analyze them, you need to combine them into one DataFrame.

---

## 2. Concatenating (`pd.concat`)
Sticking two dataframes together (Top-to-Bottom or Side-by-Side).

```python
import pandas as pd
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Stack vertically (Row-wise) - Default axis=0
# Matches columns, adds rows
result = pd.concat([df1, df2])
print(result)
```

---

## 3. Merging (`pd.merge`)
Combining based on a **Key** (like SQL JOIN).

```python
users = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
emails = pd.DataFrame({'id': [1, 2], 'email': ['a@a.com', 'b@b.com']})

# Inner Join (Only ids present in BOTH)
# Charlie (id 3) is dropped because he has no email.
merged = pd.merge(users, emails, on='id', how='inner')
print(merged)
```

### Join Types (`how=`)
- `'inner'`: Intersection (Only matching keys).
- `'left'`: Keep all Left keys, fill missing Right with NaN.
- `'right'`: Keep all Right keys.
- `'outer'`: Union (Keep all keys).

```python
# Left Join (Keep Charlie, email will be NaN)
left_merged = pd.merge(users, emails, on='id', how='left')
```

---

## 4. Practical Exercises

### Exercise 1: Building a Dataset
Create two dataframes.
1.  `products`: columns `id`, `name`.
2.  `prices`: columns `id`, `price`.
Merge them so every product has a price. If a product has no price, it should still be in the table (Left Join).

---

## 5. Summary
- **Concat**: "Stacking" tables.
- **Merge**: "Joining" tables based on logical keys (`id`).

**Next Up:** **Reading & Writing Data**—Interacting with CSVs, Excel, and JSON files.
