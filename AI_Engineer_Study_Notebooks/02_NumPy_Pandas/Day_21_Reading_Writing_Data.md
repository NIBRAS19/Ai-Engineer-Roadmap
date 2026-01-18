# Day 21: Reading and Writing Data

## 1. Introduction
The first step of any project: Loading the data.
The last step of any project: Saving the results.
Pandas supports almost every format: CSV, Excel, SQL, JSON, Parquet, HDF5.

---

## 2. Reading Data

### CSV (Comma Separated Values)
The standard for ML datasets.
```python
# Read
# header=0 means the first row contains column names
df = pd.read_csv("data.csv", header=0)

# Parsing Dates immediately
df = pd.read_csv("sales.csv", parse_dates=['Date'])
```

### Excel
Requires `openpyxl`.
```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
```

### JSON
```python
df = pd.read_json("data.json")
```

---

## 3. Writing Data
Saving your cleaned DataFrame.

```python
# Index=False (Crucial!)
# Prevents Pandas from writing a "0, 1, 2..." column for the index.
df.to_csv("clean_data.csv", index=False)

df.to_excel("report.xlsx", index=False)
```

---

## 4. Advanced: Chunky Loading
When your dataset is 10GB but you have 8GB RAM.
You read it in chunks.

```python
chunk_size = 1000
for chunk in pd.read_csv("massive_file.csv", chunksize=chunk_size):
    # Process 1000 rows at a time
    print(chunk.shape)
```

---

## 5. Practical Exercises

### Exercise 1: Pipeline Simulation
1.  Create a DataFrame `df`.
2.  Save it to `temp.csv` (index=False).
3.  Read it back into `df_loaded`.
4.  Assert that `df` equals `df_loaded` (sanity check).

---

## 6. Summary
- **Input**: `pd.read_csv()`, `pd.read_json()`.
- **Output**: `df.to_csv(index=False)`.
- **Memory**: Use `chunksize` for big data.

**CONGRATULATIONS!** You have finished **Week 3: Pandas**.
You know how to Load, Clean, Filter, Group, and Save data.
**Next Week:** **Mathematics for AI**—Calculus, Linear Algebra, and Probability.
