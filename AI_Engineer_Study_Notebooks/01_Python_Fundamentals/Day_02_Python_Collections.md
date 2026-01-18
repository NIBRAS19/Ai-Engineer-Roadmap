# Day 2: Python Collections (Lists, Tuples, Dictionaries, Sets)

## 1. Introduction
Yesterday, we stored single values (numbers, strings). In the real world, and especially in AI, we deal with **collections** of data—pixel values in an image, words in a sentence, or rows in a dataset.

Today, we master the four built-in collection types in Python. Choosing the right one affects the performance and readability of your code.

| Type | Computed As | Mutable? | Ordered? | Duplicates? | Use Case |
|------|-------------|----------|----------|-------------|----------|
| **List** | `[]` | Yes | Yes | Yes | Sequences of data (e.g., time series) |
| **Tuple** | `()` | No | Yes | Yes | Fixed data (e.g., image dimensions) |
| **Dict** | `{}` | Yes | No* | No (Keys) | Key-Value pairs (e.g., database records) |
| **Set** | `{}` | Yes | No | No | Unique items (e.g., vocabulary) |

*\*Dictionaries maintain insertion order in Python 3.7+.*

---

## 2. Lists
A generic, ordered bucket for items. Most common structure in Python.

### Basics
```python
# Creating a list of loss values from training
losses = [0.9, 0.75, 0.6, 0.45]
mixed_list = [1, "Category_A", 0.5, True]
```

### Essential Operations
```python
# 1. Accessing (Indexing)
print(losses[0])    # 0.9 (First item)
print(losses[-1])   # 0.45 (Last item)

# 2. Slicing (Vital for data splitting)
# syntax: [start:stop:step]
print(losses[0:2])  # [0.9, 0.75] - Start inclusive, stop exclusive

# 3. Modifying
losses.append(0.3)  # Adds to end -> [0.9, ... 0.3]
losses[1] = 0.7     # Updates index 1
```

### Deep Dive: Memory & Performance
Lists are dynamic arrays. Adding to the end (`append`) is fast ($O(1)$). Inserting in the middle (`insert`) is slow ($O(n)$) because all subsequent elements must shift.
**Tip:** In Deep Learning (PyTorch), we often append training losses to a list and then plot them.

---

## 3. Tuples
Immutable lists. Once defined, they cannot be changed.

### Why use them?
1.  **Safety:** Ensures data isn't accidentally modified.
2.  **Performance:** Slightly faster than lists.
3.  **Context:** Functions often return multiple values as tuples.

### Basics
```python
# Image shape: (Channels, Height, Width)
image_dims = (3, 224, 224)

# Accessing
print(image_dims[1])  # 224

# Unpacking (Very common in AI)
c, h, w = image_dims
print(h)  # 224
```

---

## 4. Dictionaries
Key-Value pairs. The most powerful structure for structured data.

### Basics
```python
# A single data sample
student_data = {
    "id": 101,
    "name": "Alice",
    "scores": [85, 90, 88],
    "passed": True
}
```

### Essential Operations
```python
# Accessing
print(student_data["name"])  # 'Alice'
print(student_data.get("age", 0))  # Returns 0 if "age" key is missing (Safe!)

# Modifying
student_data["grade"] = "A"  # New key
student_data["passed"] = False  # Update existing

# Iterating (Crucial)
for key, value in student_data.items():
    print(f"{key}: {value}")
```

### AI Context: JSON Data
Dictionaries map 1:1 with JSON (JavaScript Object Notation), which is the standard format for web APIs, config files, and saving model metadata.

---

## 5. Sets
Unordered collection of **unique** elements.

### Basics
```python
tags = {"AI", "ML", "Data", "AI"} 
print(tags)  # {'AI', 'ML', 'Data'} (Duplicates removed automatically)
```

### Why used in AI?
1.  **Vocabulary building:** In NLP, finding the set of unique words in a text corpus.
2.  **Fast Lookups:** Checking `if item in my_set` is incredibly fast ($O(1)$) compared to lists ($O(n)$).

---

## 6. Real-World AI Example: Preprocessing a Dataset

Imagine you have raw labels from a classification task.
```python
raw_labels = ["cat", "dog", "cat", "bird", "dog", "dog"]

# 1. Get unique classes
unique_classes = set(raw_labels)  # {'cat', 'dog', 'bird'}

# 2. Assign an ID to each class (for the model)
class_to_idx = {
    "cat": 0,
    "dog": 1,
    "bird": 2
}

# 3. Convert all raw labels to numbers
numeric_labels = []
for label in raw_labels:
    numeric_labels.append(class_to_idx[label])

print(numeric_labels)  # [0, 1, 0, 2, 1, 1]
```
*Note: We usually do this with libraries like Scikit-Learn, but understanding the logic is vital.*

---

## 7. Practical Exercises

### Exercise 1: List Slicing
Given `features = [0.1, 0.5, 0.2, 0.8, 0.9, 1.0]`:
1.  Extract the first 3 features.
2.  Extract the last 2 features.
3.  Reverse the list using slicing.

### Exercise 2: Dictionary Lookup
Create a dictionary representing a Neural Network configuration:
- `layers`: 5
- `activation`: "relu"
- `dropout`: 0.5

Write code to print the activation function. Then add a new key `optimizer` with value "adam".

---

## 8. Summary
- **Lists**: Ordered, mutable sequence. The go-to for data arrays.
- **Tuples**: Immutable sequence. Used for fixed configs/shapes.
- **Dictionaries**: Key-Value mapping. Used for objects, configs, and JSON-like data.
- **Sets**: Unique items. Used for removing duplicates and fast checks.

**Next Up:** **Control Flow**—teaching our programs to make decisions (`if`, `else`).
