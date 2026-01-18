# Day 1: Python Variables & Data Types

## 1. Introduction
Welcome to Day 1 of your AI Engineering journey! Today, we lay the foundation of Python programming by understanding **Variables** and **Data Types**. These are the building blocks of every program you will write, from simple scripts to complex Neural Networks.

In AI, data is everything. Understanding how to store (variables) and categorize (data types) that data is crucial for:
- Preprocessing datasets (cleaning text, handling numbers).
- Feeding data into models (inputs must be specific types).
- Interpreting model outputs (probabilities, classes).

---

## 2. Variables
### What is a Variable?
A variable is a container for storing data values. In Python, variables are "labels" or "pointers" to objects in memory. Unlike languages like C++ or Java, Python is **dynamically typed**, meaning you don't need to declare the type of a variable explicitly.

### Syntax
```python
variable_name = value
```
- `=` is the assignment operator.
- `variable_name` is the identifier.
- `value` is the data you want to store.

### Naming Conventions (PEP 8)
- Use **snake_case** for variable names (e.g., `user_name`, `learning_rate`).
- Names should be descriptive. `x` is bad; `customer_age` is good.
- Case-sensitive: `Age` and `age` are different variables.

### Deep Dive: Memory Management
When you say `a = 10`, Python creates an integer object `10` in memory and makes the name `a` point to it. If you say `b = a`, `b` points to the same object.

### Real-World Example: E-commerce User Profile
Imagine you are building a recommendation system. You need to store user details.
```python
# Poor Variable Naming
n = "Alice" 
p = 120.5
a = True

# Professional Variable Naming (Self-documenting)
user_full_name = "Alice Smith"
cart_total_amount = 120.50
is_premium_member = True
```

---

## 3. Data Types
Python has several built-in data types. We will focus on the primitive ones today.

### 3.1 Integers (`int`)
Whole numbers, positive or negative, without decimals.
- **Why used:** Counting items, iterations (loops), indices, discrete categories.
- **AI Context:** Epoch numbers, batch sizes, image pixel values (0-255).

```python
# Examples
epoch_count = 50
batch_size = 32
hidden_layers = -1  # Negative numbers allowed, though context matters
```

### 3.2 Floating-Point Numbers (`float`)
Numbers with a decimal point.
- **Why used:** Precision, measurements, probabilities, weights in Neural Networks.
- **AI Context:** Loss values (e.g., 0.0023), model weights, accuracy scores.

```python
# Examples
learning_rate = 0.001
model_accuracy = 0.985
temperature_celsius = 25.5
```

### 3.3 Strings (`str`)
Sequence of characters enclosed in quotes.
- **Why used:** Text processing, labels, file paths.
- **AI Context:** NLP (Natural Language Processing) inputs, class labels (e.g., "cat", "dog"), dataset paths.

```python
# Examples
model_name = "ResNet50"
dataset_path = "./data/images/"
sentiment_label = 'Positive'  # Single or double quotes work
```

**String Operations:**
```python
text = "Artificial Intelligence"
print(text.lower())  # 'artificial intelligence' (Normalization)
print(len(text))     # 23 (Feature extraction: length of text)
```

### 3.4 Booleans (`bool`)
Represents `True` or `False`.
- **Why used:** Logic flow, binary flags.
- **AI Context:** Binary classification targets (0/1), enabling/disabling training modes (e.g., `training=True`).

```python
# Examples
is_model_trained = False
use_gpu = True
```

---

## 4. Type Conversion (Casting)
Sometimes you need to treat one data type as another. This is common in data cleaning.

### Implicit Conversion
Python automatically converts types when it makes sense (e.g., adding int + float = float).
```python
result = 10 + 2.5  # 12.5 (float)
```

### Explicit Conversion
You force the change.
- `int()`: Truncates decimals.
- `float()`: Adds a decimal.
- `str()`: Converts into text representation.

```python
# Scenario: Reading inputs (usually strings) and using them for math
age_input = "25"
# next_year = age_input + 1  # This causes an ERROR (str + int)

# Correct way
age_int = int(age_input)
next_year = age_int + 1
print(next_year)  # 26

# Scenario: Concatenating numbers to messages
accuracy = 0.95
print("Model Accuracy: " + str(accuracy))  # "Model Accuracy: 0.95"
```

---

## 5. Practical Exercises

### Exercise 1: The Data Scientist's Variables
Create variables to store the following information for a Machine Learning experiment:
1.  Name of the dataset (String)
2.  Number of samples (Integer)
3.  Test split ratio (Float, e.g., 0.2)
4.  Whether the data is clean (Boolean)

Print the type of each variable using `type()`.

### Exercise 2: Data Cleaning Simulation
You received a price value as a string: `price_str = "199.99"`.
1.  Convert it to a float.
2.  Convert it to an integer (representing just the dollar part).
3.  Print both values.

---

## 6. Summary
- **Variables** store data; names should be descriptive.
- **Integers** are for counts; **Floats** are for precision/math.
- **Strings** are for text; **Booleans** are for logic.
- **Type Conversion** allows you to transform data formats, essential for preparing data for AI models.

**Next Up:** We will explore **Python Collections** (Lists, Dictionaries) to manage groups of data efficiently!
