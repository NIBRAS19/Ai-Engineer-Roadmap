# Day 7: File Handling & Exception Handling

## 1. Introduction
AI models need Data. That data lives in files (CSVs, images, JSONs, logs).
Also, training a model can take days. If your code crashes 20 hours in because of a tiny error, you lose everything. **Exception Handling** prevents that.

---

## 2. Reading and Writing Files

### The `open()` Function
- Modes: `'r'` (read), `'w'` (write - overwrites!), `'a'` (append).

### The `with` Statement (Context Manager)
Always use `with`. It automatically closes the file even if errors occur.

### Example: Writing Training Logs
```python
# Writing
log_message = "Epoch 1: Loss 0.5\nEpoch 2: Loss 0.3\n"

with open("training_log.txt", "w") as file:
    file.write(log_message)

# Appending
with open("training_log.txt", "a") as file:
    file.write("Epoch 3: Loss 0.1\n")
```

### Example: Reading Data
```python
with open("training_log.txt", "r") as file:
    content = file.read() # Reads whole file
    print(content)

# Reading line by line (Memory Efficient)
with open("training_log.txt", "r") as file:
    for line in file:
        print(f"Log: {line.strip()}")
```

---

## 3. Exception Handling (`try`, `except`)
Catch errors before they kill your program.

### Syntax
```python
try:
    # Risky code
except ErrorType:
    # Error handling code
finally:
    # Runs no matter what (optional)
```

### Common AI Errors
- `FileNotFoundError`: Loading a dataset that doesn't exist.
- `ZeroDivisionError`: Dividing by zero (e.g., in metrics).
- `KeyError`: Dictionary key missing.

### Example: Robust Data Loading
```python
filename = "data.csv"

try:
    with open(filename, "r") as f:
        data = f.read()
    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: {filename} not found! creating empty dataset...")
    data = ""

except Exception as e:
    print(f"Unknown error occurred: {e}")

finally:
    print("Loading process finished.")
```

---

## 4. Working with JSON (JavaScript Object Notation)
Standard format for configurations and metadata.

```python
import json

# Saving a Model Config
config = {
    "layers": 50,
    "model_name": "ResNet",
    "dropout": 0.5
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

# Loading a Config
with open("config.json", "r") as f:
    loaded_config = json.load(f)

print(loaded_config["model_name"]) # ResNet
```

---

## 5. Practical Exercises

### Exercise 1: reliable_divider
Write a function `safe_divide(a, b)` that returns `a/b`. Use `try-except` to handle division by zero. If a ZeroDivisionError occurs, return `0` and print a warning.

### Exercise 2: Dataset Logger
1.  Create a list of strings: `["img1.jpg", "img2.jpg", "img3.jpg"]`.
2.  Write them to a file `dataset.txt`, each on a new line.
3.  Read the file back and print the total number of lines (images).

---

## 6. Summary
- **Files**: Use `with open(...)` to read/write safely.
- **Exceptions**: Use `try...except` to handle errors gracefully and keep your experimental runs alive.
- **JSON**: The standard for saving non-tabular data (configs).

**CONGRATULATIONS!** You have completed **Week 1: Python Fundamentals**.
You now possess the core programming skills needed for AI.
**Next Week:** We start the real math-heavy lifting with **NumPy**—the engine of Data Science.
