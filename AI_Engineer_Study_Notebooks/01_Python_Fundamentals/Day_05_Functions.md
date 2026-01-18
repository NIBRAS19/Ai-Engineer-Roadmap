# Day 5: Functions

## 1. Introduction
Copy-pasting code is the root of all evil in programming. If you need to change a logic, you have to find every place you pasted it.

**Functions** wrap a block of code with a name. You write it once and call it whenever you need it.
In AI, everything is a function: `normalize_data()`, `calculate_loss()`, `train_epoch()`.

---

## 2. Defining a Function
### Syntax
```python
def function_name(parameters):
    """Optional Docstring describing what it does."""
    # Code block
    return value
```

### Example
```python
def calculate_accuracy(correct_predictions, total_predictions):
    """Calculates percentage accuracy."""
    return correct_predictions / total_predictions

# Using the function
acc = calculate_accuracy(95, 100)
print(f"Accuracy: {acc}")  # 0.95
```

---

## 3. Parameters and Arguments

### Positional Arguments
Order matters.
```python
def subtract(a, b):
    return a - b

print(subtract(10, 5)) # 5
print(subtract(5, 10)) # -5
```

### Keyword Arguments
Order doesn't matter if you specify keys.
```python
print(subtract(b=5, a=10)) # 5
```

### Default Parameters
Useful for optional settings (like hyperparameters).
```python
def train_model(epochs=10, learning_rate=0.01):
    print(f"Training for {epochs} epochs at lr={learning_rate}")

train_model()             # Uses defaults: 10, 0.01
train_model(epochs=50)    # 50, 0.01
train_model(learning_rate=0.001) # 10, 0.001
```

---

## 4. `*args` and `**kwargs`
Advanced features for flexible functions.

### `*args` (Arbitrary Arguments)
Accepts any number of positional arguments as a tuple.
```python
def sum_all_losses(*losses):
    return sum(losses)

print(sum_all_losses(0.1, 0.2, 0.05)) # 0.35
```

### `**kwargs` (Arbitrary Keyword Arguments)
Accepts any number of keyword arguments as a dictionary. Useful for passing config to models.
```python
def log_metrics(**metrics):
    for name, value in metrics.items():
        print(f"Metric {name}: {value}")

log_metrics(accuracy=0.9, loss=0.1, precision=0.85)
```

---

## 5. Scope: Local vs Global
Variables created inside a function are **local** (exist only inside). Variables outside are **global**.

```python
x = 10  # Global

def my_func():
    y = 5       # Local
    print(x)    # Can read global
    # print(x + y)

my_func()
# print(y) -> Error! y does not exist here.
```

---

## 6. Lambda Functions (Anonymous Functions)
Small, one-line functions without a name. Often used with `map()` or `filter()`.

**Syntax:** `lambda arguments: expression`

```python
# Regular function
def square(x):
    return x ** 2

# Lambda
square_lambda = lambda x: x ** 2

print(square_lambda(5)) # 25
```

### AI Use Case: Data Transformation
Applying a transformation to a list of file paths.
```python
paths = ["/data/img1.png", "/data/img2.JPG"]

# Normalize extensions to lowercase
normalized = list(map(lambda p: p.lower(), paths))
print(normalized) 
```

---

## 7. Practical Exercises

### Exercise 1: ReLU Function
Write a function `relu(x)` that returns `x` if `x > 0`, else `0`. Test it with values 10 and -5.

### Exercise 2: Model Summary Printer
Write a function `print_model_info` that accepts a `name` (string) and `**kwargs` for architecture details (like `layers=5`, `units=128`). It should print:
"Model: [name]"
followed by the config details line by line.

---

## 8. Summary
- **Functions** promote code reuse and modularity.
- **Parameters** can be positional, keyword, or default.
- **Scope** determines where variables are visible.
- **Lambda** functions are quick, inline operations.

**Next Up:** **Object-Oriented Programming (OOP)**—structuring code into Classes and Objects (the backbone of PyTorch/TensorFlow models).
