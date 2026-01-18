# Day 4: Loops (For, While, List Comprehensions)

## 1. Introduction
If you have 1,000 images to process, you don't write 1,000 lines of code. You write **one loop** that runs 1,000 times.
Loops allow computers to do what they do best: repetitive tasks at lightning speed.

In AI, loops are everywhere:
- Iterating over **training epochs**.
- Iterating over **batches** of data.
- Updating **weights** in a model.

---

## 2. The `for` Loop
Used for iterating over a sequence (list, tuple, string, range).

### Syntax
```python
for item in sequence:
    # Do something with item
```

### Example: Iterating over Hyperparameters
```python
learning_rates = [0.1, 0.01, 0.001]

for lr in learning_rates:
    print(f"Training model with Learning Rate: {lr}")
    # train_model(lr) ...
```

### The `range()` Function
Generates a sequence of numbers. Essential for running a loop a specific number of times.
- `range(stop)`: 0 to stop-1
- `range(start, stop)`: start to stop-1
- `range(start, stop, step)`

```python
# Epoch Loop
epochs = 5
for i in range(epochs):
    print(f"Epoch {i+1}/{epochs} completed.")
```

### Enumerate
When you need both the **index** and the **value**.
```python
filenames = ["img1.jpg", "img2.jpg", "img3.jpg"]

for index, name in enumerate(filenames):
    print(f"Processing image {index}: {name}")
```

---

## 3. The `while` Loop
Repeats code **while a condition is True**. Use this when you don't know exactly how many iterations you need (e.g., "train until loss < 0.01").

### Example
```python
loss = 1.0
threshold = 0.1
epoch = 0

while loss > threshold:
    # Simulate training step
    loss -= 0.2  
    epoch += 1
    print(f"Epoch {epoch}: Loss is {loss:.2f}")

print("Training stopped. Target loss reached.")
```

**Warning:** Infinite loops!
If `loss` never drops below `threshold`, this runs forever. Always ensure the condition eventually becomes False or use a `break`.

---

## 4. Loop Control Statements

### `break`
Exits the loop immediately.
```python
# Early Stopping: Stop if accuracy stops improving
for epoch in range(100):
    # ... train ...
    if val_loss_increasing:
        print("Early stopping triggered!")
        break
```

### `continue`
Skips the rest of the current iteration and jumps to the next one.
```python
# Skip corrupted images
images = ["cat.jpg", "corrupted.png", "dog.jpg"]

for img in images:
    if img == "corrupted.png":
        print("Skipping corrupted file...")
        continue
    
    print(f"Training on {img}")
```

---

## 5. List Comprehensions (The Pythonic Way)
A concise way to create lists. It's faster and cleaner than standard `for` loops.

### Syntax
`[expression for item in iterable if condition]`

### Example: Squaring Numbers
```python
# Traditional Way
nums = [1, 2, 3, 4]
squares = []
for n in nums:
    squares.append(n ** 2)

# List Comprehension Way
squares = [n ** 2 for n in nums]
print(squares) # [1, 4, 9, 16]
```

### Example: Filtering Data
Keep only positive values (ReLU logic).
```python
activations = [-0.5, 1.2, -0.1, 2.5]
positive_activations = [x for x in activations if x > 0]
print(positive_activations) # [1.2, 2.5]
```

---

## 6. Practical Exercises

### Exercise 1: Element-wise Operations
You have two lists of equal length representing dataset split sizes:
`train_sizes = [100, 200, 300]`
`val_sizes = [20, 40, 60]`

Write a loop to calculate the total size (`train + val`) for each pair and print it.
*Hint: Use `zip(list1, list2)` to iterate two lists at once.*

### Exercise 2: Squares of Evens
Use a **List Comprehension** to create a list of squares for all **even** numbers between 1 and 20.

---

## 7. Summary
- **For Loops**: Iterate over sequences (lists, ranges). Best for fixed iterations (Epochs).
- **While Loops**: Iterate until a condition changes. Best for convergence (Loss targets).
- **Control**: `break` to stop, `continue` to skip.
- **List Comprehensions**: The elegant, fast way to transform lists.

**Next Up:** **Functions**—packaging your code into reusable blocks.
