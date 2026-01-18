# Day 3: Control Flow Statements (If, Else, Logic)

## 1. Introduction
Programming is not just a linear sequence of instructions; it involves making decisions. **Control Flow** allows your program to execute different code blocks based on conditions.

In AI, this is fundamental for:
- **Model Logic:** "If accuracy > 95%, stop training."
- **Data Preprocessing:** "If value is missing, replace with 0, else keep it."
- **Inference:** "If prediction probability > threshold, classify as spam, else not spam."

---

## 2. The `if` Statement
The most basic decision-making structure.

### Syntax
```python
if condition:
    # Code to execute if True
    # Indentation is CRITICAL in Python
```

### Example
```python
accuracy = 0.92

if accuracy > 0.90:
    print("Great model performance!")  # This runs
```

---

## 3. `if` ... `else`
Does one thing if True, another if False.

```python
loss = 0.5

if loss < 0.1:
    print("Model has converged.")
else:
    print("Keep training.")
```

---

## 4. `if` ... `elif` ... `else`
For checking multiple conditions sequentially. Python checks them top-to-bottom and stops at the first True one.

```python
# Learning Rate Scheduler Example
epoch = 50

if epoch < 10:
    lr = 0.01
elif epoch < 30:
    lr = 0.005
elif epoch < 60:
    lr = 0.001
else:
    lr = 0.0001

print(f"Current Learning Rate: {lr}") # 0.001
```

### Common Pitfall
Using multiple `if` statements instead of `elif` means *all* conditions are checked, which might not be what you want.

---

## 5. Logical Operators
Combine multiple conditions.

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `and` | Both must be True | `True and False` | `False` |
| `or` | At least one True | `True or False` | `True` |
| `not` | Inverses boolean | `not True` | `False` |

### Real-World Example: Save Checkpoint
You want to save your model checkpoint ONLY if:
1.  The validation loss has decreased.
2.  AND the accuracy is above 80%.

```python
val_loss_improved = True
accuracy = 0.75

if val_loss_improved and accuracy > 0.80:
    print("Saving model checkpoint...")
else:
    print("Skipping save.")
```

---

## 6. Short-Circuit Evaluation
Python is lazy (efficient).
- In `A and B`, if A is False, B is never checked (result is False).
- In `A or B`, if A is True, B is never checked (result is True).

**Why it matters:** Safety.
```python
model = None

# if model.is_trained: -> CRASH (AttributeError provided model is None)

# Safe check using Short-Circuit
if model is not None and model.is_trained:
    print("Predicting...")
else:
    print("Model not ready.")
```

---

## 7. Deep Dive: Ternary Operator
A one-liner `if-else`. Very Pythonic and useful for assignments.

**Syntax:** `value_if_true if condition else value_if_false`

```python
# Traditional
score = 85
status = ""
if score >= 50:
    status = "Pass"
else:
    status = "Fail"

# Ternary
status = "Pass" if score >= 50 else "Fail"

# AI Context: Activation function (ReLU logic)
x = -5
relu_output = x if x > 0 else 0 
print(relu_output) # 0
```

---

## 8. Practical Exercises

### Exercise 1: Confidence Thresholder
Write a script that takes a `prediction_score` (float between 0 and 1).
- If score >= 0.9, print "High Confidence"
- If score >= 0.5, print "Medium Confidence"
- Else, print "Low Confidence"

### Exercise 2: Batch Validator
Given variables `batch_size` and `is_gpu_available`:
- If `is_gpu_available` is True, set `limit = 64`.
- Else (CPU mode), set `limit = 16`.
- Check if `batch_size` > `limit`. If so, print "Batch size too large for hardware!".

---

## 9. Summary
- **If/Elif/Else** structures control the flow of execution.
- **Indentation** defines the blocks of code.
- **Logical Operators** (`and`, `or`, `not`) combine conditions.
- **Ternary Operators** offer concise conditional assignments.

**Next Up:** **Loops**—automating repetitive tasks over massive datasets.
