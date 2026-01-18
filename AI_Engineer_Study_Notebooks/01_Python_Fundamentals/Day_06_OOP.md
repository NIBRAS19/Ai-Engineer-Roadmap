# Day 6: Object-Oriented Programming (OOP)

## 1. Introduction
In AI frameworks like PyTorch and TensorFlow, **everything is an Object**.
- A Model is a Class.
- A Layer is a Class.
- A Dataset is a Class.

OOP allows you to bundle **Data** (attributes) and **Behavior** (methods) together.

---

## 2. Classes and Objects
- **Class:** The blueprint (e.g., "Car Design").
- **Object:** The actual entity built from the blueprint (e.g., "Toyota Corolla").

### Syntax
```python
class NeuralNetwork:
    # Constructor: Initializes the object
    def __init__(self, name, layers):
        self.name = name       # Attribute
        self.layers = layers   # Attribute

    # Method: Action the object can do
    def describe(self):
        print(f"Model {self.name} has {self.layers} layers.")

# Creating Objects (Instantiation)
model1 = NeuralNetwork("ResNet", 50)
model2 = NeuralNetwork("VGG", 16)

model1.describe() # Model ResNet has 50 layers.
```
*Note: `self` refers to the specific object instance (e.g., `model1`).*

---

## 3. Inheritance
The superpower of OOP. You can create a new class based on an existing one, inheriting its features.
Extremely common in PyTorch (`class MyModel(nn.Module):`).

```python
# Parent Class
class Model:
    def __init__(self, name):
        self.name = name
    
    def train(self):
        print(f"Training {self.name}...")

# Child Class
class Classifier(Model):
    def predict(self, data):
        print(f"{self.name} classifies: {data}")

# Usage
clf = Classifier("CatDogModel")
clf.train()          # Inherited from Model
clf.predict("Image") # Specific to Classifier
```

---

## 4. Encapsulation (Briefly)
Hiding the internal details. In Python, we use underscores to suggest "private" variables.
```python
class Optimizer:
    def __init__(self, lr):
        self._learning_rate = lr  # "_" means internal use only

    def get_lr(self):
        return self._learning_rate
```

---

## 5. Real-World AI Example: A Custom Dataset Class
This mimics how PyTorch Datasets work.

```python
class SimpleDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """Returns size of dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Returns sample at index idx"""
        return self.data[idx], self.labels[idx]

# Usage
X = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.1]]
y = [0, 1, 0]

dataset = SimpleDataset(X, y)

print(f"Dataset Size: {len(dataset)}")  # Calls __len__
sample, label = dataset[1]              # Calls __getitem__
print(f"Sample: {sample}, Label: {label}")
```

---

## 6. Practical Exercises

### Exercise 1: The `Model` Class
Create a class `Model` with:
- Attributes: `name` (string), `is_trained` (bool, default False).
- Method `train()`: Sets `is_trained` to True and prints "Training complete".
- Method `infer(input)`: If trained, prints "Predicting {input}"; else prints "Model not trained!".

### Exercise 2: Inheritance
Create a class `Transformer` that inherits from `Model`. Add an attribute `heads` (int) and override `train()` to print "Training Transformer with {heads} heads...".

---

## 7. Summary
- **Classes** are blueprints; **Objects** are instances.
- **Methods** are functions inside a class; `__init__` is the constructor.
- **Inheritance** allows reusability (Child classes get Parent features).
- **Magic Methods** (like `__len__`, `__getitem__`) allow objects to behave like Python built-ins.

**Next Up:** **File & Exception Handling**—reading data from disks and ensuring your code doesn't crash.
