# Day 121: Introduction to MLOps and FastAPI

## 1. What is MLOps?
Machine Learning Operations (MLOps) is DevOps for AI.
It solves the problem: "It works on my laptop, but how do I get it to 100,000 users?"
- **Model Development** is only 10% of the work.
- **Serving, Monitoring, Retraining** is the other 90%.

---

## 2. Serving with FastAPI
To make your model accessible, you wrap it in an **API (Application Programming Interface)**.
**FastAPI** is the standard for Python ML serving (fast, auto-documentation).

### Implementation
```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(data: InputData):
    # Imagine 'model' is loaded globally
    # prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": 0.95}

# Run with: uvicorn main:app --reload
```
You can now send a POST request to `http://localhost:8000/predict` and get an answer.

---

## 3. Summary
- **API**: The bridge between your Python script and the User's App.
- **FastAPI**: Validates data types automatically (Pydantic).

**Next Up:** **Docker**—Packaging it up.
