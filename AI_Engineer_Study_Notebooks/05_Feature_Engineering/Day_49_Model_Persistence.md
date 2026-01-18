# Day 49: Model Persistence

## 1. Introduction
You trained a model for 2 days. You close your laptop.
 RAM is cleared. The model is gone.
You need to **Serialize** (Save) the model to disk so you can load it later or deploy it to a server.

---

## 2. Using Joblib
Faster for NumPy arrays than standard Pickle.

```python
import joblib

# Save
joblib.dump(model, 'my_model.pkl')

# Load
loaded_model = joblib.load('my_model.pkl')
loaded_model.predict(new_data)
```

## 3. What to Save?
If you used a Pipeline, save the **Pipeline**, not just the model.
Otherwise, you lose the Scaler and Encoder!

```python
joblib.dump(full_pipeline, 'pipeline_v1.pkl')
```

---

## 4. ONNX (Open Neural Network Exchange)
For production environments (Java, C++, Mobile), we often export to **ONNX** format instead of Python-specific Pickle.

---

## 5. Summary
- **Joblib**: Standard for Scikit-Learn.
- **Save the Pipeline**: Always include preprocessing steps in the artifact.

**Next Up:** **The ML Project**—Putting 8 weeks of knowledge into practice.
