# Day 123: Experiment Tracking & Data Versioning

## 1. The Problem: "Which dataset did I use?"
You trained a model that got 98% accuracy.
Two months later, you try to reproduce it, but your CSV file has changed. You're lost.

**Solution**:
1.  **MLflow**: Tracks Code, Metrics, and Hyperparameters.
2.  **DVC (Data Version Control)**: Tracks Datasets and Large Model Files.

---

## 2. MLflow (The Lab Notebook)

```python
import mlflow

mlflow.set_experiment("Cat vs Dog")

with mlflow.start_run():
    # 1. Log Params
    mlflow.log_param("lr", 0.01)
    
    # 2. Log Metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # 3. Log Model (The Artifact)
    mlflow.sklearn.log_model(model, "model")
```
*Run `mlflow ui` to visualize comparison charts.*

---

## 3. DVC (Git for Data)
Git is bad at large files (10GB CSVs). DVC solves this.
It stores the *actual* data in S3/GCS/Drive, and stores a small pointer file (`data.csv.dvc`) in Git.

### Workflow
1.  **Initialize**: `dvc init`
2.  **Add Data**: `dvc add data.csv` (Creates `data.csv.dvc`)
3.  **Track in Git**: `git add data.csv.dvc .gitignore`
4.  **Push**: `dvc push` (Sends data to S3)

Now, to go back to last month's dataset:
`git checkout <old-commit>`
`dvc checkout` (Downloads the old data matching that commit)

---

## 4. Feature Stores (Feast)
In big companies, Team A calculates "User Click Rate" and saves it. Team B calculates it again.
**Feature Store** avoids this duplication.
- **Offline Store**: Cheap storage (Parquet/S3) for Training.
- **Online Store**: Fast storage (Redis) for Inference.
- **Feast**: Open source feature store. Ensures training data matches production data (**Training-Serving Skew** prevention).

---

## 5. Summary
- **MLflow**: Tracks the Experiment (Code + Params + Metrics).
- **DVC**: Tracks the Data (Storage + Versioning).
- **Feature Store**: Reusable features for Training and Serving.

**Next Up:** **CI/CD**—Automating deployment.

