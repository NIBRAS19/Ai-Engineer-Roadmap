# Day 124: CI/CD for ML (GitHub Actions)

## 1. What is CI/CD?
- **Continuous Integration (CI)**: Automatically testing your code every time you push to GitHub.
- **Continuous Deployment (CD)**: Automatically deploying your model to the server if tests pass.

---

## 2. GitHub Actions
You define a `.yml` file in `.github/workflows/`.

```yaml
name: ML Pipeline

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2
      
      - name: Install Python
        uses: actions/setup-python@v2
      
      - name: Install Dependencies
        run: pip install -r requirements.txt
      
      - name: Run Tests
        run: pytest tests.py
        
      - name: Build Docker Image
        run: docker build -t my-app .
```

---

## 3. Automated Training
You can even trigger a "Retrain" job automatically when new data arrives in your S3 bucket.

---

## 4. Summary
- **Automation**: Removes human error.
- **Safety**: Prevents breaking production with bad code.

**Next Up:** **Monitoring**—Watching the model in the wild.
