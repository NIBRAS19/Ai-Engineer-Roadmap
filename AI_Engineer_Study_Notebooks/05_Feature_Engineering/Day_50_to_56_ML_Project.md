# Days 50-56: End-to-End Machine Learning Project

## 1. The Challenge: Housing Price Prediction

Your goal is to build a **production-ready** price prediction model following professional data science practices.

**Dataset Options**:
- California Housing (Built-in Sklearn) — simpler, good for learning
- Ames Housing (Kaggle) — complex, realistic feature engineering

### 🎯 What This Project Teaches You
> This is your **capstone** for classical ML. By the end, you'll have a reproducible workflow you can apply to ANY tabular ML problem: credit scoring, churn prediction, demand forecasting, etc.

---

## 2. Day-by-Day Breakdown

| Day | Phase | Focus |
|:---:|:------|:------|
| 50 | Data Loading & EDA | Understand your data deeply |
| 51 | Feature Engineering | Create powerful features |
| 52 | Preprocessing Pipeline | Automate transformations |
| 53 | Model Selection | Compare algorithms systematically |
| 54 | Hyperparameter Tuning | Squeeze out performance |
| 55 | Error Analysis | Understand failures |
| 56 | Packaging & Deployment | Make it usable |

---

## 3. Day 50: Data Loading & EDA Template

### 3.1 Project Structure
```
housing_project/
├── data/
│   ├── raw/              # Original data (never modify)
│   └── processed/        # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   └── train.py
├── models/
│   └── housing_model.pkl
└── README.md
```

### 3.2 EDA Checklist (Copy This Template!)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# STEP 1: Load and First Look
# ========================================
df = pd.read_csv('data/raw/housing.csv')

print("Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
display(df.head())

# ========================================
# STEP 2: Missing Values
# ========================================
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print(missing_df[missing_df['Missing'] > 0].sort_values('Percent', ascending=False))

# Visualize
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# ========================================
# STEP 3: Target Variable Distribution
# ========================================
target = 'median_house_value'  # Change to your target

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df[target].hist(bins=50)
plt.title(f'{target} Distribution')

plt.subplot(1, 2, 2)
np.log1p(df[target]).hist(bins=50)
plt.title(f'Log({target}) - More Normal?')
plt.tight_layout()
plt.show()

print(f"Skewness: {df[target].skew():.2f}")
# If skewness > 1, consider log-transforming the target

# ========================================
# STEP 4: Numerical Features Statistics
# ========================================
print(df.describe().T)

# ========================================
# STEP 5: Correlation Analysis
# ========================================
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Top correlations with target
print(f"\nTop Correlations with {target}:")
print(corr_matrix[target].sort_values(ascending=False)[1:6])

# ========================================
# STEP 6: Categorical Features
# ========================================
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    
# ========================================
# STEP 7: Outlier Detection
# ========================================
num_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(nrows=len(num_cols)//3 + 1, ncols=3, figsize=(15, 4*len(num_cols)//3))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
```

### 3.3 Key Questions to Answer in EDA

| Question | Why It Matters |
|:---------|:---------------|
| Is target skewed? | May need log-transform for regression |
| Which features correlate with target? | Focus feature engineering here |
| Are there outliers? | May need clipping or removal |
| Missing value patterns? | Random vs systematic affects strategy |
| Feature distributions? | Normal → StandardScaler, Skewed → RobustScaler |

---

## 4. Day 51-52: Feature Engineering & Pipeline

### 4.1 Feature Engineering Ideas

```python
# Create new features based on domain knowledge
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Binning continuous variables
df['income_category'] = pd.cut(df['median_income'], 
                                bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                labels=[1, 2, 3, 4, 5])

# Geographic features (if lat/long available)
# df['distance_to_city_center'] = haversine(df['lat'], df['long'], city_lat, city_long)
```

### 4.2 Complete Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
num_features = ['median_income', 'housing_median_age', 'total_rooms', 
                'total_bedrooms', 'population', 'households']
cat_features = ['ocean_proximity']

# Numeric pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full pipeline with model
from sklearn.ensemble import RandomForestRegressor

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit
full_pipeline.fit(X_train, y_train)
```

---

## 5. Day 53: Model Selection Strategy

### 🎯 The CRISP Approach: Don't Tune Bad Models

> **Rule**: First find the RIGHT algorithm, THEN tune it. Spending 10 hours tuning Linear Regression when Random Forest is 10x better is wasted effort.

### 5.1 Spot-Check Multiple Algorithms

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Create test harness
def evaluate_model(model, X, y, cv=5):
    """Evaluate model using cross-validation."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()

# Prepare data
X_processed = preprocessor.fit_transform(X_train)

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
}

# Evaluate all
results = []
for name, model in models.items():
    mean_rmse, std_rmse = evaluate_model(model, X_processed, y_train)
    results.append({'Model': name, 'RMSE': mean_rmse, 'Std': std_rmse})
    print(f"{name:25} RMSE: {mean_rmse:.2f} (+/- {std_rmse:.2f})")

# Visualize
results_df = pd.DataFrame(results).sort_values('RMSE')
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['RMSE'])
plt.xlabel('RMSE (lower is better)')
plt.title('Model Comparison')
plt.tight_layout()
plt.show()
```

### 5.2 Model Selection Decision Tree

```
Is your data large (>100k rows)?
├── Yes → Start with Linear models, Gradient Boosting
└── No → Try everything

Is interpretability required?
├── Yes → Linear Regression, Decision Tree
└── No → Ensemble methods (RF, XGBoost)

Is training time critical?
├── Yes → Linear, Decision Tree, LightGBM
└── No → Try complex ensembles
```

---

## 6. Day 54: Hyperparameter Tuning

### 6.1 Random Search (Faster than Grid Search)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define hyperparameter distributions
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random Search
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions,
    n_iter=50,  # Number of random combinations to try
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_processed, y_train)

print(f"\nBest RMSE: {-random_search.best_score_:.2f}")
print(f"Best Parameters: {random_search.best_params_}")
```

### 6.2 Learning Curves (Diagnose Bias vs Variance)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    random_search.best_estimator_,
    X_processed, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_root_mean_squared_error'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training Error')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Interpretation:
# - Big gap = Overfitting (try more regularization, simpler model)
# - Both high = Underfitting (try more features, complex model)
# - Converging = Good fit
```

---

## 7. Day 55: Error Analysis (The Pro Move)

### 🎯 Why Error Analysis Matters
> Most beginners stop at "my RMSE is X." Professionals ask: "**WHERE** does my model fail, and **WHY**?"

### 7.1 Residual Analysis

```python
# Get predictions
y_pred = best_model.predict(X_test_processed)
residuals = y_test - y_pred

# Plot 1: Residuals vs Predicted
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted')
# Pattern = Model is missing something

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=50)
plt.xlabel('Residual')
plt.title('Residual Distribution')
# Should be normal, centered at 0

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()
```

### 7.2 Find Worst Predictions

```python
# Add predictions and errors to test set
test_df = X_test.copy()
test_df['actual'] = y_test
test_df['predicted'] = y_pred
test_df['error'] = abs(residuals)
test_df['pct_error'] = abs(residuals) / y_test * 100

# Worst predictions
print("Top 10 Worst Predictions:")
display(test_df.nlargest(10, 'error')[['actual', 'predicted', 'error', 'pct_error']])

# Analyze patterns in errors
print("\nMean error by category:")
print(test_df.groupby('ocean_proximity')['error'].mean().sort_values(ascending=False))
# If one category has much higher error → need more data or features for it
```

### 7.3 Error Analysis Questions

| Pattern Observed | Likely Cause | Fix |
|:-----------------|:-------------|:----|
| Underpredicts high values | Model can't capture extremes | Log-transform target, add polynomial features |
| High error in specific category | Insufficient training data | Get more data, or train separate model |
| Residuals show pattern | Missing interaction/feature | Create new features |
| Errors correlate with a feature | Feature leakage or missing transformation | Check data pipeline |

---

## 8. Day 56: Packaging & Deployment

### 8.1 Final Model Training

```python
# Train on ALL data (train + validation)
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(**best_params, random_state=42))
])

final_pipeline.fit(X_full, y_full)

# Final evaluation on held-out test set
final_rmse = np.sqrt(mean_squared_error(y_test, final_pipeline.predict(X_test)))
print(f"Final Test RMSE: {final_rmse:.2f}")
```

### 8.2 Save Model

```python
import joblib

# Save
joblib.dump(final_pipeline, 'models/housing_model.pkl')

# Load (for inference)
loaded_model = joblib.load('models/housing_model.pkl')

# Predict on new data
new_house = pd.DataFrame({
    'median_income': [4.5],
    'housing_median_age': [25],
    # ... all features
})
prediction = loaded_model.predict(new_house)
print(f"Predicted Price: ${prediction[0]:,.0f}")
```

### 8.3 Simple Streamlit App

```python
# app.py
import streamlit as st
import joblib
import pandas as pd

st.title("🏠 Housing Price Predictor")

# Load model
model = joblib.load('models/housing_model.pkl')

# Input widgets
income = st.slider("Median Income", 0.5, 15.0, 4.0)
age = st.slider("Housing Age", 1, 52, 25)
rooms = st.number_input("Total Rooms", 100, 10000, 2000)

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'median_income': [income],
        'housing_median_age': [age],
        'total_rooms': [rooms],
        # ... other features with defaults
    })
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${prediction:,.0f}")
```

Run with: `streamlit run app.py`

---

## 9. Project Checklist

### Phase 1: Exploration ✓
- [ ] Load data, check shape and types
- [ ] Analyze missing values
- [ ] Plot target distribution (consider log-transform)
- [ ] Create correlation heatmap
- [ ] Identify outliers

### Phase 2: Feature Engineering ✓
- [ ] Handle missing values
- [ ] Create domain-specific features
- [ ] Encode categorical variables
- [ ] Scale numerical variables
- [ ] Build preprocessing pipeline

### Phase 3: Model Selection ✓
- [ ] Spot-check 5+ algorithms
- [ ] Select top 2 performers
- [ ] Plot learning curves

### Phase 4: Tuning ✓
- [ ] RandomizedSearchCV on best model
- [ ] Analyze learning curves for bias/variance

### Phase 5: Error Analysis ✓
- [ ] Residual plots
- [ ] Find worst predictions
- [ ] Analyze error patterns by segment

### Phase 6: Deployment ✓
- [ ] Train final model on all data
- [ ] Save pipeline to .pkl
- [ ] Create simple app or API

---

## 10. Summary

This project taught you the **professional ML workflow**:

1. **EDA First**: Understand data before modeling
2. **Pipeline Everything**: Reproducibility is key
3. **Compare Models**: Don't prematurely commit to one algorithm
4. **Tune Wisely**: Random Search >> Grid Search
5. **Analyze Errors**: Know WHERE your model fails
6. **Package for Use**: A model in a notebook is useless

**CONGRATULATIONS!** You have completed **Part 1: Classical AI**.
You are now a competent Data Scientist.

**Next Week:** We enter the modern era. **Deep Learning and PyTorch**.
From here on, data becomes unstructured (Images, Text), and models become massive Neural Networks.

