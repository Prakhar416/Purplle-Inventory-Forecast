# Purplle Inventory Forecasting — End-to-End Machine Learning & Deployment Project

**Author:** Prakhar Sethiya  
**Date:** October 6, 2025

---

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Model Development and Validation](#model-development-and-validation)  
5. [Model Training and Prediction](#model-training-and-prediction)  
6. [Local Deployment — FastAPI](#local-deployment--fastapi)  
7. [Containerization — Docker](#containerization--docker)  
8. [Deployment — GCP Cloud Run](#deployment--gcp-cloud-run)  
9. [Deployment — GCP Cloud Function (Flask)](#deployment--gcp-cloud-function-flask)  
10. [Commands Reference](#commands-reference)  
11. [Key Code Snippets & Explanations](#key-code-snippets--explanations)  
12. [Results and Learnings](#results-and-learnings)  
13. [Summary & Reflection](#summary--reflection)  
14. [Appendix — Key Files](#appendix--key-files)

---

## 1. Overview

This repository implements an **end-to-end inventory forecasting system** for Purplle. The project follows a full data science and MLOps pipeline:

- Exploratory Data Analysis (EDA) on multi-year sales data.  
- Model experimentation and selection (comparing regressors).  
- Final model training, forecasting next 2 years, and iterative retraining using forecasted data.  
- Exposing predictions through a FastAPI REST service locally.
- Containerizing the application with Docker and deploying to **GCP Cloud Run**.  
- Providing a lightweight serverless endpoint via **GCP Cloud Function** (Flask).

The work represents careful analysis, iterative testing, and production-minded deployment — reflecting sustained effort, attention to reproducibility, and practical MLOps skills.

---

## 2. Project Structure
```
Purplle-Inventory-Forecast-main/
│
├── 5years_data_visualization.ipynb
├── Model Selection and Validation.ipynb
├── XGBoost Model Training and Prediction.ipynb
├── app/
│ ├── main.py # FastAPI app
│ ├── utils.py # preprocessing + helpers
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── models/ # saved model artifacts (.json, .joblib)
│ ├── static/ # optional front-end
│ └── data/ # CSVs used in notebooks
└── Cloudfunction/
├── main.py # Flask-based cloud function
└── requirements.txt
```

---

## 3. Exploratory Data Analysis (EDA)

**Purpose:**  
Understand seasonality, trends, missingness, outliers, and feature relationships so feature engineering and model decisions are informed.

**Typical steps performed:**
- Parse and normalize dates; create time features (month, quarter, year, day-of-week).
- Aggregate sales by SKU/category/region and visualize seasonal patterns.
- Fill or impute missing values intelligently; treat or cap outliers.
- Visualize correlations and feature importances to guide feature selection.

**Representative snippet:**
parse dates and add time features
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

quick seasonal plot
sns.lineplot(x='Date', y='Sales', data=df.groupby('Date')['Sales'].sum().reset_index())
plt.title('Daily Sales (5 years)')
plt.show()
```

---

## 4. Model Development and Validation

**Models evaluated:**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (final selection)

**Validation approach:**
- Temporal train/validation split (respecting time order) and K-fold cross-validation where applicable.
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R².
- Hyperparameter tuning via grid search or lightweight CV for RandomForest / XGBoost.

**Evaluation snippet:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = mean_squared_error(y_val, preds, squared=False)
r2 = r2_score(y_val, preds)
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
```

*Rationale for XGBoost*: handled non-linearities and interactions well, robust to heterogeneous features, trained quickly to high performance and generalization.

---

## 5. Model Training and Prediction

**Workflow:**
- Train final model on the full historical dataset (after validation).
- Generate forecasts for the next 2 years using appropriate feature engineering for future dates.
- Append the generated 2-year forecast as "pseudo-observations" and retrain the model to help it internalize projected trends (iterative refinement).

**Training snippet:**
```python
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=10)
model.save_model("models/purplle_xgb_5yrs.json")
```

**Forecasting snippet:**
```python
construct future features for next 24 months
future_dates = pd.date_range(start=df['Date'].max()+pd.Timedelta(days=1), periods=24, freq='MS')
future_features = make_time_features(future_dates) # user-defined helper
preds = model.predict(future_features)
pd.DataFrame({'Date': future_dates, 'Forecast': preds}).to_csv('2year_forecast.csv', index=False)
```

*Notes:* `make_time_features` should mirror feature engineering used during training (lags, rolling means, categorical encodings, etc.).

---

## 6. Local Deployment — FastAPI

**Goal:** expose model predictions as a REST endpoint for integration and testing.

**Key app/main.py snippet:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class PredictRequest(BaseModel):
date: str
sku: str
store_id: int
# add other features required

app = FastAPI()
model = joblib.load("models/purplle_xgb_5yrs.joblib")

@app.post("/predict/")
def predict(req: PredictRequest):
row = pd.DataFrame([req.dict()])
# preprocess row (same pipeline used at training time)
X = preprocess_input(row)
pred = model.predict(X)
return {"forecast": float(pred)}
```

**Run locally:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
text

**Sample cURL:**
```bash
curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json"
-d '{"date":"2026-01-01","sku":"SKU123","store_id":10}'
```

---

## 7. Containerization — Docker

**Guarantee consistent environment across deployments.**

**Sample Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```bash


**Build & run:**
```bash
docker build -t purplle-inventory:latest .
docker run -p 8000:8000 purplle-inventory:latest
```

> *Ensure model files are available in the image or as a mounted volume.*

---

## 8. Deployment — GCP Cloud Run

**Steps:**
- Build and push Docker image.
- Deploy to Cloud Run for autoscaling and https.

**Commands (replace `{PROJECT_ID}`):**
```bash
gcloud builds submit --tag gcr.io/{PROJECT_ID}/purplle-inventory
gcloud run deploy purplle-inventory
--image gcr.io/{PROJECT_ID}/purplle-inventory
--platform managed --region asia-south1 --allow-unauthenticated
```

> *Tune memory/CPU settings and consider model-hosting strategy for large artifacts.*

---

## 9. Deployment — GCP Cloud Function (Flask)

**For lightweight, event-driven serving.**

**Example (`Cloudfunction/main.py`):**
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(name)
model = joblib.load('model.json')

@app.route('/predict', methods=['POST'])
def predict():
data = request.get_json()
df = pd.DataFrame([data])
X = preprocess_input(df)
pred = model.predict(X)
return jsonify({'forecast': float(pred)})
```

**Deploy:**
```bash
gcloud functions deploy purplle-function
--runtime python310
--trigger-http
--allow-unauthenticated
--entry-point app
```

> *Cloud Functions are best for low throughput and quick demos due to resource limits and cold starts.*

---

## 10. Commands Reference

**FastAPI (local):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Docker:**
```bash
docker build -t purplle-inventory:latest .
docker run -p 8000:8000 purplle-inventory:latest
```

**GCP Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/{PROJECT_ID}/purplle-inventory
gcloud run deploy purplle-inventory --image gcr.io/{PROJECT_ID}/purplle-inventory --platform managed --region asia-south1
```

**GCP Cloud Function:**
```bash
gcloud functions deploy purplle-function --runtime python310 --trigger-http --allow-unauthenticated
```
> *Replace `{PROJECT_ID}` with your actual GCP project id.*

---

## 11. Key Code Snippets & Explanations

**A. EDA — Date Parsing and Trend Plotting**
```python
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
monthly_sales = df['Sales'].resample('M').sum()
monthly_sales.plot()
```
*Converts dates, resamples to monthly level, and visualizes sales trends.*

---

**B. Model Evaluation**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
*MAE expresses average error in sales units.*

---

**C. Training and Saving XGBoost**
```python
model.fit(X_train, y_train)
model.save_model("models/purplle_xgb_5yrs.json")
```
*Model is trained and persisted to disk for later inference.*

---

**D. FastAPI Endpoint Example**
```python
@app.post("/predict/")
def predict(input_data: dict):
df = pd.DataFrame([input_data])
X = preprocess_input(df)
pred = model.predict(X)
return {"forecast": float(pred)}
```

*Receives JSON, preprocesses, returns prediction.*

---

**E. Minimal Dockerfile**
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 12. Results and Learnings

- **XGBoost** provided the most stable and high-performing forecasts.
- Dockerizing workflows made deployment predictable.
- Retraining on predicted data improves trend robustness (use cautiously to avoid feedback bias).
- Cloud Run is ideal for scale and reliability; Cloud Function is great for demos or low volume.
- Addressed feature drift and reproducible preprocessing.

---

## 13. Summary & Reflection

This project demonstrates the transformation of raw data into a reliable forecasting service, with robust EDA, validation, iterative improvements, and practical deployment.

Productionization required repeated testing, careful engineering, and prioritization of reproducibility and scale.

---

## 14. Appendix — Key Files

- `5years_data_visualization.ipynb`: EDA and visualizations
- `Model Selection and Validation.ipynb`: model comparisons & validation
- `XGBoost Model Training and Prediction.ipynb`: final model training/forecasting
- `app/main.py`: FastAPI REST API logic
- `app/utils.py`: feature engineering and preprocessing utilities
- `Dockerfile`: Docker container instructions
- `Cloudfunction/main.py`: Flask Cloud Function (GCP)
- `models/`: trained model artifacts
