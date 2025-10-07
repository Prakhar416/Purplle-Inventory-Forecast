# Purplle Inventory Forecasting — End-to-End Machine Learning & Deployment Project




## Table of Contents

1. [Overview](#1-overview)  
2. [Project Structure](#2-project-structure)  
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)  
4. [Model Development and Validation](#4-model-development-and-validation)  
5. [Model Training and Prediction](#5-model-training-and-prediction)  
6. [Local Deployment — FastAPI](#6-local-deployment--fastapi)  
7. [Containerization — Docker](#7-containerization--docker)  
8. [Deployment — GCP Cloud Run](#8-deployment--gcp-cloud-run)  
9. [Deployment — GCP Cloud Function](#9-deployment--gcp-cloud-function)  
10. [Commands Reference](#10-commands-reference)  
11. [Key Code Snippets & Explanations](#11-key-code-snippets--explanations)  
12. [Results and Learnings](#12-results-and-learnings)  
13. [Summary & Reflection](#13-summary--reflection)  
14. [Appendix — Key Files](#14-appendix--key-files)

---

## 1. Overview

This repository implements an **end-to-end inventory forecasting system** for Purplle. The project follows a full data science and MLOps pipeline:

- Exploratory Data Analysis (EDA) on multi-year sales data.  
- Model experimentation and selection (comparing regressors).  
- Final model training, forecasting next 2 years, and iterative retraining using forecasted data.  
- Exposing predictions through a FastAPI REST service locally.
- Containerizing the application with Docker and deploying to **GCP Cloud Run** (FastAPI).  
- Providing a lightweight serverless endpoint via **GCP Cloud Function**.



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



## 3. Exploratory Data Analysis (EDA)

**Purpose:**  
Understand seasonality, trends, missingness, outliers, and feature relationships so feature engineering and model decisions are informed.

[EDA File](purplle.ipynb)

**Typical steps performed:**
- Cleaning and preprocessing the raw sales data.
- Parse and normalize dates; create time features (month, quarter, year, day-of-week,...).
- Identifying **monthly and yearly sales trends**.  
- Fill or impute missing values intelligently; treat or cap outliers. 
- Visualizing **regional** and **category-wise** sales performance.
- Visualize correlations and feature importances to guide feature selection.



## 4. Model Development and Validation

[Model Development](purplle_predictice.ipynb)
[Model Selection and Validation](Model-Selection-and-Validation.ipynb)

**Models evaluated:**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (final selection)

**Validation approach:**
- Temporal train/validation split (respecting time order).
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



## 5. Model Training and Prediction

[Model Training and Prediction](XGBoost-Model-Training-and-Prediction.ipynb)

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
future_features = create_daily_features(future_dates) # user-defined helper
preds = model.predict(future_features)
pd.DataFrame({'Date': future_dates, 'Forecast': preds}).to_csv('2year_forecast.csv', index=False)
```

*Notes:* `create_daily_features` should mirror feature engineering used during training (lags, rolling means, categorical encodings, etc.).



## 6. Local Deployment — FastAPI

[Main.py](app/main.py)

**Goal:** expose model predictions as a REST endpoint for integration and testing.

**Key app/main.py snippet:**
```python
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
import pandas as pd
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="XGBoost Sales Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path("static/index.html")
    return index_path.read_text()

class PredictRequest(BaseModel):
    ean_code: str
    order_date: str

class PredictResponse(BaseModel):
    predicted_quantity: int

model = xgb.XGBRegressor()

model.load_model("model.json")
agg_df = pd.read_csv("data.csv", parse_dates=['order_date'])
agg_df = agg_df.sort_values(['ean_code', 'order_date']).reset_index(drop=True)
le = joblib.load("label_encoder.joblib")

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict(
    request: PredictRequest,
    # api_key: str = Depends(get_api_key)
):
    try:
        features = prepare_features(agg_df, request.ean_code, request.order_date, le)
        input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)[0]
        predicted_quantity = max(0, int(round(prediction)))
        return PredictResponse(predicted_quantity=predicted_quantity)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal prediction error")

```

**Run locally:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Sample cURL:**
```bash
curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json"
-d '{"order_date":"2026-01-01","ean_code":"8904362500005"}'
```

---

## 7. Containerization — Docker

**Guarantee consistent environment across deployments.**

[Dockerfile](app/Dockerfile)

**Sample Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


**Build & run:**
```bash
docker build -t purplle-inventory:latest .
docker run -p 8000:8000 purplle-inventory:latest
```

> *Ensure model files are available in the image or as a mounted volume.*



## 8. Deployment — GCP Cloud Run

**Steps:**
- Build and push Docker image.
- Deploy to Cloud Run for autoscaling and https.

**Commands (replace `{PROJECT_ID}`):**
```bash
gcloud builds submit --tag gcr.io/{PROJECT_ID}/purplle-inventory
gcloud run deploy purplle-inventory
--image gcr.io/{PROJECT_ID}/purplle-inventory
--platform managed
--region asia-south1
--allow-unauthenticated
```

> *Tune memory/CPU settings and consider model-hosting strategy for large artifacts.*



## 9. Deployment — GCP Cloud Function

**For lightweight, event-driven serving.**

**Example (`Cloudfunction/main.py`):**
```python
import functions_framework
import pandas as pd
import joblib
import xgboost as xgb
import json
from google.cloud import storage
import tempfile
import os

model = None
le = None
agg_df = None
storage_client = storage.Client()
BUCKET_NAME = os.environ.get('MODEL_BUCKET')

@functions_framework.http
def predict(request):
    try:
        load_resources()
        request_json = request.get_json()
        if not request_json:
            print("Missing JSON body")
            body = json.dumps({"error": "Missing JSON body"})
            return (body, 400, {"Content-Type": "application/json"})

        ean_code = request_json.get('ean_code')
        order_date = request_json.get('order_date')

        if not ean_code or not order_date:
            print(f"Missing parameters - ean_code: {ean_code}, order_date: {order_date}")
            body = json.dumps({"error": "Missing 'ean_code' or 'order_date'"})
            return (body, 400, {"Content-Type": "application/json"})

        print(f"Received request: ean_code={ean_code}, order_date={order_date}")

        features = prepare_features(agg_df, ean_code, order_date, le)
        print(f"Prepared features: {features}")

        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        predicted_quantity = max(0, int(round(prediction)))

        print(f"Model prediction: {prediction}, Rounded prediction: {predicted_quantity}")

        body = json.dumps({"predicted_quantity": predicted_quantity})
        return (body, 200, {"Content-Type": "application/json"})

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        body = json.dumps({"error": str(ve)})
        return (body, 400, {"Content-Type": "application/json"})
    except Exception as e:
        print(f"Exception: {str(e)}")
        body = json.dumps({"error": "Internal prediction error", "details": str(e)})
        return (body, 500, {"Content-Type": "application/json"})
```

**Deploy:**
```bash
gcloud functions deploy purplle-forecast `
--runtime=python311 `
--trigger-http `
--entry-point=predict `
--set-env-vars MODEL_BUCKET=purplle-forecast `
--allow-unauthenticated `
--source . `
--memory=1GB
```

> *Cloud Functions are best for low throughput and quick demos due to resource limits and cold starts.*



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



## 12. Results and Learnings

- **XGBoost** provided the most stable and high-performing forecasts.
- Dockerizing workflows made deployment predictable.
- Retraining on predicted data improves trend robustness (use cautiously to avoid feedback bias).
- Cloud Run is ideal for scale and reliability; Cloud Function is great for demos or low volume.
- Addressed feature drift and reproducible preprocessing.



## 13. Summary & Reflection

This project demonstrates the transformation of raw data into a reliable forecasting service, with robust EDA, validation, iterative improvements, and practical deployment.

Productionization required repeated testing, careful engineering, and prioritization of reproducibility and scale.



## 14. Appendix — Key Files

- `purplle.ipynb`: Exploratory Data Analysis
- `5years_data_visualization.ipynb`: Visualizations
- `Model Selection and Validation.ipynb`: model comparisons & validation
- `XGBoost Model Training and Prediction.ipynb`: final model training/forecasting
- `app/main.py`: FastAPI REST API logic
- `app/utils.py`: feature engineering and preprocessing utilities
- `app/Dockerfile`: Docker container instructions
- `Cloudfunction/main.py`: Flask Cloud Function (GCP)
- `app/models/`: trained model artifacts
