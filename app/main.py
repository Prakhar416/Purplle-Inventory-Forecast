from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
# import os
# from dotenv import load_dotenv
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

# load_dotenv(dotenv_path="./.env")
# API_KEY = os.getenv("API_key")

app = FastAPI(title="XGBoost Sales Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# async def get_api_key(api_key_header: str = Depends(api_key_header)):
#     if api_key_header == API_KEY:
#         return api_key_header
#     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

# Assuming your index.html is in a "static" folder alongside main.py
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
# model.load_model("app/models/Purplle XGB Trained Model_5yrs.json")
# agg_df = pd.read_csv("app/data/5year_data.csv", parse_dates=['order_date'])
# agg_df = agg_df.sort_values(['ean_code', 'order_date']).reset_index(drop=True)
# le = joblib.load("app/models/label_encoder.joblib")

model.load_model("model.json")
agg_df = pd.read_csv("data.csv", parse_dates=['order_date'])
agg_df = agg_df.sort_values(['ean_code', 'order_date']).reset_index(drop=True)
le = joblib.load("label_encoder.joblib")

FEATURE_COLUMNS = [
    'lag_1', 'lag_2', 'lag_7', 'roll_mean_3', 'roll_mean_7', 'roll_mean_14',
    'dayofweek', 'dayofmonth', 'weekofyear', 'month', 'quarter',
    'is_month_start', 'is_month_end', 'is_weekend', 'ean_code_encoded'
]

def prepare_features(agg_df, ean_code, order_date, le):
    order_date = pd.to_datetime(order_date)
    df = agg_df[agg_df['ean_code'] == ean_code].copy()
    if df.empty:
        raise ValueError("EAN code not found in data")
    df = df.sort_values('order_date').set_index('order_date')

    if order_date <= df.index.max():
        raise ValueError("Order date must be after last historical date.")

    lag_1 = df['quantity'].get(order_date - pd.Timedelta(days=1), 0)
    lag_2 = df['quantity'].get(order_date - pd.Timedelta(days=2), 0)
    lag_7 = df['quantity'].get(order_date - pd.Timedelta(days=7), 0)

    roll_mean_3 = df['quantity'].loc[(order_date - pd.Timedelta(days=3)):(order_date - pd.Timedelta(days=1))].mean()
    roll_mean_7 = df['quantity'].loc[(order_date - pd.Timedelta(days=7)):(order_date - pd.Timedelta(days=1))].mean()
    roll_mean_14 = df['quantity'].loc[(order_date - pd.Timedelta(days=14)):(order_date - pd.Timedelta(days=1))].mean()

    dayofweek = order_date.dayofweek
    dayofmonth = order_date.day
    weekofyear = order_date.isocalendar().week
    month = order_date.month
    quarter = order_date.quarter
    is_month_start = int(order_date.is_month_start)
    is_month_end = int(order_date.is_month_end)
    is_weekend = int(dayofweek >= 5)

    ean_code_encoded = le.transform([ean_code])[0]

    features = {
        'lag_1': lag_1 if pd.notna(lag_1) else 0,
        'lag_2': lag_2 if pd.notna(lag_2) else 0,
        'lag_7': lag_7 if pd.notna(lag_7) else 0,
        'roll_mean_3': roll_mean_3 if pd.notna(roll_mean_3) else 0,
        'roll_mean_7': roll_mean_7 if pd.notna(roll_mean_7) else 0,
        'roll_mean_14': roll_mean_14 if pd.notna(roll_mean_14) else 0,
        'dayofweek': dayofweek,
        'dayofmonth': dayofmonth,
        'weekofyear': weekofyear,
        'month': month,
        'quarter': quarter,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'is_weekend': is_weekend,
        'ean_code_encoded': ean_code_encoded
    }
    return features

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
