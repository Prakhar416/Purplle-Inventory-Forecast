from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
import xgboost as xgb
from fastapi import Query
from app.utils import prepare_features, FEATURE_COLUMNS
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path="app/.env")

app = FastAPI(title="XGBoost Sales Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_key")

print(f"Loaded API_KEY in server: {repr(API_KEY)}")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)) -> APIKey:
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")

model = xgb.XGBRegressor()
model.load_model("app/models/Purplle XGB Trained Model_5yrs.json")

agg_df = pd.read_csv("app/data/5year_data.csv", parse_dates=['order_date'])
agg_df = agg_df.sort_values(['ean_code', 'order_date']).reset_index(drop=True)

le = joblib.load("app/models/label_encoder.joblib")

class PredictResponse(BaseModel):
    predicted_quantity: int 

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict(
    ean_code: str = Query(..., description="Product EAN code"),
    order_date: str = Query(..., description="Date for prediction YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    try:
        features = prepare_features(agg_df, ean_code, order_date, le)
        input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        prediction = model.predict(input_df)[0]
        predicted_quantity = max(0, int(round(prediction)))

        return PredictResponse(predicted_quantity=predicted_quantity)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal prediction error")