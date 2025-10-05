import functions_framework
import pandas as pd
import joblib
import xgboost as xgb
import json
from google.cloud import storage
import tempfile
import os
from flask import jsonify
from datetime import datetime

model = None
le = None
agg_df = None
storage_client = storage.Client()
BUCKET_NAME = os.environ.get('MODEL_BUCKET')

def download_blob_to_tempfile(blob_name):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    return temp_file.name

def load_resources():
    global model, le, agg_df
    if model is None or le is None or agg_df is None:
        model_path = download_blob_to_tempfile("model.json")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        os.unlink(model_path)

        le_path = download_blob_to_tempfile("label_encoder.joblib")
        le = joblib.load(le_path)
        os.unlink(le_path)

        data_path = download_blob_to_tempfile("data.csv")
        agg_df = pd.read_csv(data_path, parse_dates=['order_date'])
        agg_df = agg_df.sort_values(['ean_code', 'order_date']).reset_index(drop=True)
        os.unlink(data_path)

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

@functions_framework.http
def predict(request):
    try:
        load_resources()
        request_json = request.get_json()
        if not request_json:
            return jsonify({"error": "Missing JSON body"}), 400
        ean_code = request_json.get('ean_code')
        order_date = request_json.get('order_date')

        if not ean_code or not order_date:
            return jsonify({"error": "Missing 'ean_code' or 'order_date'"}), 400

        features = prepare_features(agg_df, ean_code, order_date, le)
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        predicted_quantity = max(0, int(round(prediction)))

        return jsonify({"predicted_quantity": predicted_quantity})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal prediction error", "details": str(e)}), 500
