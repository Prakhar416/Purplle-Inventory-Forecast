import pandas as pd

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
