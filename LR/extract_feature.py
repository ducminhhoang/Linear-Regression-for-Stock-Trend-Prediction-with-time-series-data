import numpy as np
from joblib import load
import pandas as pd



def extract(df):
    df_new = df[['open', 'high', 'low', 'close']]
    print(df_new)
    return df_new.values.flatten()

def predict_n_days(recent_data, n):
    scaler = load(r'D:\Hoc_tren_lop\MachineLearning\BTL\checkpoint\scaler.joblib')
    model = load(r'D:\Hoc_tren_lop\MachineLearning\BTL\checkpoint\model.joblib')
    model_open = load(r'D:\Hoc_tren_lop\MachineLearning\BTL\checkpoint\model_open.joblib')
    model_high = load(r'D:\Hoc_tren_lop\MachineLearning\BTL\checkpoint\model_high.joblib')
    model_low = load(r'D:\Hoc_tren_lop\MachineLearning\BTL\checkpoint\model_low.joblib')
    features = extract(recent_data)
    predictions = []
    current_data = features.copy()

    for _ in range(n):
        X_pred = current_data[:104]
        X_pred = scaler.transform(X_pred.reshape(1, -1))
        next_close = model.predict(X_pred)[0]  # Dự đoán giá trị tiếp theo
        predictions.append(next_close)

        # Cập nhật dữ liệu hiện tại với giá trị dự đoán
        next_open = model_open.predict(X_pred)[0]
        next_high = model_high.predict(X_pred)[0]
        next_low = model_low.predict(X_pred)[0]
        new_row = np.array([next_open, next_high, next_low, next_close])
        current_data = np.concatenate([new_row, current_data])

    return np.array(predictions)