import pandas as pd
import numpy as np
import joblib
import json
import requests
from pathlib import Path

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Simulating predict.py load_prediction_row
df = pd.read_parquet(PROCESSED_DIR / "features_daily.parquet")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)

df_pred_row = pd.read_parquet(PROCESSED_DIR / "features_latest.parquet")
df_pred_row["date"] = pd.to_datetime(df_pred_row["date"], utc=True)
pred_row_date = df_pred_row["date"].iloc[-1]
last_training_date = df["date"].iloc[-1]

print(f"last_training_date in features_daily: {last_training_date.date()}")
print(f"pred_row_date in features_latest: {pred_row_date.date()}")

if pred_row_date > last_training_date:
    row = df_pred_row.iloc[[-1]]
    pred_date = pred_row_date + pd.Timedelta(days=1)
else:
    row = df.iloc[[-1]]
    pred_date = last_training_date + pd.Timedelta(days=1)

print(f"Using row: {row['date'].iloc[0].date()}")
print(f"Predicting date: {pred_date.date()}")

# Download real forecast
print("Downloading real forecast from Open-Meteo...")
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude":       42.8467,
    "longitude":      -2.6716,
    "hourly":         ",".join([
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "rain", "snowfall", "wind_speed_10m", "wind_direction_10m",
        "wind_gusts_10m", "cloud_cover", "boundary_layer_height",
        "sunshine_duration", "weather_code",
    ]),
    "forecast_days":  3,
    "timezone":       "UTC",
    "wind_speed_unit": "ms",
}
r = requests.get(url, params=params, timeout=15)
data = r.json()
df_fc = pd.DataFrame(data.get("hourly", {}))
df_fc["timestamp"] = pd.to_datetime(df_fc["time"], utc=True)
df_fc["date"]      = df_fc["timestamp"].dt.floor("D")

tomorrow = pred_date.normalize()
df_tomorrow = df_fc[df_fc["date"] == tomorrow]
forecast_override = {}
for col in ["temperature_2m", "wind_speed_10m", "boundary_layer_height", "relative_humidity_2m"]:
    forecast_override[f"fc_{col}_d1"] = df_tomorrow[col].mean()

# Apply forecast override to row
row = row.copy()
for feat, val in forecast_override.items():
    row[feat] = val
    print(f"Forecast override: {feat} = {val}")

# Now let's trace refine_with_meta_models for NO2_out_d1
results = {
    "NO2_out_d1": {
        "prediction": 10.97,
        "rmse_cv": 4.49
    },
    "NO2_zbe_d1": {
        "prediction": 10.27,
        "rmse_cv": 3.915
    }
}

for target, r in results.items():
    print(f"\n=== TRACING {target} ===")
    cutoff_date = pred_date - pd.Timedelta(days=3)
    df_history_clean = df[df["date"] <= cutoff_date].sort_values("date").reset_index(drop=True)
    history_8d = df_history_clean.tail(8).copy()
    
    # Load model
    m1_path = MODELS_DIR / f"lgbm_v8_{target}.pkl"
    f1_path = MODELS_DIR / f"lgbm_v8_{target}_features.json"
    model_v1 = joblib.load(m1_path)
    feats_v1 = json.loads(f1_path.read_text(encoding="utf-8"))
    
    errors = []
    for _, h_row in history_8d.iterrows():
        X_h = h_row.to_frame().T.reindex(columns=feats_v1, fill_value=0).fillna(0).astype(float)
        p_h = float(model_v1.predict(X_h)[0])
        actual = h_row.get(f"target_{target}")
        if pd.notna(actual):
            errors.append(actual - p_h)
            
    error_lag_1d = errors[-1] if len(errors) >= 1 else 0
    error_roll_7d = np.mean(errors) if len(errors) >= 1 else 0
    
    # Load meta model
    model_meta_path = MODELS_DIR / f"meta_model_{target}.pkl"
    meta_model = joblib.load(model_meta_path)
    
    temp_val = row["temperature_2m"].iloc[0]
    if pd.isna(temp_val) and "fc_temperature_2m_d1" in row.columns:
        temp_val = row["fc_temperature_2m_d1"].iloc[0]
        
    wind_val = row["wind_speed_10m"].iloc[0]
    if pd.isna(wind_val) and "fc_wind_speed_10m_d1" in row.columns:
        wind_val = row["fc_wind_speed_10m_d1"].iloc[0]
        
    boundary_val = row["boundary_layer_height"].iloc[0]
    if pd.isna(boundary_val) and "fc_boundary_layer_height_d1" in row.columns:
        boundary_val = row["fc_boundary_layer_height_d1"].iloc[0]
        
    humidity_val = row["relative_humidity_2m"].iloc[0]
    if pd.isna(humidity_val) and "fc_relative_humidity_2m_d1" in row.columns:
        humidity_val = row["fc_relative_humidity_2m_d1"].iloc[0]

    meta_input = {
        "pred_v1": float(r["prediction"]),
        "error_lag_1d": float(error_lag_1d),
        "error_roll_mean_7d": float(error_roll_7d),
        "temperature_2m": float(pd.to_numeric(pd.Series([temp_val]), errors='coerce').fillna(0).iloc[0]),
        "wind_speed_10m": float(pd.to_numeric(pd.Series([wind_val]), errors='coerce').fillna(0).iloc[0]),
        "boundary_layer_height": float(pd.to_numeric(pd.Series([boundary_val]), errors='coerce').fillna(0).iloc[0]),
        "relative_humidity_2m": float(pd.to_numeric(pd.Series([humidity_val]), errors='coerce').fillna(0).iloc[0]),
        "is_weekend": float(pd.to_numeric(row["is_weekend"], errors='coerce').fillna(0).iloc[0]),
        "es_domingo": float(pd.to_numeric(row["es_domingo"], errors='coerce').fillna(0).iloc[0]),
        "es_invierno_estricto": float(pd.to_numeric(row["es_invierno_estricto"], errors='coerce').fillna(0).iloc[0]),
    }
    
    print("\nMeta Input:")
    for k, v in meta_input.items():
        print(f"  {k}: {v}")
        
    X_meta = pd.DataFrame([meta_input]).astype(float)
    pred_v2 = float(meta_model.predict(X_meta)[0])
    print(f"\nFinal Pred_v2: {pred_v2:.2f} (correction: {pred_v2 - r['prediction']:.2f})")

