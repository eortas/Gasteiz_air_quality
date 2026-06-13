import pandas as pd
import json
import joblib
from pathlib import Path
import urllib.request

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# 1. Fetch tomorrow's forecast
url = "https://api.open-meteo.com/v1/forecast?latitude=42.8467&longitude=-2.6716&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,boundary_layer_height&forecast_days=2&timezone=UTC&wind_speed_unit=ms"
response = urllib.request.urlopen(url)
data = json.loads(response.read().decode())

df_fc = pd.DataFrame(data["hourly"])
df_fc["timestamp"] = pd.to_datetime(df_fc["time"], utc=True)
df_fc["date"] = df_fc["timestamp"].dt.floor("D")

# Tomorrow date
tomorrow = pd.Timestamp.now(tz="UTC").floor("D") + pd.Timedelta(days=1)
df_tomorrow = df_fc[df_fc["date"] == tomorrow]

temp_fc = df_tomorrow["temperature_2m"].mean()
wind_fc = df_tomorrow["wind_speed_10m"].mean()
boundary_fc = df_tomorrow["boundary_layer_height"].mean()
humidity_fc = df_tomorrow["relative_humidity_2m"].mean()

print(f"Forecast for tomorrow ({tomorrow.date()}):")
print(f"  temp: {temp_fc:.2f} C")
print(f"  wind: {wind_fc:.2f} m/s")
print(f"  boundary_layer: {boundary_fc:.2f} m")
print(f"  humidity: {humidity_fc:.2f} %")

# Load hist and latest
df_hist = pd.read_parquet(PROCESSED_DIR / "features_daily.parquet")
df_hist["date"] = pd.to_datetime(df_hist["date"], utc=True)
df_hist = df_hist.sort_values("date").reset_index(drop=True)

target = "NO2_out_d1"
m1_path = MODELS_DIR / f"lgbm_v8_{target}.pkl"
f1_path = MODELS_DIR / f"lgbm_v8_{target}_features.json"
model_v1 = joblib.load(m1_path)
feats_v1 = json.loads(f1_path.read_text(encoding="utf-8"))

# Load predictions_latest.json to get v1 pred
with open(PROCESSED_DIR / "predictions_latest.json", "r", encoding="utf-8") as f:
    preds_latest = json.load(f)
pred_v1 = preds_latest["targets"][target]["prediction_v1"]

# Recompute errors
history_8d = df_hist.tail(8).copy()
errors = []
for idx, h_row in history_8d.iterrows():
    X_h = h_row.to_frame().T.reindex(columns=feats_v1, fill_value=0).fillna(0).astype(float)
    p_h = float(model_v1.predict(X_h)[0])
    actual = h_row.get(f"target_{target}")
    if pd.notna(actual):
        errors.append(actual - p_h)

error_lag_1d = errors[-1] if len(errors) >= 1 else 0
error_roll_7d = sum(errors)/len(errors) if len(errors) >= 1 else 0

# Calendar features for tomorrow
is_weekend = 1.0 if tomorrow.dayofweek >= 5 else 0.0
es_domingo = 1.0 if tomorrow.dayofweek == 6 else 0.0
es_invierno_estricto = 1.0 if tomorrow.month in [12, 1, 2] else 0.0

meta_input = {
    "pred_v1": float(pred_v1),
    "error_lag_1d": float(error_lag_1d),
    "error_roll_mean_7d": float(error_roll_7d),
    "temperature_2m": float(temp_fc),
    "wind_speed_10m": float(wind_fc),
    "boundary_layer_height": float(boundary_fc),
    "relative_humidity_2m": float(humidity_fc),
    "is_weekend": float(is_weekend),
    "es_domingo": float(es_domingo),
    "es_invierno_estricto": float(es_invierno_estricto),
}

print("\nCorrected meta-model input features for tomorrow:")
for k, v in meta_input.items():
    print(f"  {k}: {v}")

meta_model_path = MODELS_DIR / f"meta_model_{target}.pkl"
meta_model = joblib.load(meta_model_path)
pred_v2 = meta_model.predict(pd.DataFrame([meta_input]))[0]
print(f"\nCalculated prediction v2 (with fix): {pred_v2:.4f}")
