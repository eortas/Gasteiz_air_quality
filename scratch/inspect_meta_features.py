import pandas as pd
import json
import joblib
from pathlib import Path

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

df_hist = pd.read_parquet(PROCESSED_DIR / "features_daily.parquet")
df_hist["date"] = pd.to_datetime(df_hist["date"], utc=True)
df_hist = df_hist.sort_values("date").reset_index(drop=True)

# Latest prediction row
pred_row_path = PROCESSED_DIR / "features_latest.parquet"
df_pred = pd.read_parquet(pred_row_path)
row = df_pred.iloc[[-1]]

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
    "pred_v1": float(pred_v1),
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

print("Meta-model input features for tomorrow:")
for k, v in meta_input.items():
    print(f"  {k}: {v}")

meta_model_path = MODELS_DIR / f"meta_model_{target}.pkl"
meta_model = joblib.load(meta_model_path)
pred_v2 = meta_model.predict(pd.DataFrame([meta_input]))[0]
print(f"\nCalculated prediction v2: {pred_v2:.4f}")
