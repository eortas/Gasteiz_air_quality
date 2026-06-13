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

# Latest row
row = df_hist.iloc[[-1]]
date = row["date"].iloc[0]
print(f"Latest row date: {date.date()}")

# Print history NO2_out
print("\nRecent NO2_out actuals vs predictions (last 8 days):")
history_8d = df_hist.tail(8).copy()

target = "NO2_out_d1"
m1_path = MODELS_DIR / f"lgbm_v8_{target}.pkl"
f1_path = MODELS_DIR / f"lgbm_v8_{target}_features.json"
model_v1 = joblib.load(m1_path)
feats_v1 = json.loads(f1_path.read_text(encoding="utf-8"))

errors = []
for idx, h_row in history_8d.iterrows():
    X_h = h_row.to_frame().T.reindex(columns=feats_v1, fill_value=0).fillna(0).astype(float)
    p_h = float(model_v1.predict(X_h)[0])
    actual = h_row.get(f"target_{target}")
    err = actual - p_h if pd.notna(actual) else None
    if err is not None:
        errors.append(err)
    print(f"Date: {h_row['date'].date()} | Actual: {actual} | Pred v1: {p_h:.2f} | Error: {err}")

print(f"\nCollected errors: {errors}")
error_lag_1d = errors[-1] if len(errors) >= 1 else 0
error_roll_7d = sum(errors)/len(errors) if len(errors) >= 1 else 0
print(f"error_lag_1d: {error_lag_1d:.2f}")
print(f"error_roll_mean_7d: {error_roll_7d:.2f}")

# Meta model features
meta_model_path = MODELS_DIR / f"meta_model_{target}.pkl"
meta_model = joblib.load(meta_model_path)
print(f"\nMeta-model coefficients: {meta_model.coef_}")
print(f"Meta-model intercept: {meta_model.intercept_}")

# Let's inspect what is in predictions_latest.json
pred_path = PROCESSED_DIR / "predictions_latest.json"
if pred_path.exists():
    with open(pred_path, "r", encoding="utf-8") as f:
        latest = json.load(f)
    print(f"\nFrom predictions_latest.json for {target}:")
    print(json.dumps(latest["targets"][target], indent=2))
