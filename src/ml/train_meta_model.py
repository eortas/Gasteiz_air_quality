"""
train_meta_model.py
===================
Entrena los meta-modelos (correctores) para cada target.
Usa Ridge Regression para que la corrección sea estable y lineal,
ajustando la predicción v1 según los errores recientes y la meteorología.

Salida:
  - models/meta_model_{target}.pkl
  - models/meta_metrics.json
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent.parent
PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models"
META_DATA_PATH  = PROCESSED_DIR / "meta_training_data.parquet"
METRICS_PATH    = MODELS_DIR / "meta_metrics.json"

TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
    "ICA_zbe_d1", "ICA_out_d1",
]

# Features de entrada para el Meta-Modelo
META_FEATURES = [
    "pred_v1", 
    "error_lag_1d", 
    "error_roll_mean_7d",
    "temperature_2m", "wind_speed_10m",
    "boundary_layer_height", "relative_humidity_2m",
    "is_weekend", "es_domingo", "es_invierno_estricto"
]

def load_data():
    if not META_DATA_PATH.exists():
        print(f"ERR: No existe {META_DATA_PATH}. Ejecuta primero prepare_meta_data.py")
        return None
    return pd.read_parquet(META_DATA_PATH)

def train_meta_models(df):
    meta_results = {}
    
    print(f"\nEntrenando Meta-Modelos (Correctores) RIDGE - {datetime.now().date()}")
    print(f"{'Target':<20} | {'RMSE v1':>10} -> {'RMSE v2':>10} | {'Mejora %':>8}")
    print("-" * 65)

    for target in TARGETS:
        sub = df[df["target_name"] == target].sort_values("date").reset_index(drop=True)
        if sub.empty: continue
        
        # Split temporal: entrenamos con el 80% inicial y validamos con el 20% final
        split = int(len(sub) * 0.8)
        train_sub = sub.iloc[:split]
        test_sub  = sub.iloc[split:]
        
        X_train = train_sub[META_FEATURES].fillna(0)
        y_train = train_sub["actual"]
        
        X_test  = test_sub[META_FEATURES].fillna(0)
        y_test  = test_sub["actual"]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Predicci?n del meta-modelo (v2)
        y_pred_v2 = model.predict(X_test)
        y_pred_v1 = test_sub["pred_v1"] # La predicci?n original
        
        rmse_v1 = np.sqrt(mean_squared_error(y_test, y_pred_v1))
        rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
        gain = (rmse_v1 - rmse_v2) / rmse_v1 * 100
        
        # Guardar modelo
        model_path = MODELS_DIR / f"meta_model_{target}.pkl"
        joblib.dump(model, model_path)
        
        meta_results[target] = {
            "rmse_v1": round(float(rmse_v1), 4),
            "rmse_v2": round(float(rmse_v2), 4),
            "improvement_pct": round(float(gain), 2),
            "r2_v2": round(float(r2_score(y_test, y_pred_v2)), 4),
            "coefficients": dict(zip(META_FEATURES, model.coef_.tolist()))
        }
        
        status = "[OK]" if gain > 0 else "[--]"
        print(f"{target:<20} | {rmse_v1:>8.3f} -> {rmse_v2:>8.3f} | {gain:>7.1f}% {status}")

    return meta_results

def main():
    df = load_data()
    if df is None: return
    
    results = train_meta_models(df)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Modelos guardados en {MODELS_DIR}")
    print(f"[OK] Métricas guardadas en {METRICS_PATH}")

if __name__ == "__main__":
    main()
