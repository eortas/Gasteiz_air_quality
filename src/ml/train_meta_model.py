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
    df = pd.read_parquet(META_DATA_PATH)
    # Fix auditoría #4: Entrenar con datos completos, no solo post-ZBE
    # El filtro anterior (df["date"] >= "2025-09-01") limitaba el entrenamiento
    # al periodo post-ZBE, lo que causaba sobreajuste y hacía necesario un bypass frágil
    df["date"] = pd.to_datetime(df["date"], utc=True)
    print(f"  Datos cargados: {len(df)} filas ({df['date'].min().date()} -> {df['date'].max().date()})")
    return df

def train_meta_models(df):
    """
    Entrena meta-modelos con validación walk-forward de 3 folds.

    Fix auditoría #4:
    - Entrena con datos completos (no solo post-ZBE)
    - Valida con walk-forward temporal (3 folds)
    - Solo guarda el meta-modelo si mejora RMSE en TODOS los folds
    - Si no mejora, NO se guarda → predict.py usará v1 directamente
    """
    meta_results = {}

    print(f"\nEntrenando Meta-Modelos (Correctores) RIDGE - {datetime.now().date()}")
    print(f"  Validación: Walk-forward temporal (3 folds)")
    print(f"{'Target':<20} | {'RMSE v1':>10} -> {'RMSE v2':>10} | {'Mejora %':>8} | {'Guardado':>8}")
    print("-" * 75)

    n_folds = 3

    for target in TARGETS:
        sub = df[df["target_name"] == target].sort_values("date").reset_index(drop=True)
        if len(sub) < 30:
            continue

        # Walk-forward: dividir en n_folds + 1 bloques temporales
        fold_size = len(sub) // (n_folds + 1)
        fold_improvements = []

        for fold_idx in range(n_folds):
            # Train: todo hasta el inicio del fold de validación
            train_end = fold_size * (fold_idx + 1)
            val_start = train_end
            val_end   = min(train_end + fold_size, len(sub))

            if val_end - val_start < 5:
                continue

            train_sub = sub.iloc[:train_end]
            test_sub  = sub.iloc[val_start:val_end]

            # Imputar NaN con media del train
            X_train_raw = train_sub[META_FEATURES]
            X_test_raw  = test_sub[META_FEATURES]
            means = X_train_raw.mean().fillna(0.0)

            X_train = X_train_raw.fillna(means)
            y_train = train_sub["actual"]
            X_test  = X_test_raw.fillna(means)
            y_test  = test_sub["actual"]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            y_pred_v2 = model.predict(X_test)
            y_pred_v1 = test_sub["pred_v1"]

            rmse_v1 = np.sqrt(mean_squared_error(y_test, y_pred_v1))
            rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
            fold_improvements.append(rmse_v2 < rmse_v1)

        # Decidir si guardar: solo si mejora en TODOS los folds
        all_folds_improve = len(fold_improvements) > 0 and all(fold_improvements)

        # Métricas finales con split 80/20 para reportar
        split = int(len(sub) * 0.8)
        train_sub = sub.iloc[:split]
        test_sub  = sub.iloc[split:]

        X_train_raw = train_sub[META_FEATURES]
        X_test_raw  = test_sub[META_FEATURES]
        means = X_train_raw.mean().fillna(0.0)

        X_train = X_train_raw.fillna(means)
        y_train = train_sub["actual"]
        X_test  = X_test_raw.fillna(means)
        y_test  = test_sub["actual"]

        model_eval = Ridge(alpha=1.0)
        model_eval.fit(X_train, y_train)

        y_pred_v2 = model_eval.predict(X_test)
        y_pred_v1 = test_sub["pred_v1"]

        rmse_v1 = np.sqrt(mean_squared_error(y_test, y_pred_v1))
        rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
        gain = (rmse_v1 - rmse_v2) / rmse_v1 * 100

        if all_folds_improve:
            # Entrenar modelo final con todo el histórico
            X_full_raw = sub[META_FEATURES]
            full_means = X_full_raw.mean().fillna(0.0)
            X_full = X_full_raw.fillna(full_means)
            y_full = sub["actual"]
            final_model = Ridge(alpha=1.0)
            final_model.fit(X_full, y_full)

            model_path = MODELS_DIR / f"meta_model_{target}.pkl"
            joblib.dump(final_model, model_path)
            saved = True
        else:
            # No guardar: eliminar modelo existente si lo hay
            model_path = MODELS_DIR / f"meta_model_{target}.pkl"
            if model_path.exists():
                model_path.unlink()
            saved = False

        meta_results[target] = {
            "rmse_v1": round(float(rmse_v1), 4),
            "rmse_v2": round(float(rmse_v2), 4),
            "improvement_pct": round(float(gain), 2),
            "r2_v2": round(float(r2_score(y_test, y_pred_v2)), 4),
            "saved": saved,
            "folds_improved": sum(fold_improvements),
            "folds_total": len(fold_improvements),
        }

        status = "[OK]" if saved else "[SKIP]"
        print(f"{target:<20} | {rmse_v1:>8.3f} -> {rmse_v2:>8.3f} | {gain:>7.1f}% {status:>8}"
              f"  ({sum(fold_improvements)}/{len(fold_improvements)} folds)")

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
