"""
prepare_meta_data.py
===================
Genera el dataset de entrenamiento para el Meta-modelo de corrección.
Evita el data leakage usando Out-of-Fold (OOF) predictions siguiendo
exactamente la misma lógica de TimeSeriesSplit de train_model_v8.py.

Salida:
  - data/processed/meta_training_data.parquet
"""

import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"
OUT_PATH      = PROCESSED_DIR / "meta_training_data.parquet"

# ─── CONFIG (Igual que train_model_v8.py) ───────────────────────────────────
TARGETS  = ["NO2", "PM10", "PM2.5", "ICA"]
ZONES    = ["zbe", "out"]
N_SPLITS = 5
HORIZON  = "d1"

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# Features exógenas de interés para el meta-modelo
EXOGENOUS_FEATS = [
    "temperature_2m", "precipitation", "wind_speed_10m", "wind_u", "wind_v",
    "boundary_layer_height", "cloud_cover", "relative_humidity_2m",
    "sunshine_duration", "is_weekend", "day_of_week", "month", "season",
    "es_domingo", "es_invierno_estricto", "domingo_invierno",
    "exp_traffic_volume_d1", "exp_traffic_occupancy_d1"
]

def load_dataset():
    if not DATASET_PATH.exists():
        print(f"ERR: No se encuentra {DATASET_PATH}")
        return None
    df = pd.read_parquet(DATASET_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
    return df

def generate_oof_predictions(df):
    meta_rows = []
    
    # 1. Identificar targets reales (sin el prefijo 'target_')
    all_oof_results = []

    for cont in TARGETS:
        for zone in ZONES:
            target_key = f"{cont}_{zone}_{HORIZON}"
            y_col = f"target_{target_key}"
            
            if y_col not in df.columns:
                print(f"WARN: Target {y_col} no encontrado. Saltando...")
                continue
            
            print(f"\nGenerando OOF para {target_key}...")
            
            # Cargar features seleccionadas para este modelo (v8)
            feat_path = MODELS_DIR / f"lgbm_v8_{target_key}_features.json"
            if not feat_path.exists():
                print(f"WARN: No hay features guardadas para {target_key}. Saltando...")
                continue
            
            features = json.loads(feat_path.read_text(encoding="utf-8"))
            
            # Preparar datos (mismo pre-procesamiento que v8)
            df_target = df.dropna(subset=[y_col]).copy()
            X = df_target[features].fillna(0)
            y = df_target[y_col]
            dates = df_target["date"]
            
            tscv = TimeSeriesSplit(n_splits=N_SPLITS)
            
            target_oof = pd.Series(index=df_target.index, dtype=float)
            
            for tr_idx, va_idx in tscv.split(X):
                model = LGBMRegressor(**LGBM_PARAMS)
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                target_oof.iloc[va_idx] = model.predict(X.iloc[va_idx])
            
            # Guardamos solo las filas que tienen predicción (las del primer train fold son NaN)
            valid_oof = target_oof.dropna()
            
            res_df = pd.DataFrame({
                "date": dates.loc[valid_oof.index],
                "target_name": target_key,
                "pred_v1": valid_oof.values,
                "actual": y.loc[valid_oof.index].values
            })
            
            # Añadir variables exógenas
            for feat in EXOGENOUS_FEATS:
                if feat in df.columns:
                    res_df[feat] = df_target.loc[valid_oof.index, feat].values
            
            all_oof_results.append(res_df)

    if not all_oof_results:
        return None
        
    final_meta_df = pd.concat(all_oof_results).reset_index(drop=True)
    
    # Calcular errores históricos (Error en t-1)
    # Necesitamos pivotar o procesar por target para no mezclar dateranges
    sorted_meta = []
    for t_name in final_meta_df["target_name"].unique():
        sub = final_meta_df[final_meta_df["target_name"] == t_name].sort_values("date")
        sub["error"] = sub["actual"] - sub["pred_v1"]
        
        # El error que conocemos hoy es el del día anterior (t-1)
        sub["error_lag_1d"] = sub["error"].shift(1)
        sub["error_roll_mean_7d"] = sub["error"].shift(1).rolling(7).mean()
        
        sorted_meta.append(sub)
        
    return pd.concat(sorted_meta).dropna().reset_index(drop=True)

def main():
    print("Iniciando preparación de datos para Meta-modelo...")
    df = load_dataset()
    if df is None: return
    
    meta_df = generate_oof_predictions(df)
    
    if meta_df is not None:
        meta_df.to_parquet(OUT_PATH)
        print(f"\n[OK] Dataset de meta-entrenamiento guardado: {OUT_PATH}")
        print(f"Total registros: {len(meta_df)}")
        print(meta_df.head())
    else:
        print("\n[ERR] No se han podido generar datos meta.")

if __name__ == "__main__":
    main()
