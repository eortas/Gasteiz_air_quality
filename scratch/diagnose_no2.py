"""
Diagnóstico del error de predicción de NO2.
Compara valores reales vs predicciones para encontrar el origen del error.
"""
import pandas as pd
import json
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(r"c:\Users\ortas\OneDrive\Documentos\Vitoria_AG")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

print("=" * 70)
print("  DIAGNÓSTICO ERROR NO2 - Vitoria Air Quality")
print("=" * 70)

# 1. Cargar station_daily para valores reales
print("\n=== 1. DATOS REALES (station_daily.csv) ===")
csv_path = PROCESSED / "station_daily.csv"
df_st = pd.read_csv(csv_path)
print(f"Columnas: {df_st.columns.tolist()}")
print(f"Filas: {len(df_st)}")
date_col = "date"
dates = df_st[date_col].tail(5).tolist()
print(f"Ultimas 5 fechas: {dates}")
print()

# Ver columnas NO2
no2_cols = [c for c in df_st.columns if "NO2" in c.upper() or "no2" in c.lower()]
print(f"Columnas NO2: {no2_cols}")
print()
print("=== ULTIMOS 15 DIAS - NO2 ===")
print(df_st[["date"] + no2_cols].tail(15).to_string())

# 2. Cargar features_daily.parquet
print("\n\n=== 2. FEATURES_DAILY (parquet) ===")
df_feat = pd.read_parquet(PROCESSED / "features_daily.parquet")
df_feat["date"] = pd.to_datetime(df_feat["date"], utc=True)
print(f"Shape: {df_feat.shape}")
print(f"Rango: {df_feat['date'].min()} -> {df_feat['date'].max()}")

# Valores NO2 recientes
no2_feat_cols = [c for c in ["date", "NO2_zbe", "NO2_out", 
                              "target_NO2_zbe_d1", "target_NO2_out_d1",
                              "target_PM10_zbe_d1", "target_PM2.5_zbe_d1"] 
                 if c in df_feat.columns]
print("\n=== NO2 y targets recientes ===")
print(df_feat[no2_feat_cols].tail(15).to_string())

# 3. features_latest.parquet - fila que se usa para predecir
print("\n\n=== 3. FEATURES_LATEST (fila de predicción) ===")
df_latest = pd.read_parquet(PROCESSED / "features_latest.parquet")
df_latest["date"] = pd.to_datetime(df_latest["date"], utc=True)
print(f"Shape: {df_latest.shape}")
last_date = df_latest["date"].iloc[-1]
print(f"Ultima fecha: {last_date}")

last_row = df_latest.iloc[-1]
key_feats = [
    "NO2_zbe", "NO2_out", "NO2_zbe_roll_mean_7d", "NO2_zbe_lag_1d", 
    "NO2_zbe_lag_2d", "NO2_out_roll_mean_7d",
    "PM10_zbe", "PM10_out", "PM10_zbe_roll_mean_7d",
    "PM2.5_zbe", "PM2.5_out",
    "fc_temperature_2m_d1", "fc_wind_speed_10m_d1", "HDD", "HDD_acum_7d",
    "fc_HDD_d1",
]

print("\nFeatures clave de la ultima fila:")
for f in key_feats:
    if f in last_row.index:
        val = last_row[f]
        is_nan = pd.isna(val) if not isinstance(val, str) else False
        print(f"  {f:40s} = {val}  {'*** NaN ***' if is_nan else ''}")
    else:
        print(f"  {f:40s} = *** NO ENCONTRADA ***")

# 4. Verificar qué ocurre al predecir con el modelo NO2
print("\n\n=== 4. SIMULACIÓN DE PREDICCIÓN NO2 ===")

for target in ["NO2_zbe_d1", "NO2_out_d1", "PM10_zbe_d1", "PM2.5_zbe_d1"]:
    model_path = MODELS / f"lgbm_v8_{target}.pkl"
    feat_path = MODELS / f"lgbm_v8_{target}_features.json"
    median_path = MODELS / f"lgbm_v8_{target}_medians.json"
    
    if not model_path.exists():
        print(f"  {target}: MODELO NO ENCONTRADO")
        continue
    
    model = joblib.load(model_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    medians = json.loads(median_path.read_text(encoding="utf-8"))
    
    # Construir vector de entrada como lo hace predict.py
    row = df_latest.iloc[[-1]]
    fill_values = {f: medians.get(f, 0) for f in features}
    
    # Verificar cuántas features están NaN vs presentes
    missing = []
    nan_filled = []
    for f in features:
        if f not in row.columns:
            missing.append(f)
        elif pd.isna(row[f].iloc[0]):
            nan_filled.append(f)
    
    X = row.reindex(columns=features).fillna(fill_values).astype(float)
    pred = float(model.predict(X)[0])
    pred = max(0.0, pred)
    
    # También obtener contribuciones
    contribs = model.predict(X, pred_contrib=True)
    base_value = float(contribs[0, -1])
    feats_impact = list(zip(features, contribs[0, :-1]))
    feats_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n--- {target} ---")
    print(f"  Predicción base (v1): {pred:.2f}")
    print(f"  Base value (SHAP):    {base_value:.2f}")
    print(f"  Features faltantes:   {len(missing)}")
    print(f"  Features NaN->median: {len(nan_filled)}")
    
    if nan_filled:
        print(f"  [CLAVE] Features que son NaN y se rellenan con mediana:")
        for f in nan_filled[:10]:
            med_val = medians.get(f, 0)
            print(f"    {f:45s} -> mediana={med_val:.3f}")
    
    # Top 5 contribuciones positivas y negativas
    print(f"  Top 5 contribuciones POSITIVAS:")
    pos = [x for x in feats_impact if x[1] > 0]
    for f, v in pos[:5]:
        actual_val = X[f].iloc[0]
        med_val = medians.get(f, 0)
        print(f"    {f:45s} contrib={v:+.3f}  val={actual_val:.3f}  med={med_val:.3f}")
    
    print(f"  Top 5 contribuciones NEGATIVAS:")
    neg = [x for x in feats_impact if x[1] < 0]
    for f, v in neg[:5]:
        actual_val = X[f].iloc[0]
        med_val = medians.get(f, 0)
        print(f"    {f:45s} contrib={v:+.3f}  val={actual_val:.3f}  med={med_val:.3f}")

# 5. Backtest: comparar predicciones pasadas vs reales
print("\n\n=== 5. BACKTEST - Error NO2 vs PM10/PM2.5 ultimos 30 dias ===")
df_bt = df_feat.copy()
for target in ["NO2_zbe_d1", "NO2_out_d1", "PM10_zbe_d1", "PM10_out_d1", "PM2.5_zbe_d1", "PM2.5_out_d1"]:
    model_path = MODELS / f"lgbm_v8_{target}.pkl"
    feat_path = MODELS / f"lgbm_v8_{target}_features.json"
    median_path = MODELS / f"lgbm_v8_{target}_medians.json"
    
    if not model_path.exists():
        continue
    
    model = joblib.load(model_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    medians = json.loads(median_path.read_text(encoding="utf-8"))
    
    target_col = f"target_{target}"
    if target_col not in df_bt.columns:
        continue
    
    # Usar ultimos 30 dias con targets disponibles
    mask = df_bt[target_col].notna()
    df_test = df_bt[mask].tail(30)
    
    if len(df_test) == 0:
        print(f"  {target}: Sin datos de test")
        continue
    
    fill_values = {f: medians.get(f, 0) for f in features}
    X_test = df_test.reindex(columns=features).fillna(fill_values).astype(float)
    y_true = df_test[target_col].values
    y_pred = model.predict(X_test).clip(min=0)
    
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae_val = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (solo donde y_true > 3)
    mask_mape = y_true > 3
    if mask_mape.sum() > 0:
        mape_val = np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100
    else:
        mape_val = float("nan")
    
    # Error porcentual medio
    mean_real = y_true.mean()
    mean_pred = y_pred.mean()
    pct_error = (mean_pred - mean_real) / mean_real * 100 if mean_real > 0 else 0
    
    print(f"  {target:20s}  RMSE={rmse_val:.2f}  MAE={mae_val:.2f}  MAPE={mape_val:.1f}%  "
          f"Real_media={mean_real:.2f}  Pred_media={mean_pred:.2f}  Error%={pct_error:+.1f}%")

# 6. Análisis de la distribución del NO2 en verano vs invierno
print("\n\n=== 6. DISTRIBUCIÓN NO2 POR ESTACIÓN ===")
df_feat["month"] = df_feat["date"].dt.month
verano = df_feat[df_feat["month"].isin([6, 7, 8, 9])]
invierno = df_feat[df_feat["month"].isin([11, 12, 1, 2])]

for zone in ["zbe", "out"]:
    col = f"NO2_{zone}"
    if col in df_feat.columns:
        print(f"  {col} verano:   media={verano[col].mean():.2f}  std={verano[col].std():.2f}")
        print(f"  {col} invierno: media={invierno[col].mean():.2f}  std={invierno[col].std():.2f}")
        print(f"  {col} global:   media={df_feat[col].mean():.2f}  std={df_feat[col].std():.2f}")
        print()

# 7. ¿El modelo NO2 fue entrenado con datos de verano?
print("\n=== 7. DISTRIBUCIÓN TEMPORAL DEL TRAINING SET ===")
df_feat["year_month"] = df_feat["date"].dt.strftime("%Y-%m")
print(df_feat.groupby("year_month")["NO2_zbe"].agg(["count", "mean", "std"]).tail(18).to_string())

print("\n\n=== FIN DIAGNÓSTICO ===")
