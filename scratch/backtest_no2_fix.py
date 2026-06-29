"""
Backtest: Compara predicciones OLD (medianas anuales, sin clamp) vs NEW (estacionales + clamp)
sobre los últimos 60 días del dataset para verificar que el fix de NO2 funciona.
"""
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(r"c:\Users\ortas\OneDrive\Documentos\Vitoria_AG")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
]

print("=" * 75)
print("  BACKTEST: OLD (medianas anuales) vs NEW (medianas estacionales)")
print("=" * 75)

# Cargar dataset
df = pd.read_parquet(PROCESSED / "features_daily.parquet")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)

# Cargar meta-modelos
meta_models = {}
meta_metrics_path = MODELS / "meta_metrics.json"
if meta_metrics_path.exists():
    meta_metrics = json.loads(meta_metrics_path.read_text(encoding="utf-8"))

for target in TARGETS:
    model_path = MODELS / f"lgbm_v8_{target}.pkl"
    feat_path = MODELS / f"lgbm_v8_{target}_features.json"
    median_path = MODELS / f"lgbm_v8_{target}_medians.json"
    seasonal_path = MODELS / f"lgbm_v8_{target}_medians_seasonal.json"
    meta_path = MODELS / f"meta_model_{target}.pkl"
    
    if not model_path.exists() or not feat_path.exists():
        continue
    
    model = joblib.load(model_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    global_medians = json.loads(median_path.read_text(encoding="utf-8")) if median_path.exists() else {}
    seasonal_medians = json.loads(seasonal_path.read_text(encoding="utf-8")) if seasonal_path.exists() else {}
    
    meta_model = joblib.load(meta_path) if meta_path.exists() else None
    
    target_col = f"target_{target}"
    if target_col not in df.columns:
        continue
    
    # Últimos 60 días con targets disponibles
    mask = df[target_col].notna()
    df_test = df[mask].tail(60).copy()
    
    if len(df_test) == 0:
        continue
    
    y_true = df_test[target_col].values
    dates = df_test["date"].values
    months = pd.to_datetime(df_test["date"]).dt.month.values
    
    # ─── Método OLD: medianas globales ────────────────
    fill_old = {f: global_medians.get(f, 0) for f in features}
    X_old = df_test.reindex(columns=features).fillna(fill_old).astype(float)
    y_pred_old_v1 = model.predict(X_old).clip(min=0)
    
    # Aplicar meta-modelo OLD (sin clamp)
    y_pred_old = y_pred_old_v1.copy()
    if meta_model is not None:
        for i in range(len(y_pred_old)):
            try:
                meta_input = pd.DataFrame([{
                    "pred_v1": float(y_pred_old_v1[i]),
                    "error_lag_1d": 0.0,
                    "error_roll_mean_7d": 0.0,
                    "temperature_2m": float(df_test.iloc[i].get("temperature_2m", 12.0)) if pd.notna(df_test.iloc[i].get("temperature_2m")) else 12.0,
                    "wind_speed_10m": float(df_test.iloc[i].get("wind_speed_10m", 2.5)) if pd.notna(df_test.iloc[i].get("wind_speed_10m")) else 2.5,
                    "boundary_layer_height": float(df_test.iloc[i].get("boundary_layer_height", 500)) if pd.notna(df_test.iloc[i].get("boundary_layer_height")) else 500.0,
                    "relative_humidity_2m": float(df_test.iloc[i].get("relative_humidity_2m", 75)) if pd.notna(df_test.iloc[i].get("relative_humidity_2m")) else 75.0,
                    "is_weekend": float(df_test.iloc[i].get("is_weekend", 0)) if pd.notna(df_test.iloc[i].get("is_weekend")) else 0.0,
                    "es_domingo": float(df_test.iloc[i].get("es_domingo", 0)) if pd.notna(df_test.iloc[i].get("es_domingo")) else 0.0,
                    "es_invierno_estricto": float(df_test.iloc[i].get("es_invierno_estricto", 0)) if pd.notna(df_test.iloc[i].get("es_invierno_estricto")) else 0.0,
                }]).astype(float)
                y_pred_old[i] = max(0, float(meta_model.predict(meta_input)[0]))
            except Exception:
                pass
    
    # ─── Método NEW: medianas estacionales + clamp ────
    y_pred_new_v1 = np.zeros(len(df_test))
    for i in range(len(df_test)):
        month = months[i]
        month_key = str(month)
        month_meds = seasonal_medians.get(month_key, {})
        
        fill_new = {}
        for f in features:
            if f in month_meds:
                fill_new[f] = month_meds[f]
            elif f in global_medians:
                fill_new[f] = global_medians[f]
            else:
                fill_new[f] = 0
        
        X_row = df_test.iloc[[i]].reindex(columns=features).fillna(fill_new).astype(float)
        y_pred_new_v1[i] = max(0, float(model.predict(X_row)[0]))
    
    # Aplicar meta-modelo NEW (con bypass para NO2 en verano y clamp ±30% para el resto)
    y_pred_new = y_pred_new_v1.copy()
    if meta_model is not None:
        for i in range(len(y_pred_new)):
            try:
                meta_input = pd.DataFrame([{
                    "pred_v1": float(y_pred_new_v1[i]),
                    "error_lag_1d": 0.0,
                    "error_roll_mean_7d": 0.0,
                    "temperature_2m": float(df_test.iloc[i].get("temperature_2m", 12.0)) if pd.notna(df_test.iloc[i].get("temperature_2m")) else 12.0,
                    "wind_speed_10m": float(df_test.iloc[i].get("wind_speed_10m", 2.5)) if pd.notna(df_test.iloc[i].get("wind_speed_10m")) else 2.5,
                    "boundary_layer_height": float(df_test.iloc[i].get("boundary_layer_height", 500)) if pd.notna(df_test.iloc[i].get("boundary_layer_height")) else 500.0,
                    "relative_humidity_2m": float(df_test.iloc[i].get("relative_humidity_2m", 75)) if pd.notna(df_test.iloc[i].get("relative_humidity_2m")) else 75.0,
                    "is_weekend": float(df_test.iloc[i].get("is_weekend", 0)) if pd.notna(df_test.iloc[i].get("is_weekend")) else 0.0,
                    "es_domingo": float(df_test.iloc[i].get("es_domingo", 0)) if pd.notna(df_test.iloc[i].get("es_domingo")) else 0.0,
                    "es_invierno_estricto": float(df_test.iloc[i].get("es_invierno_estricto", 0)) if pd.notna(df_test.iloc[i].get("es_invierno_estricto")) else 0.0,
                }]).astype(float)
                pred_meta = max(0, float(meta_model.predict(meta_input)[0]))
                
                v1 = y_pred_new_v1[i]
                is_no2 = "NO2" in target
                global_median_target = None
                if is_no2:
                    target_medians_map = {
                        "NO2_zbe_d1": 10.0,
                        "NO2_out_d1": 11.5,
                    }
                    global_median_target = target_medians_map.get(target, 10.0)
                
                if is_no2 and global_median_target and v1 < global_median_target:
                    # BYPASS: usar v1 directamente
                    pred_v2 = v1
                else:
                    # CLAMP: ±30%
                    max_correction_pct = 0.30
                    if v1 > 0:
                        lo = v1 * (1 - max_correction_pct)
                        hi = v1 * (1 + max_correction_pct)
                        pred_v2 = max(0.0, min(pred_meta, hi))
                        pred_v2 = max(lo, pred_v2)
                    else:
                        pred_v2 = pred_meta
                
                y_pred_new[i] = pred_v2
            except Exception:
                pass
    
    # ─── Métricas ────────────────────────────────────
    def calc_metrics(y_t, y_p, label=""):
        rmse_val = np.sqrt(np.mean((y_t - y_p) ** 2))
        mae_val = np.mean(np.abs(y_t - y_p))
        mask_mape = y_t > 3
        mape_val = np.mean(np.abs((y_t[mask_mape] - y_p[mask_mape]) / y_t[mask_mape])) * 100 if mask_mape.sum() > 0 else float("nan")
        mean_real = y_t.mean()
        mean_pred = y_p.mean()
        bias_pct = (mean_pred - mean_real) / mean_real * 100 if mean_real > 0 else 0
        return rmse_val, mae_val, mape_val, mean_real, mean_pred, bias_pct
    
    r_old = calc_metrics(y_true, y_pred_old)
    r_new = calc_metrics(y_true, y_pred_new)
    r_v1_old = calc_metrics(y_true, y_pred_old_v1)
    r_v1_new = calc_metrics(y_true, y_pred_new_v1)
    
    print(f"\n{'='*75}")
    print(f"  {target}")
    print(f"{'='*75}")
    print(f"  {'Método':<35} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'Bias%':>8} {'PredMedia':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'OLD v1 (medianas anuales)':<35} {r_v1_old[0]:>8.2f} {r_v1_old[1]:>8.2f} {r_v1_old[2]:>8.1f} {r_v1_old[5]:>+8.1f} {r_v1_old[4]:>10.2f}")
    print(f"  {'OLD v2 (meta sin clamp)':<35} {r_old[0]:>8.2f} {r_old[1]:>8.2f} {r_old[2]:>8.1f} {r_old[5]:>+8.1f} {r_old[4]:>10.2f}")
    print(f"  {'NEW v1 (medianas estacionales)':<35} {r_v1_new[0]:>8.2f} {r_v1_new[1]:>8.2f} {r_v1_new[2]:>8.1f} {r_v1_new[5]:>+8.1f} {r_v1_new[4]:>10.2f}")
    print(f"  {'NEW v2 (estacional + clamp 50%)':<35} {r_new[0]:>8.2f} {r_new[1]:>8.2f} {r_new[2]:>8.1f} {r_new[5]:>+8.1f} {r_new[4]:>10.2f}")
    print(f"  Real media: {r_old[3]:.2f}")
    
    # Mejora
    rmse_improve = (r_old[0] - r_new[0]) / r_old[0] * 100
    bias_improve = abs(r_old[5]) - abs(r_new[5])
    print(f"  >>> RMSE mejora: {rmse_improve:+.1f}%")
    print(f"  >>> Bias mejora: {abs(r_old[5]):.1f}% -> {abs(r_new[5]):.1f}%")
    
    # Últimos 30 días (período más problemático - verano)
    if len(df_test) >= 30:
        last30 = slice(-30, None)
        r_old_30 = calc_metrics(y_true[last30], y_pred_old[last30])
        r_new_30 = calc_metrics(y_true[last30], y_pred_new[last30])
        r_v1_old_30 = calc_metrics(y_true[last30], y_pred_old_v1[last30])
        r_v1_new_30 = calc_metrics(y_true[last30], y_pred_new_v1[last30])
        print(f"\n  --- Últimos 30 días (verano - más problemático) ---")
        print(f"  {'OLD v1 (últimos 30d)':<35} {r_v1_old_30[0]:>8.2f} {r_v1_old_30[1]:>8.2f} {r_v1_old_30[2]:>8.1f} {r_v1_old_30[5]:>+8.1f} {r_v1_old_30[4]:>10.2f}")
        print(f"  {'OLD v2 (últimos 30d)':<35} {r_old_30[0]:>8.2f} {r_old_30[1]:>8.2f} {r_old_30[2]:>8.1f} {r_old_30[5]:>+8.1f} {r_old_30[4]:>10.2f}")
        print(f"  {'NEW v1 (últimos 30d)':<35} {r_v1_new_30[0]:>8.2f} {r_v1_new_30[1]:>8.2f} {r_v1_new_30[2]:>8.1f} {r_v1_new_30[5]:>+8.1f} {r_v1_new_30[4]:>10.2f}")
        print(f"  {'NEW v2 (últimos 30d)':<35} {r_new_30[0]:>8.2f} {r_new_30[1]:>8.2f} {r_new_30[2]:>8.1f} {r_new_30[5]:>+8.1f} {r_new_30[4]:>10.2f}")
        print(f"  Real media (30d): {r_old_30[3]:.2f}")

print("\n" + "=" * 75)
print("  BACKTEST COMPLETADO")
print("=" * 75)
