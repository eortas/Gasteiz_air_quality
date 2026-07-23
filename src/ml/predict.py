"""
predict.py
==================
Genera predicciones d1 (ma?ana) para los 8 targets de calidad del aire
usando los modelos LightGBM v8 entrenados.

Cada modelo carga sus propias features seleccionadas por permutation importance
(77-80 de las 256 totales) desde los archivos lgbm_v5_*_features.json.

Fuentes de datos para la predicci?n:
  - data/processed/features_daily.parquet  -> ?ltima fila con features conocidas
  - Open-Meteo API (opcional)              -> pron?stico meteorol?gico real d1

Salida:
  - Tabla en consola con predicciones + intervalos de confianza aproximados
  - data/processed/predictions_latest.json -> para consumo por otros m?dulos

Uso:
    python src/ml/predict.py
    python src/ml/predict.py --with-forecast   # usa pron?stico real Open-Meteo
    python src/ml/predict.py --date 2026-01-15 # predice para un d?a hist?rico
    python src/ml/predict.py --json            # solo salida JSON (para pipelines)
"""

import sys
import json
import os
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"

# Cargar variables de entorno
env_path = ROOT_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"
STATION_DAILY_PATH = PROCESSED_DIR / "station_daily.csv"

# ??? CONFIG ???????????????????????????????????????????????????????????????????
TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
]

# Función para obtener métricas CV v8 dinámicas
def get_cv_rmse() -> dict:
    metrics_path = MODELS_DIR / "metrics_v8.json"
    defaults = {
        "NO2_zbe_d1": 3.458, "NO2_out_d1": 4.557,
        "PM10_zbe_d1": 4.758, "PM10_out_d1": 4.435,
        "PM2.5_zbe_d1": 3.151, "PM2.5_out_d1": 3.359,
        "ICA_zbe_d1": 4.758, "ICA_out_d1": 4.557,
    }
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            res = {}
            for k, v in data.items():
                clean_k = k.replace("target_", "")
                res[clean_k] = v.get("cv_rmse", defaults.get(clean_k, 4.0))
            res["ICA_zbe_d1"] = res.get("PM10_zbe_d1", 4.758)
            res["ICA_out_d1"] = res.get("NO2_out_d1", 4.557)
            return res
        except Exception:
            pass
    return defaults

CV_RMSE = get_cv_rmse()

# Unidades y umbrales de alerta (WHO 2021 guidelines, medias diarias)
META = {
    "NO2":   {"unit": "ug/m3", "alert": 25.0,  "label": "NO?"},
    "PM10":  {"unit": "ug/m3", "alert": 45.0,  "label": "PM10"},
    "PM2.5": {"unit": "ug/m3", "alert": 15.0,  "label": "PM2.5"},
    "ICA":   {"unit": "idx",   "alert": 40.0,  "label": "ICA"},
}

LATITUDE  = 42.8467
LONGITUDE = -2.6716

FORECAST_VARS = [
    "temperature_2m", "precipitation", "rain", "snowfall",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "cloud_cover", "relative_humidity_2m", "boundary_layer_height",
    "weather_code", "sunshine_duration",
]


# ??? HELPERS ??????????????????????????????????????????????????????????????????
def log(msg=""):
    print(msg)

def section(title):
    log(); log("=" * 65); log(f"  {title}"); log("=" * 65)


# ??? 1. CARGAR MODELOS Y FEATURES ????????????????????????????????????????????
def load_models() -> dict:
    """Carga los modelos v8 (LightGBM y CatBoost) y sus listas de features seleccionadas."""
    models = {}
    missing = []

    for target in TARGETS:
        model_path = MODELS_DIR / f"lgbm_v8_{target}.pkl"
        feat_path  = MODELS_DIR / f"lgbm_v8_{target}_features.json"

        if not model_path.exists():
            missing.append(str(model_path))
            continue
        if not feat_path.exists():
            missing.append(str(feat_path))
            continue

        model    = joblib.load(model_path)
        features = json.loads(feat_path.read_text(encoding="utf-8"))

        # A1: Cargar medianas globales de features guardadas durante entrenamiento
        median_path = MODELS_DIR / f"lgbm_v8_{target}_medians.json"
        medians = {}
        if median_path.exists():
            medians = json.loads(median_path.read_text(encoding="utf-8"))

        # A2: Cargar medianas ESTACIONALES (por mes) para imputación correcta
        seasonal_path = MODELS_DIR / f"lgbm_v8_{target}_medians_seasonal.json"
        seasonal_medians = {}
        if seasonal_path.exists():
            seasonal_medians = json.loads(seasonal_path.read_text(encoding="utf-8"))

        models[target] = {
            "model": model,
            "features": features,
            "medians": medians,
            "seasonal_medians": seasonal_medians,
        }

    if missing:
        log(f"\n  ? Archivos no encontrados:")
        for m in missing:
            log(f"     {m}")
        log(f"\n  -> Ejecuta primero: python src/ml/train_model_v8.py")
        sys.exit(1)

    log(f"  [OK] {len(models)} modelos cargados (LightGBM)")
    for target, m in models.items():
        medianas_status = "sí" if m.get("medians") else "no"
        seasonal_status = f"{len(m.get('seasonal_medians', {}))} meses" if m.get("seasonal_medians") else "no"
        log(f"     {target:<20} - {len(m['features'])} features, medianas: {medianas_status}, estacionales: {seasonal_status}")

    return models


def get_seasonal_fill_values(features: list, medians: dict, seasonal_medians: dict,
                              pred_month: int) -> dict:
    """
    Devuelve los valores de imputación para features NaN.
    Usa medianas del mes de predicción si están disponibles,
    con fallback a medianas globales.
    
    Esto es CRÍTICO para NO2, que tiene estacionalidad extrema:
    la mediana anual de NO2_zbe es ~10, pero en verano es ~4.
    Usar la mediana anual en verano infla las predicciones.
    """
    month_key = str(pred_month)
    month_medians = seasonal_medians.get(month_key, {})

    fill_values = {}
    for f in features:
        # Prioridad: mediana del mes > mediana global > 0
        if f in month_medians:
            fill_values[f] = month_medians[f]
        elif f in medians:
            fill_values[f] = medians[f]
        else:
            fill_values[f] = 0
    return fill_values


# ??? 2. CARGAR ROW DE PREDICCI?N ?????????????????????????????????????????????
def load_prediction_row(target_date: pd.Timestamp = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Carga la fila del parquet correspondiente al d?a de predicci?n.

    El parquet tiene features construidas con datos hasta el d?a D,
    y el target es D+1. Para predecir ma?ana (D+1) usamos la ?ltima fila (D).
    Para un d?a hist?rico (--date), usamos la fila del d?a anterior.
    """
    if not DATASET_PATH.exists():
        log(f"  ? No se encuentra {DATASET_PATH}")
        log(f"  -> Ejecuta primero: python src/features/build_features.py")
        sys.exit(1)

    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # ?? Verificar si hay features_latest.parquet m?s reciente ?????????????????
    # build_features_v6.py guarda todo el dataset (con features completas pero
    # sin d2/d3 targets) antes de filtrar. Esto permite predecir ma?ana real.
    pred_row_path = PROCESSED_DIR / "features_latest.parquet"

    if target_date is None:
        last_training_date = df["date"].iloc[-1]

        if pred_row_path.exists():
            df_pred_row = pd.read_parquet(pred_row_path)
            df_pred_row["date"] = pd.to_datetime(df_pred_row["date"], utc=True)
            pred_row_date = df_pred_row["date"].iloc[-1]

            if pred_row_date > last_training_date:
                row = df_pred_row.iloc[[-1]]
                pred_date = pred_row_date + pd.Timedelta(days=1)
                log(f"  Usando features_latest.parquet: {pred_row_date.date()} -> predice {pred_date.date()}")
            else:
                row = df.iloc[[-1]]
                pred_date = last_training_date + pd.Timedelta(days=1)
        else:
            row = df.iloc[[-1]]
            pred_date = last_training_date + pd.Timedelta(days=1)
    else:
        # Predecir un d?a hist?rico: buscar la fila del d?a anterior
        target_date = pd.Timestamp(target_date, tz="UTC")
        source_date = target_date - pd.Timedelta(days=1)
        row = df[df["date"] == source_date]
        if row.empty:
            # Intentar con la fila m?s cercana anterior
            row = df[df["date"] <= source_date].iloc[[-1]]
            if row.empty:
                log(f"  ? No hay datos disponibles para predecir el {target_date.date()}")
                sys.exit(1)
            log(f"  [WARN]  Fecha exacta no encontrada - usando fila m?s cercana: {row['date'].iloc[0].date()}")
        pred_date = target_date

    source_date = row["date"].iloc[0]
    log(f"  Fila fuente   : {source_date.date()} (datos conocidos hasta este d?a)")
    log(f"  Predicci?n    : {pred_date.date()} (ma?ana)")
    log(f"  Features disp.: {len(df.columns)} columnas en parquet")

    return row, pred_date


# ??? 3. PRON?STICO OPEN-METEO (OPCIONAL) ?????????????????????????????????????
def fetch_forecast_d1(target_date: pd.Timestamp) -> dict:
    """Descarga pron?stico Open-Meteo para el d?a de predicci?n y devuelve dict de features fc_*_d1."""
    try:
        import requests
    except ImportError:
        log("  [WARN]  requests no instalado - pip install requests")
        return {}

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":       LATITUDE,
        "longitude":      LONGITUDE,
        "hourly":         ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "rain", "snowfall", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", "cloud_cover", "boundary_layer_height",
            "sunshine_duration", "weather_code",
        ]),
        "forecast_days":  3,  # Aumentado a 3 para evitar problemas de desfase horario (timezone/UTC) en el runner de GitHub Actions
        "timezone":       "UTC",
        "wind_speed_unit": "ms",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  [WARN]  Error Open-Meteo: {e} - usando proxy hist?rico del parquet")
        return {}

    df = pd.DataFrame(data.get("hourly", {}))
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df["date"]      = df["timestamp"].dt.floor("D")

    # Alinear la fecha objetivo en UTC (normalizada a las 00:00:00 UTC)
    tomorrow = target_date.normalize()
    df_tomorrow = df[df["date"] == tomorrow]

    if df_tomorrow.empty:
        log(f"  [WARN]  Sin datos de {tomorrow.date()} en Open-Meteo")
        return {}

    agg = {}
    for col in FORECAST_VARS:
        if col not in df_tomorrow.columns:
            continue
        if col in ["precipitation", "rain", "snowfall", "sunshine_duration"]:
            agg[f"fc_{col}_d1"] = df_tomorrow[col].sum()
        elif col == "wind_direction_10m":
            rad = np.deg2rad(df_tomorrow[col])
            direction = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360
            agg[f"fc_{col}_d1"] = direction
        else:
            agg[f"fc_{col}_d1"] = df_tomorrow[col].mean()

    # Componentes del viento
    if "fc_wind_speed_10m_d1" in agg and "fc_wind_direction_10m_d1" in agg:
        rad = np.deg2rad(agg["fc_wind_direction_10m_d1"])
        agg["fc_wind_u_d1"] = -agg["fc_wind_speed_10m_d1"] * np.sin(rad)
        agg["fc_wind_v_d1"] = -agg["fc_wind_speed_10m_d1"] * np.cos(rad)

    # Punto de rocío del pronóstico
    temp_tomorrow = df_tomorrow["temperature_2m"].mean()
    rh_tomorrow = df_tomorrow["relative_humidity_2m"].mean()
    agg["fc_dew_point_d1"] = temp_tomorrow - ((100.0 - rh_tomorrow) / 5.0)

    # HDD del pronóstico
    agg["fc_HDD_d1"] = max(0.0, 15.0 - temp_tomorrow) if pd.notna(temp_tomorrow) else 0.0

    # Índice de ventilación del pronóstico
    if "boundary_layer_height" in df_tomorrow.columns and "wind_speed_10m" in df_tomorrow.columns:
        agg["fc_ventilation_index_d1"] = df_tomorrow["boundary_layer_height"].mean() * df_tomorrow["wind_speed_10m"].mean()

    # Lluvia acumulada últimos 3 y 7 días (Scavenging Effect)
    try:
        if DATASET_PATH.exists():
            df_hist = pd.read_parquet(DATASET_PATH)
            df_hist["date"] = pd.to_datetime(df_hist["date"], utc=True)
            last_precip = df_hist.sort_values("date")["precipitation"].tail(6).values.tolist()
            precip_tomorrow = agg.get("fc_precipitation_d1", 0.0)
            
            # 3d: tomorrow + last 2 days of history
            agg["fc_precipitation_acum_3d_d1"] = precip_tomorrow + sum(last_precip[-2:]) if len(last_precip) >= 2 else precip_tomorrow
            # 7d: tomorrow + last 6 days of history
            agg["fc_precipitation_acum_7d_d1"] = precip_tomorrow + sum(last_precip) if len(last_precip) >= 6 else precip_tomorrow
    except Exception as e:
        log(f"  [WARN] Falló cálculo de lluvia acumulada pronosticada: {e}")

    log(f"  [OK] Pron?stico descargado: {len(agg)} features fc_*_d1 para {tomorrow.date()}")
    return agg


def refine_with_meta_models(results: dict, row: pd.DataFrame, df_history: pd.DataFrame, pred_date: pd.Timestamp) -> dict:
    """
    Usa los meta-modelos Ridge para corregir las predicciones v1.
    Calcula error_lag_1d y error_roll_mean_7d usando el histórico reciente.
    
    FIX v8.1: Clamp del meta-modelo para NO2 en períodos de valores bajos.
    El meta-modelo Ridge tiene un intercepto alto (~7.6 para NO2_zbe) que
    infla las predicciones cuando los valores reales son muy bajos (verano).
    """
    refined_results = {}
    
    # 1. Calcular errores recientes para las features del meta-modelo
    cutoff_date = pred_date - pd.Timedelta(days=2)
    df_history_clean = df_history[df_history["date"] <= cutoff_date].sort_values("date").reset_index(drop=True)
    history_8d = df_history_clean.tail(8).copy()
    
    # Calcular medias históricas de clima para imputación robusta
    # Usar medias del mes actual para coherencia estacional
    pred_month = pred_date.month
    month_mask = df_history["date"].dt.month == pred_month
    df_month = df_history[month_mask] if month_mask.any() else df_history
    
    default_temp = df_month["temperature_2m"].mean() if "temperature_2m" in df_month.columns else 12.0
    default_wind = df_month["wind_speed_10m"].mean() if "wind_speed_10m" in df_month.columns else 2.5
    default_boundary = df_month["boundary_layer_height"].mean() if "boundary_layer_height" in df_month.columns else 500.0
    default_humidity = df_month["relative_humidity_2m"].mean() if "relative_humidity_2m" in df_month.columns else 75.0
    
    # Asegurar que no son NaN
    default_temp = default_temp if pd.notna(default_temp) else 12.0
    default_wind = default_wind if pd.notna(default_wind) else 2.5
    default_boundary = default_boundary if pd.notna(default_boundary) else 500.0
    default_humidity = default_humidity if pd.notna(default_humidity) else 75.0
    
    for target, r in results.items():
        model_meta_path = MODELS_DIR / f"meta_model_{target}.pkl"
        if not model_meta_path.exists():
            refined_results[target] = r
            continue
            
        try:
            meta_model = joblib.load(model_meta_path)
            
            # Cargar el modelo 1 original para calcular errores pasados
            m1_path = MODELS_DIR / f"lgbm_v8_{target}.pkl"
            f1_path = MODELS_DIR / f"lgbm_v8_{target}_features.json"
            model_v1 = joblib.load(m1_path)
            feats_v1 = json.loads(f1_path.read_text(encoding="utf-8"))
            
            # Cargar medianas de features para el modelo v1
            med1_path = MODELS_DIR / f"lgbm_v8_{target}_medians.json"
            medians_v1 = {}
            if med1_path.exists():
                try:
                    medians_v1 = json.loads(med1_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            fill_values = {f: medians_v1.get(f, 0.0) for f in feats_v1}
            
            # Calcular errores de los últimos 7 días (si hay datos)
            errors = []
            for _, h_row in history_8d.iterrows():
                X_h = h_row.to_frame().T.reindex(columns=feats_v1).fillna(fill_values).astype(float)
                p_h = float(model_v1.predict(X_h)[0])
                
                # target es "NO2_zbe_d1" -> la columna real en el parquet es "NO2_zbe"
                contam_col = target.replace("_d1", "")
                actual = h_row.get(contam_col) 
                if pd.notna(actual):
                    errors.append(actual - p_h)
            
            error_lag_1d = errors[-1] if len(errors) >= 1 else 0
            error_roll_7d = np.mean(errors) if len(errors) >= 1 else 0
            
            # Leer valores de clima del pronóstico o de la fila
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
                "temperature_2m": float(pd.to_numeric(pd.Series([temp_val]), errors='coerce').fillna(default_temp).iloc[0]),
                "wind_speed_10m": float(pd.to_numeric(pd.Series([wind_val]), errors='coerce').fillna(default_wind).iloc[0]),
                "boundary_layer_height": float(pd.to_numeric(pd.Series([boundary_val]), errors='coerce').fillna(default_boundary).iloc[0]),
                "relative_humidity_2m": float(pd.to_numeric(pd.Series([humidity_val]), errors='coerce').fillna(default_humidity).iloc[0]),
                "is_weekend": float(pd.to_numeric(row["is_weekend"], errors='coerce').fillna(0).iloc[0]),
                "es_domingo": float(pd.to_numeric(row["es_domingo"], errors='coerce').fillna(0).iloc[0]),
                "es_invierno_estricto": float(pd.to_numeric(row["es_invierno_estricto"], errors='coerce').fillna(0).iloc[0]),
            }
            
            X_meta = pd.DataFrame([meta_input]).astype(float)
            pred_v2 = float(meta_model.predict(X_meta)[0])
            pred_v2 = max(0.0, pred_v2)
            
            # Fix auditoría #4b: Clamp simple (±30%) sin bypass frágil
            # El bypass con medianas hardcodeadas fue eliminado porque:
            # - train_meta_model.py ahora NO guarda modelos que no mejoran
            # - Si llegamos aquí, el meta-modelo fue validado con walk-forward
            pred_v1 = float(r["prediction"])
            max_correction_pct = 0.30
            
            if pred_v1 > 0:
                lower_bound = pred_v1 * (1 - max_correction_pct)
                upper_bound = pred_v1 * (1 + max_correction_pct)
                pred_v2_clamped = max(0.0, min(pred_v2, upper_bound))
                pred_v2_clamped = max(lower_bound, pred_v2_clamped)
                
                if abs(pred_v2 - pred_v2_clamped) > 0.01:
                    log(f"  [CLAMP] {target}: meta={pred_v2:.2f} -> clamped={pred_v2_clamped:.2f} "
                        f"(v1={pred_v1:.2f}, rango=[{lower_bound:.2f}, {upper_bound:.2f}])")
                    pred_v2 = pred_v2_clamped
            
            refined_results[target] = {
                "prediction_v1": r["prediction"],
                "prediction":    round(pred_v2, 2),
                "lower":         round(max(0, pred_v2 - 1.28 * r["rmse_cv"]), 2),
                "upper":         round(pred_v2 + 1.28 * r["rmse_cv"], 2),
                "rmse_cv":       r["rmse_cv"],
                "correction":    round(pred_v2 - r["prediction"], 2)
            }
            if "foresight" in r:
                refined_results[target]["foresight"] = r["foresight"]
        except Exception as e:
            log(f"  [WARN] Fallo en meta-modelo {target}: {e}")
            refined_results[target] = r
            
    # Recalcular ICA determinista basándose en las predicciones refinadas
    add_deterministic_ica(refined_results)
    return refined_results


def generate_llm_narrative(target: str, pred_val: float, base_val: float, positive_feats: list, negative_feats: list) -> dict:
    """Generate bilingual narratives using Groq LLM based on SHAP contributions."""
    import os
    import requests
    import json

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "es": f"Predicción de {pred_val} µg/m³. Subidas por: {', '.join([f[0] for f in positive_feats[:2]])}.",
            "eu": f"{pred_val} µg/m³-ko iragarpena. Igoerak: {', '.join([f[0] for f in positive_feats[:2]])}."
        }

    feat_map = {
        "fc_temperature_2m_d1": "Temperatura alta",
        "fc_wind_speed_10m_d1": "Velocidad del viento",
        "fc_wind_gusts_10m_d1": "Ráfagas de viento",
        "traffic_volume_lag_1d": "Tráfico reciente",
        "NO2_zbe_roll_mean_7d": "Acumulación reciente de NO2",
        "PM10_zbe_roll_mean_14d": "Acumulación reciente de PM10",
        "is_weekend": "Día del fin de semana",
        "es_domingo": "Domingo",
        "fc_precipitation_d1": "Precipitación prevista",
    }
    
    pos_str = ", ".join([f"{feat_map.get(f[0], f[0])} (+{round(f[1], 2)})" for f in positive_feats[:3]])
    neg_str = ", ".join([f"{feat_map.get(f[0], f[0])} ({round(f[1], 2)})" for f in negative_feats[:3]])

    prompt = (
        f"Contaminante: {target.split('_')[0]} | Zona: {target.split('_')[1].upper()} | Predicción: {round(pred_val, 1)} µg/m³ | Valor base histórico: {round(base_val, 1)} µg/m³\n"
        f"Variables que AUMENTAN la concentración: {pos_str}\n"
        f"Variables que REDUCEN la concentración: {neg_str}\n\n"
        f"Genera un análisis ambiental directo y fluido en 2 párrafos breves explicando la influencia del tráfico, viento y condiciones meteorológicas.\n"
        f"Responde ÚNICAMENTE con un JSON con el formato: {{\"es\": \"texto en castellano\", \"eu\": \"texto en euskera\"}}"
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwen3.6-27b",
                "messages": [
                    {"role": "system", "content": "Eres un meteorólogo y analista ambiental de Vitoria-Gasteiz. Tu tarea es generar explicaciones en lenguaje natural claro y conciso sobre la calidad del aire. No incluyas explicaciones de razonamiento ni etiquetas <think>. Devuelve únicamente el objeto JSON bilingüe ('es' y 'eu')."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            },
            timeout=25
        )
        if response.status_code == 200:
            res_text = response.json()["choices"][0]["message"]["content"]
            import re
            cleaned_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL).strip()
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
            parsed = json.loads(cleaned_text)
            if "es" in parsed and "eu" in parsed and len(parsed["es"]) > 30:
                return parsed
    except Exception as e:
        log(f"  [INFO] Groq API fallback activado: {e}")

    # Fallback enriquecido en lenguaje natural de 2 párrafos si no responde la API
    cont_name = target.split('_')[0]
    zone_name = target.split('_')[1].upper()
    
    es_text = (
        f"Para mañana se estima una concentración de {cont_name} de {round(pred_val, 1)} µg/m³ en la zona {zone_name}, "
        f"frente a la media habitual de {round(base_val, 1)} µg/m³. "
        f"Esta tendencia viene determinada principalmente por la dinámica meteorológica y el volumen de emisiones locales, "
        f"donde factores como {pos_str} actúan incrementando los niveles atmosféricos.\n\n"
        f"Por otra parte, la ventilación del valle de Vitoria y condiciones meteorológicas como {neg_str} "
        f"contribuyen a limitar la acumulación de contaminantes, manteniendo la calidad del aire en niveles aceptables dentro de la ciudad."
    )
    
    eu_text = (
        f"Biharko {cont_name} kontzentrazioa {round(pred_val, 1)} µg/m³-koa izatea aurreikusten da {zone_name} eremuan, "
        f"ohiko {round(base_val, 1)} µg/m³-ko batez bestekoaren aldean. "
        f"Joera hau egoera meteorologikoak eta bertako isurketek baldintzatzen dute, "
        f"bereziki {pos_str} bezalako faktoreek kontzentrazioa handituz.\n\n"
        f"Beste alde batetik, Gasteizko haraneko aireztapenak eta {neg_str} bezalako baldintzek "
        f"kutsatzaileen metaketa mugatzen laguntzen dute, airearen kalitatea maila egokietan mantenduz."
    )
    
    return {"es": es_text, "eu": eu_text}


def add_deterministic_ica(results: dict):
    """Calcula el ICA con subíndices CAQI europeos normalizados.
    
    Fix auditoría #10: Reemplaza el Ridge entrenado al vuelo (sin validación)
    por un cálculo determinista con umbrales estándar europeos (CAQI).
    """
    # Importar umbrales desde config central
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import compute_ica_subindex
    
    try:
        for zone in ["zbe", "out"]:
            no2_key  = f"NO2_{zone}_d1"
            pm10_key = f"PM10_{zone}_d1"
            pm25_key = f"PM2.5_{zone}_d1"
            
            if no2_key not in results or pm10_key not in results or pm25_key not in results:
                continue
            
            pred_no2  = results[no2_key]["prediction"]
            pred_pm10 = results[pm10_key]["prediction"]
            pred_pm25 = results[pm25_key]["prediction"]
            
            # Calcular subíndices CAQI normalizados
            sub_no2  = compute_ica_subindex("NO2",   pred_no2)
            sub_pm10 = compute_ica_subindex("PM10",  pred_pm10)
            sub_pm25 = compute_ica_subindex("PM2.5", pred_pm25)
            
            # El ICA es el máximo de los subíndices (estándar CAQI)
            pred_ica = max(sub_no2, sub_pm10, sub_pm25)
            
            # También calcular para v1
            pred_no2_v1  = results[no2_key].get("prediction_v1", pred_no2)
            pred_pm10_v1 = results[pm10_key].get("prediction_v1", pred_pm10)
            pred_pm25_v1 = results[pm25_key].get("prediction_v1", pred_pm25)
            
            sub_no2_v1  = compute_ica_subindex("NO2",   pred_no2_v1)
            sub_pm10_v1 = compute_ica_subindex("PM10",  pred_pm10_v1)
            sub_pm25_v1 = compute_ica_subindex("PM2.5", pred_pm25_v1)
            pred_ica_v1 = max(sub_no2_v1, sub_pm10_v1, sub_pm25_v1)
            
            # RMSE aproximado del ICA (propagación de errores)
            rmse_cv = max(
                results[no2_key].get("rmse_cv", 5.0),
                results[pm10_key].get("rmse_cv", 5.0),
                results[pm25_key].get("rmse_cv", 5.0)
            )
            
            results[f"ICA_{zone}_d1"] = {
                "prediction_v1": round(pred_ica_v1, 2),
                "prediction":    round(pred_ica, 2),
                "lower":         round(max(0, pred_ica - 1.28 * rmse_cv * 0.5), 2),
                "upper":         round(pred_ica + 1.28 * rmse_cv * 0.5, 2),
                "rmse_cv":       round(rmse_cv * 0.5, 3),
                "correction":    round(pred_ica - pred_ica_v1, 2),
                "foresight": {
                    "base_value": round(pred_ica, 2),
                    "positive_top": [],
                    "negative_top": [],
                    "narrative": {
                        "es": f"ICA calculado con subíndices CAQI: NO2={sub_no2:.0f}, PM10={sub_pm10:.0f}, PM2.5={sub_pm25:.0f}. "
                              f"Concentraciones: NO2={pred_no2:.1f}, PM10={pred_pm10:.1f}, PM2.5={pred_pm25:.1f} µg/m³.",
                        "eu": f"ICA CAQI azpi-indizeekin: NO2={sub_no2:.0f}, PM10={sub_pm10:.0f}, PM2.5={sub_pm25:.0f}."
                    }
                }
            }
    except Exception as e:
        log(f"  [WARN] Falló el cálculo del ICA determinista: {e}")


# ??? 4. PREDECIR ??????????????????????????????????????????????????????????????
def predict(models: dict, row: pd.DataFrame, forecast_override: dict = None,
            pred_date: pd.Timestamp = None) -> dict:
    """
    Genera predicciones para los targets usando sus respectivas features.
    Si forecast_override tiene features fc_*_d1 reales, las sustituye en la fila.
    Usa medianas estacionales (del mes de pred_date) para imputar NaN.
    """
    results = {}

    # Determinar el mes de predicción para medianas estacionales
    pred_month = pred_date.month if pred_date is not None else datetime.now().month

    # Aplicar pronóstico real si está disponible
    if forecast_override:
        row = row.copy()
        for feat, val in forecast_override.items():
            row[feat] = val

    for target, m in models.items():
        model    = m["model"]
        features = m["features"]
        medians  = m.get("medians", {})
        seasonal_medians = m.get("seasonal_medians", {})

        # Construir vector de entrada con las features del modelo
        missing_features = [f for f in features if f not in row.columns]
        if missing_features:
            log(f"  [WARN]  {target}: {len(missing_features)} features no encontradas en parquet")
            log(f"       Primeras: {missing_features[:5]}")

        # A2: Usar medianas ESTACIONALES del mes actual (no anuales)
        # Esto es crítico para NO2 que tiene estacionalidad 2x
        fill_values = get_seasonal_fill_values(features, medians, seasonal_medians, pred_month)

        # Contar features imputadas para diagnóstico
        X_raw = row.reindex(columns=features)
        n_nan = int(X_raw.isna().sum(axis=1).iloc[0])
        X = X_raw.fillna(fill_values).astype(float)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)  # los contaminantes no pueden ser negativos

        if n_nan > 0:
            log(f"  [INFO] {target}: {n_nan}/{len(features)} features imputadas con mediana mes {pred_month}")
        
        # Calcular SHAP / Feature Contributions
        try:
            contribs = model.predict(X, pred_contrib=True)
            base_value = float(contribs[0, -1])
            feats_impact = list(zip(features, contribs[0, :-1]))
            # Ordenar por magnitud de impacto
            feats_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Asegurarnos de tener al menos las top features para el gráfico
            positive_feats = [f for f in feats_impact if f[1] >= 0]
            negative_feats = [f for f in feats_impact if f[1] < 0]
            
            # Si no hay ninguna negativa, incluimos las más cercanas a cero
            if not negative_feats:
                negative_feats = [f for f in feats_impact if f[1] <= 0]
            
            # Generar narrativa para todos los targets
            narrative = generate_llm_narrative(target, pred, base_value, positive_feats, negative_feats)
                
            foresight = {
                "base_value": round(float(base_value), 2),
                "positive_top": [{"feature": str(f[0]), "value": round(float(f[1]), 2)} for f in positive_feats[:5]],
                "negative_top": [{"feature": str(f[0]), "value": round(float(f[1]), 2)} for f in negative_feats[:5]],
                "narrative": narrative
            }
        except Exception as e:
            foresight = {"error": str(e)}

        rmse_cv = CV_RMSE.get(target, 0)
        results[target] = {
            "prediction": round(pred, 2),
            "lower":      round(max(0, pred - 1.28 * rmse_cv), 2),  # ~90% CI
            "upper":      round(pred + 1.28 * rmse_cv, 2),
            "rmse_cv":    rmse_cv,
            "foresight":  foresight
        }

    # Añadir el ICA calculado de manera determinista en base a los contaminantes base
    add_deterministic_ica(results)

    return results


# ??? 5. MOSTRAR RESULTADOS ????????????????????????????????????????????????????
def print_results(results: dict, pred_date: pd.Timestamp, with_forecast: bool):
    section(f"Predicciones para {pred_date.date()}")

    log(f"  {'Contaminante':<12} {'Zona':<6} {'Pred':>8} {'IC 90%':>18}  {'Alerta'}")
    log(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*18}  {'-'*8}")

    for cont in ["NO2", "PM10", "PM2.5", "ICA"]:
        meta = META[cont]
        log()
        for zone in ["zbe", "out"]:
            key = f"{cont}_{zone}_d1"
            if key not in results:
                continue
            r = results[key]
            pred  = r["prediction"]
            lower = r["lower"]
            upper = r["upper"]
            alert = "[WARN]  ALERTA" if pred >= meta["alert"] else "[OK]  OK"
            zone_label = "ZBE" if zone == "zbe" else "OUT"
            log(f"  {meta['label']:<12} {zone_label:<6} {pred:>7.2f} {meta['unit']}  "
                f"[{lower:.1f} - {upper:.1f}]  {alert}")

    log()
    log(f"  Intervalo de confianza: ?1.28 ? RMSE_CV (~90%)")
    log(f"  Correcci?n: Aplicado Meta-Modelo Ridge (Error Correction)")
    source = "pron?stico Open-Meteo real" if with_forecast else "proxy hist?rico (parquet)"
    log(f"  Meteorolog?a d1: {source}")
    log(f"  Modelos: LightGBM v8 + Ridge Meta-Model ? TimeSeriesSplit 5 folds")


# ??? 6. GUARDAR JSON ??????????????????????????????????????????????????????????
def save_json(results: dict, pred_date: pd.Timestamp):
    out = {
        "prediction_date": pred_date.date().isoformat(),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "model_version":   "v8",
        "targets": results,
    }
    out_path = PROCESSED_DIR / "predictions_latest.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  [OK] JSON guardado: {out_path}")

    # --- Persistir historial de predicciones para el dashboard ---
    # Cada ejecución añade una entrada con las predicciones finales del día.
    # El dashboard usará este historial en lugar de recalcular el backtest.
    history_path = PROCESSED_DIR / "predictions_history.json"
    try:
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            history = []

        # Crear entrada con la fecha y las predicciones finales
        entry = {
            "prediction_date": pred_date.date().isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": {k: round(v["prediction"], 2) for k, v in results.items()}
        }

        # Evitar duplicados: si ya existe una entrada para esa fecha, la reemplazamos
        history = [h for h in history if h["prediction_date"] != entry["prediction_date"]]
        history.append(entry)

        # Mantener máximo 90 días, ordenados por fecha
        history = sorted(history, key=lambda x: x["prediction_date"])[-90:]

        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"  [OK] Historial actualizado: {history_path} ({len(history)} entradas)")
    except Exception as e:
        log(f"  [WARN] No se pudo actualizar el historial de predicciones: {e}")

    return out


# ??? MAIN ?????????????????????????????????????????????????????????????????????
def main():
    with_forecast = "--with-forecast" in sys.argv
    json_only     = "--json"          in sys.argv
    date_arg      = next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                          if a == "--date" and i+1 < len(sys.argv)), None)

    if not json_only:
        meteo_desc = "Open-Meteo real" if with_forecast else "proxy histórico"
        log("=" * 65)
        log("  PREDICT - Vitoria Air Quality")
        log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        log(f"  Modelo: LightGBM v8 \u00d7 8 targets d1")
        log(f"  Meteo d1: {meteo_desc}")
        log("=" * 65)

    # 1. Cargar modelos
    if not json_only:
        section("1. Cargando modelos v8")
    models = load_models()

    # 2. Fila de predicción
    target_date = pd.Timestamp(date_arg) if date_arg else None
    if not json_only:
        section("2. Preparando fila de predicción")
    row, pred_date = load_prediction_row(target_date)

    # 3. Pronóstico meteorológico (opcional)
    forecast_override = {}
    if with_forecast:
        if not json_only:
            section("3. Descargando pronóstico Open-Meteo")
        forecast_override = fetch_forecast_d1(pred_date)

    # Aplicar pronóstico real a row si está disponible
    if forecast_override:
        row = row.copy()
        for feat, val in forecast_override.items():
            row[feat] = val

    # FIX v8.1: Propagar variables fc_* del pronóstico a variables base cuando son NaN
    # Esto es CRÍTICO: sin esto, variables como temperature_2m, HDD, wind_speed_10m
    # quedan NaN en la fila de producción y se imputan con medianas que sesgan NO2.
    fc_to_base_map = {
        "fc_temperature_2m_d1": "temperature_2m",
        "fc_wind_speed_10m_d1": "wind_speed_10m",
        "fc_wind_gusts_10m_d1": "wind_gusts_10m",
        "fc_wind_direction_10m_d1": "wind_direction_10m",
        "fc_boundary_layer_height_d1": "boundary_layer_height",
        "fc_cloud_cover_d1": "cloud_cover",
        "fc_relative_humidity_2m_d1": "relative_humidity_2m",
        "fc_precipitation_d1": "precipitation",
        "fc_rain_d1": "rain",
        "fc_sunshine_duration_d1": "sunshine_duration",
        "fc_weather_code_d1": "weather_code",
        "fc_dew_point_d1": "dew_point",
        "fc_ventilation_index_d1": "ventilation_index",
        "fc_HDD_d1": "HDD",
        "fc_wind_u_d1": "wind_u",
        "fc_wind_v_d1": "wind_v",
    }
    propagated = 0
    for fc_col, base_col in fc_to_base_map.items():
        if base_col in row.columns and fc_col in row.columns:
            if pd.isna(row[base_col].iloc[0]) and pd.notna(row[fc_col].iloc[0]):
                row = row.copy() if propagated == 0 else row
                row[base_col] = row[fc_col].iloc[0]
                propagated += 1
    if propagated > 0 and not json_only:
        log(f"  [FIX] Propagadas {propagated} variables fc_*_d1 -> base (relleno de NaN)")

    # 4. Predecir (con medianas estacionales)
    results = predict(models, row, None, pred_date=pred_date)

    # 4b. Refinar con Meta-Modelos
    if "--no-meta" not in sys.argv:
        if not json_only:
            section("4. Refinando con Meta-Modelos (v2) + Clamp v8.1")
        df_history = pd.read_parquet(DATASET_PATH)
        df_history["date"] = pd.to_datetime(df_history["date"], utc=True)
        results = refine_with_meta_models(results, row, df_history, pred_date)

    # 5. Mostrar o volcar JSON
    if json_only:
        out = save_json(results, pred_date)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print_results(results, pred_date, with_forecast=bool(forecast_override))
        save_json(results, pred_date)

        log()
        log("=" * 65)
        log("  [OK] PREDICCI?N COMPLETADA")
        log("=" * 65)


if __name__ == "__main__":
    main()
