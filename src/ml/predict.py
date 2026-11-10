"""
predict.py
==================
Genera predicciones D+1 y D+2 para los indicadores de calidad del aire.
Prioriza el ensemble forecast v10 y conserva v8 como compatibilidad.

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
FORECAST_HISTORY_PATH = PROCESSED_DIR / "weather_forecast_history.jsonl"
PREDICTION_CUTOFF_HOUR = 22

# ??? CONFIG ???????????????????????????????????????????????????????????????????
HORIZONS = (1, 2)
TARGETS = [
    f"{pollutant}_{zone}_d{horizon}"
    for horizon in HORIZONS
    for pollutant in ["NO2", "PM10", "PM2.5"]
    for zone in ["zbe", "out"]
]

# Función para obtener métricas CV v8 dinámicas
def get_cv_rmse() -> dict:
    """Carga el RMSE de CV real. Sin métricas no se inventan intervalos."""
    metrics_path = MODELS_DIR / "forecast_v10_metrics.json"
    if not metrics_path.exists():
        metrics_path = MODELS_DIR / "metrics_v8.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        res = {}
        for k, v in data.items():
            clean_k = k.replace("target_", "")
            rmse_value = v.get("cv_rmse")
            if rmse_value is not None:
                res[clean_k] = rmse_value
        return res
    except Exception:
        return {}

CV_RMSE = get_cv_rmse()


def get_model_metrics() -> dict:
    """Carga la decisión de producción obtenida en el bloque final reservado."""
    metrics_path = MODELS_DIR / "forecast_v10_metrics.json"
    if not metrics_path.exists():
        metrics_path = MODELS_DIR / "metrics_v8.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


MODEL_METRICS = get_model_metrics()

# Unidades y umbrales de alerta (WHO 2021 guidelines, medias diarias).
# Los umbrales se centralizan en config.py para que dashboard y modelo no divergan.
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALERT_THRESHOLDS  # noqa: E402

META = {
    "NO2":   {"unit": "ug/m3", "alert": ALERT_THRESHOLDS["NO2"],   "label": "NO?"},
    "PM10":  {"unit": "ug/m3", "alert": ALERT_THRESHOLDS["PM10"],  "label": "PM10"},
    "PM2.5": {"unit": "ug/m3", "alert": ALERT_THRESHOLDS["PM2.5"], "label": "PM2.5"},
    "ICA":   {"unit": "idx",   "alert": ALERT_THRESHOLDS["ICA"],   "label": "ICA"},
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
QUIET = False


def log(msg=""):
    if not QUIET:
        print(msg)

def section(title):
    log(); log("=" * 65); log(f"  {title}"); log("=" * 65)


def target_horizon(target: str) -> int:
    """Obtenemos el horizonte incluido en el nombre del target."""
    return int(target.rsplit("_d", 1)[1])


def current_air_column(target: str) -> str:
    """Convertimos NO2_zbe_d2 en la concentración conocida NO2_zbe."""
    return target.rsplit("_d", 1)[0]


# ??? 1. CARGAR MODELOS Y FEATURES ????????????????????????????????????????????
def load_models() -> dict:
    """Carga forecast v10 y usa v8 solo como compatibilidad."""
    models = {}
    missing = []
    v10_paths = [MODELS_DIR / f"forecast_v10_{target}.joblib" for target in TARGETS]
    use_v10 = all(path.exists() for path in v10_paths)

    for target in TARGETS:
        if use_v10:
            artifact = joblib.load(MODELS_DIR / f"forecast_v10_{target}.joblib")
            metric = MODEL_METRICS.get(f"target_{target}", {})
            models[target] = {
                "model": artifact["lightgbm"],
                "extra_model": artifact["extra_trees"],
                "ridge_model": artifact.get("ridge"),
                "features": artifact["features"],
                "medians": artifact["medians"],
                "seasonal_medians": artifact.get("seasonal_medians", {}),
                "weights": artifact["weights"],
                "persistence_weight": artifact.get("persistence_weight", 1.0),
                "production_method": artifact["production_method"],
                "prediction_mode": artifact["prediction_mode"],
                "interval_method": artifact["interval_method"],
                "interval_half_width_90": artifact["interval_half_width_90"],
                "quality_gate": artifact["quality_gate"],
                "model_version": artifact["model_version"],
                "cv_rmse": metric.get("cv_rmse", 0),
                "cv_skill_vs_persistence": metric.get("cv_skill_vs_persistence", 0),
                "test_persistence_rmse": metric.get("test_persistence_rmse"),
            }
            continue

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
            "production_method": MODEL_METRICS.get(
                f"target_{target}", {}
            ).get("production_method", "lightgbm"),
            "prediction_mode": MODEL_METRICS.get(
                f"target_{target}", {}
            ).get("prediction_mode", "level"),
            "residual_scale": MODEL_METRICS.get(
                f"target_{target}", {}
            ).get("residual_scale", 1.0),
            "test_persistence_rmse": MODEL_METRICS.get(
                f"target_{target}", {}
            ).get("test_persistence_rmse"),
            "model_version": "v8",
            "quality_gate": "legacy",
        }

    if missing:
        log(f"\n  ? Archivos no encontrados:")
        for m in missing:
            log(f"     {m}")
        log(f"\n  -> Ejecuta primero: python src/ml/train_model_v8.py")
        sys.exit(1)

    model_label = "forecast v10 (LightGBM + ExtraTrees + Ridge)" if use_v10 else "v8 (compatibilidad)"
    log(f"  [OK] {len(models)} modelos cargados: {model_label}")
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


def add_target_calendar_features(
    row: pd.DataFrame, first_prediction_date: pd.Timestamp
) -> pd.DataFrame:
    """Añadimos el calendario conocido de D+1 y D+2."""
    result = row.copy()
    first_date = pd.Timestamp(first_prediction_date)
    for horizon in HORIZONS:
        date = first_date + pd.Timedelta(days=horizon - 1)
        prefix = f"d{horizon}"
        result[f"{prefix}_day_of_week"] = date.dayofweek
        result[f"{prefix}_month"] = date.month
        result[f"{prefix}_day_of_year"] = date.dayofyear
        result[f"{prefix}_is_weekend"] = int(date.dayofweek >= 5)
        result[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * date.dayofweek / 7)
        result[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * date.dayofweek / 7)
        result[f"{prefix}_doy_sin"] = np.sin(2 * np.pi * date.dayofyear / 365.25)
        result[f"{prefix}_doy_cos"] = np.cos(2 * np.pi * date.dayofyear / 365.25)
    return result


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
    source = df
    if pred_row_path.exists():
        source = pd.read_parquet(pred_row_path)
        source["date"] = pd.to_datetime(source["date"], utc=True)
        source = source.sort_values("date").reset_index(drop=True)

    if target_date is None:
        now_local = pd.Timestamp.now(tz="Europe/Madrid")
        # Si ejecutamos a partir de las 22:00, tomamos el corte del día en curso.
        # Si ejecutamos antes de las 22:00 (por ejemplo, reintento o rescate matutino),
        # tomamos como corte el día anterior cerrado completo.
        if now_local.hour >= PREDICTION_CUTOFF_HOUR:
            latest_allowed = now_local.normalize().tz_localize(None).tz_localize("UTC")
        else:
            latest_allowed = (now_local.normalize() - pd.Timedelta(days=1)).tz_localize(None).tz_localize("UTC")
            log(f"  [INFO] Ejecutando antes de las {PREDICTION_CUTOFF_HOUR:02d}:00 locales. "
                f"Tomamos como corte el último día cerrado completo: {latest_allowed.date()}")

        required = ["NO2_zbe", "NO2_out", "PM10_zbe", "PM10_out", "PM2.5_zbe", "PM2.5_out",
                    "traffic_volume", "temperature_2m"]
        available = [column for column in required if column in source.columns]

        source_target = source[source["date"] == latest_allowed].dropna(subset=available)
        if source_target.empty:
            # Si el día exacto de corte no tiene todas las variables requeridas,
            # seleccionamos la fila más reciente con datos completos hasta esa fecha.
            valid_source = source[source["date"] <= latest_allowed].dropna(subset=available)
            if valid_source.empty:
                raise RuntimeError(
                    f"No hay registros con aire, tráfico y meteorología completos hasta {latest_allowed.date()}."
                )
            row = valid_source.iloc[[-1]]
            log(f"  [WARN] El corte de {latest_allowed.date()} no tiene variables completas. "
                f"Usamos la fila más reciente disponible: {row['date'].iloc[0].date()}")
        else:
            row = source_target.iloc[[-1]]

        # Gate de frescura: nunca publicar como "actual" una fila antigua.
        source_date = row["date"].iloc[0]
        if source_date != latest_allowed:
            message = (
                f"El corte {latest_allowed.date()} no tiene datos completos y la fila "
                f"más reciente es {source_date.date()}. Se aborta para no publicar una "
                "predicción antigua como actual. Usa --allow-stale para forzar."
            )
            if "--allow-stale" not in sys.argv:
                raise RuntimeError(message)
            log(f"  [WARN] {message}")

        pred_date = row["date"].iloc[0] + pd.Timedelta(days=1)
    else:
        # Predecir un d?a hist?rico: buscar la fila del d?a anterior
        target_date = pd.Timestamp(target_date, tz="UTC")
        source_date = target_date - pd.Timedelta(days=1)
        row = source[source["date"] == source_date]
        if row.empty:
            # Intentar con la fila m?s cercana anterior
            row = source[source["date"] <= source_date].iloc[[-1]]
            if row.empty:
                log(f"  ? No hay datos disponibles para predecir el {target_date.date()}")
                sys.exit(1)
            log(f"  [WARN]  Fecha exacta no encontrada - usando fila m?s cercana: {row['date'].iloc[0].date()}")
        pred_date = target_date

    source_date = row["date"].iloc[0]
    log(f"  Fila fuente   : {source_date.date()} (datos conocidos hasta este d?a)")
    log(f"  Predicci?n    : {pred_date.date()} (mañana desde el corte local de las {PREDICTION_CUTOFF_HOUR:02d}:00)")
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
        "forecast_days":  3,
        "timezone":       "Europe/Madrid",
        "wind_speed_unit": "ms",
        "models":          "icon_seamless",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  [WARN]  Error Open-Meteo: {e} - usando proxy hist?rico del parquet")
        return {}

    df = pd.DataFrame(data.get("hourly", {}))
    # Open-Meteo devuelve horas locales y agregamos el día civil de Vitoria.
    df["timestamp"] = pd.to_datetime(df["time"])
    df["date"] = df["timestamp"].dt.date

    tomorrow = target_date.date()
    df_tomorrow = df[df["date"] == tomorrow]

    if df_tomorrow.empty:
        log(f"  [WARN]  Sin datos de {tomorrow} en Open-Meteo")
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

    log(f"  [OK] Pron?stico descargado: {len(agg)} features fc_*_d1 para {tomorrow}")
    return agg


def save_forecast_snapshot(target_date: pd.Timestamp, forecast: dict):
    """Guarda el pronóstico emitido para poder entrenar con forecasts reales."""
    if not forecast:
        return
    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.date().isoformat(),
        "features": forecast,
    }
    with open(FORECAST_HISTORY_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_forecasts(first_target_date: pd.Timestamp) -> dict:
    """Descargamos los pronósticos D+1 y D+2 con el mismo corte operativo."""
    forecasts = {}
    for horizon in HORIZONS:
        target_date = pd.Timestamp(first_target_date) + pd.Timedelta(days=horizon - 1)
        horizon_forecast = fetch_forecast_d1(target_date)
        forecasts.update({
            key.replace("_d1", f"_d{horizon}"): value
            for key, value in horizon_forecast.items()
        })
    return forecasts


def save_forecast_snapshots(first_target_date: pd.Timestamp, forecast: dict):
    """Guardamos por separado las emisiones meteorológicas D+1 y D+2."""
    if not forecast:
        return

    generated_at = datetime.now(timezone.utc).isoformat()
    with open(FORECAST_HISTORY_PATH, "a", encoding="utf-8") as file:
        for horizon in HORIZONS:
            suffix = f"_d{horizon}"
            features = {
                key: value for key, value in forecast.items() if key.endswith(suffix)
            }
            if not features:
                continue
            target_date = pd.Timestamp(first_target_date) + pd.Timedelta(days=horizon - 1)
            record = {
                "generated_at": generated_at,
                "target_date": target_date.date().isoformat(),
                "source": "live-open-meteo",
                "lead_days": horizon,
                "features": features,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")



def generate_llm_narrative(target: str, pred_val: float, base_val: float, positive_feats: list, negative_feats: list) -> dict:
    """Generate bilingual narratives using Groq LLM based on SHAP contributions."""
    import os
    import json

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "es": f"Predicción de {pred_val:.1f} µg/m³. Factores al alza: {', '.join([f[0] for f in positive_feats[:2]])}.",
            "eu": f"{pred_val:.1f} µg/m³-ko iragarpena. Goranzko faktoreak: {', '.join([f[0] for f in positive_feats[:2]])}."
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
    
    def feature_label(name):
        return feat_map.get(name, feat_map.get(name.replace("_d2", "_d1"), name))

    pos_str = ", ".join([f"{feature_label(f[0])} (+{round(f[1], 2)})" for f in positive_feats[:3]])
    neg_str = ", ".join([f"{feature_label(f[0])} ({round(f[1], 2)})" for f in negative_feats[:3]])

    horizon = target_horizon(target)
    period_es = "mañana" if horizon == 1 else "pasado mañana"
    period_eu = "Bihar" if horizon == 1 else "Etzi"
    prompt = (
        f"Contaminante: {target.split('_')[0]} | Zona: {target.split('_')[1].upper()} | Predicción: {round(pred_val, 1)} µg/m³ | Valor base histórico: {round(base_val, 1)} µg/m³\n"
        f"Variables que AUMENTAN la concentración: {pos_str}\n"
        f"Variables que REDUCEN la concentración: {neg_str}\n\n"
        f"Genera un análisis ambiental directo y fluido para {period_es}, en 2 párrafos breves, explicando la influencia del tráfico, viento y condiciones meteorológicas.\n"
        f"Responde ÚNICAMENTE con un JSON con el formato: {{\"es\": \"texto en castellano\", \"eu\": \"texto en euskera\"}}"
    )

    try:
        import requests
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
        f"Para {period_es} se estima una concentración de {cont_name} de {round(pred_val, 1)} µg/m³ en la zona {zone_name}, "
        f"frente a la media habitual de {round(base_val, 1)} µg/m³. "
        f"Esta tendencia viene determinada principalmente por la dinámica meteorológica y el volumen de emisiones locales, "
        f"donde factores como {pos_str} actúan incrementando los niveles atmosféricos.\n\n"
        f"Por otra parte, la ventilación del valle de Vitoria y condiciones meteorológicas como {neg_str} "
        f"contribuyen a limitar la acumulación de contaminantes, manteniendo la calidad del aire en niveles aceptables dentro de la ciudad."
    )
    
    eu_text = (
        f"{period_eu} {cont_name} kontzentrazioa {round(pred_val, 1)} µg/m³-koa izatea aurreikusten da {zone_name} eremuan, "
        f"ohiko {round(base_val, 1)} µg/m³-ko batez bestekoaren aldean. "
        f"Joera hau egoera meteorologikoak eta bertako isurketek baldintzatzen dute, "
        f"bereziki {pos_str} bezalako faktoreek kontzentrazioa handituz.\n\n"
        f"Beste alde batetik, Gasteizko haraneko aireztapenak eta {neg_str} bezalako baldintzek "
        f"kutsatzaileen metaketa mugatzen laguntzen dute, airearen kalitatea maila egokietan mantenduz."
    )
    
    return {"es": es_text, "eu": eu_text}


def add_deterministic_ica(results: dict, horizon: int = 1):
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
            no2_key  = f"NO2_{zone}_d{horizon}"
            pm10_key = f"PM10_{zone}_d{horizon}"
            pm25_key = f"PM2.5_{zone}_d{horizon}"
            
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
            
            # El RMSE del ICA se deriva de sus intervalos (no de un factor 0.5
            # inventado): la incertidumbre es la del subíndice dominante.
            pollutant_results = [results[no2_key], results[pm10_key], results[pm25_key]]
            if all(r.get("lower") is not None and r.get("upper") is not None for r in pollutant_results):
                lower_ica = max(
                    compute_ica_subindex("NO2", results[no2_key]["lower"]),
                    compute_ica_subindex("PM10", results[pm10_key]["lower"]),
                    compute_ica_subindex("PM2.5", results[pm25_key]["lower"]),
                )
                upper_ica = max(
                    compute_ica_subindex("NO2", results[no2_key]["upper"]),
                    compute_ica_subindex("PM10", results[pm10_key]["upper"]),
                    compute_ica_subindex("PM2.5", results[pm25_key]["upper"]),
                )
                rmse_cv = max(upper_ica - pred_ica, pred_ica - lower_ica) / 1.645
                ica_lower = round(max(0, lower_ica), 2)
                ica_upper = round(upper_ica, 2)
                ica_interval_method = "derived_from_pollutant_intervals"
            else:
                rmse_cv = None
                ica_lower = None
                ica_upper = None
                ica_interval_method = "unavailable"
            
            results[f"ICA_{zone}_d{horizon}"] = {
                "prediction_v1": round(pred_ica_v1, 2),
                "prediction":    round(pred_ica, 2),
                "lower":         ica_lower,
                "upper":         ica_upper,
                "rmse_cv":       round(rmse_cv, 3) if rmse_cv is not None else None,
                "correction":    round(pred_ica - pred_ica_v1, 2),
                "method":        "deterministic_caqi",
                "model_version": results[no2_key].get("model_version", "v8"),
                "quality_gate":  results[no2_key].get("quality_gate", "legacy"),
                "interval_method": ica_interval_method,
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


def extra_trees_local_contributions(model, X: pd.DataFrame):
    """Descomponemos la predicción de Extra Trees siguiendo cada árbol."""
    values = X.iloc[0].to_numpy(dtype=float)
    contributions = np.zeros(len(values), dtype=float)
    base_value = 0.0

    for estimator in model.estimators_:
        tree = estimator.tree_
        node = 0
        base_value += float(tree.value[node].reshape(-1)[0])

        while tree.feature[node] >= 0:
            feature_index = int(tree.feature[node])
            if values[feature_index] <= tree.threshold[node]:
                next_node = int(tree.children_left[node])
            else:
                next_node = int(tree.children_right[node])
            parent_value = float(tree.value[node].reshape(-1)[0])
            child_value = float(tree.value[next_node].reshape(-1)[0])
            contributions[feature_index] += child_value - parent_value
            node = next_node

    n_estimators = len(model.estimators_)
    return base_value / n_estimators, contributions / n_estimators


def ridge_local_contributions(model, X: pd.DataFrame):
    """Descomponemos Ridge después de aplicar el escalado robusto entrenado."""
    scaler = model.named_steps["robustscaler"]
    ridge = model.named_steps["ridge"]
    scaled_values = scaler.transform(X)[0]
    contributions = scaled_values * np.asarray(ridge.coef_, dtype=float)
    base_value = float(np.asarray(ridge.intercept_).reshape(-1)[0])
    return base_value, contributions


# ??? 4. PREDECIR ??????????????????????????????????????????????????????????????
def predict(models: dict, row: pd.DataFrame, forecast_override: dict = None,
            pred_date: pd.Timestamp = None) -> dict:
    """
    Genera predicciones para los targets usando sus respectivas features.
    Si forecast_override tiene features meteorológicas reales, las sustituye en la fila.
    Usa medianas estacionales (del mes de pred_date) para imputar NaN.
    """
    results = {}

    # Determinar el mes de predicción para medianas estacionales
    pred_month = pred_date.month if pred_date is not None else datetime.now().month
    if pred_date is not None:
        row = add_target_calendar_features(row, pred_date)

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
        production_method = m.get("production_method", "lightgbm")
        prediction_mode = m.get("prediction_mode", "level")
        residual_scale = m.get("residual_scale", 1.0)
        weights = m.get(
            "weights", {"lightgbm": 1.0, "extra_trees": 0.0, "ridge": 0.0}
        )
        persistence_weight = m.get("persistence_weight", 1.0)
        extra_output = 0.0
        ridge_output = 0.0

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
        current_col = current_air_column(target)
        if production_method == "persistence":
            if current_col not in row.columns or pd.isna(row.iloc[0][current_col]):
                raise RuntimeError(f"Falta {current_col} para la predicción de persistencia")
            pred = float(row.iloc[0][current_col])
            foresight = {
                "method": "persistence",
                "narrative": "La persistencia superó al modelo en el bloque final reservado.",
            }
        elif production_method == "ensemble":
            if current_col not in row.columns or pd.isna(row.iloc[0][current_col]):
                raise RuntimeError(
                    f"Falta {current_col} para reconstruir la predicción del ensamblado"
                )
            model_output = float(model.predict(X)[0])
            extra_output = float(m["extra_model"].predict(X)[0])
            if m.get("ridge_model") is not None:
                ridge_output = float(m["ridge_model"].predict(X)[0])
            pred = (
                float(row.iloc[0][current_col])
                + weights["lightgbm"] * model_output
                + weights["extra_trees"] * extra_output
                + weights.get("ridge", 0.0) * ridge_output
            )
            if persistence_weight < 1.0:
                base_pred = float(row.iloc[0][current_col])
                pred = base_pred + persistence_weight * (pred - base_pred)
            foresight = None
        else:
            model_output = float(model.predict(X)[0])
            if prediction_mode == "residual":
                if current_col not in row.columns or pd.isna(row.iloc[0][current_col]):
                    raise RuntimeError(
                        f"Falta {current_col} para reconstruir la predicción residual"
                    )
                pred = float(row.iloc[0][current_col]) + residual_scale * model_output
            else:
                pred = model_output
            foresight = None
        pred = max(0.0, pred)  # los contaminantes no pueden ser negativos

        if n_nan > 0:
            log(f"  [INFO] {target}: {n_nan}/{len(features)} features imputadas con mediana mes {pred_month}")
        
        # Calcular SHAP / Feature Contributions
        if foresight is None:
            try:
                contribs = model.predict(X, pred_contrib=True)
                base_value = float(contribs[0, -1])
                if production_method == "ensemble":
                    extra_base, extra_contributions = extra_trees_local_contributions(
                        m["extra_model"], X
                    )
                    ridge_base, ridge_contributions = ridge_local_contributions(
                        m["ridge_model"], X
                    )
                    base_value = (
                        float(row.iloc[0][current_col])
                        + weights["lightgbm"] * base_value
                        + weights["extra_trees"] * extra_base
                        + weights.get("ridge", 0.0) * ridge_base
                    )
                    feature_contributions = (
                        weights["lightgbm"] * contribs[0, :-1]
                        + weights["extra_trees"] * extra_contributions
                        + weights.get("ridge", 0.0) * ridge_contributions
                    )
                    if persistence_weight < 1.0:
                        base_pred = float(row.iloc[0][current_col])
                        base_value = base_pred + persistence_weight * (base_value - base_pred)
                        feature_contributions = persistence_weight * feature_contributions
                elif prediction_mode == "residual":
                    base_value = (
                        float(row.iloc[0][current_col]) + residual_scale * base_value
                    )
                    feature_contributions = residual_scale * contribs[0, :-1]
                else:
                    feature_contributions = contribs[0, :-1]
                feats_impact = list(zip(features, feature_contributions))
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
                    "narrative": narrative,
                    "explanation_method": (
                        "weighted_additive_components"
                        if production_method == "ensemble"
                        else "lightgbm_contributions"
                    ),
                }
            except Exception as e:
                foresight = {"error": str(e)}

        rmse_cv = m.get("test_persistence_rmse") if production_method == "persistence" else m.get("cv_rmse")
        if rmse_cv is None:
            rmse_cv = CV_RMSE.get(target)
        relative_half = m.get("interval_relative_half_width_90")
        min_abs_half = m.get("interval_min_abs_90", 0.5)
        if relative_half is not None:
            interval_half_width = max(relative_half * max(abs(pred), 1.0), min_abs_half)
            interval_method = "conformal_relative_90"
        elif rmse_cv is not None and rmse_cv > 0:
            interval_half_width = m.get("interval_half_width_90", 1.645 * rmse_cv)
            interval_method = m.get("interval_method", "normal_rmse_approximation")
        else:
            # Sin métricas no publicamos una falsa incertidumbre.
            interval_half_width = None
            interval_method = "unavailable"
        results[target] = {
            "prediction": round(pred, 2),
            "lower":      round(max(0, pred - interval_half_width), 2) if interval_half_width is not None else None,
            "upper":      round(pred + interval_half_width, 2) if interval_half_width is not None else None,
            "rmse_cv":    rmse_cv,
            "method":     production_method,
            "prediction_mode": prediction_mode,
            "model_version": m.get("model_version", "v8"),
            "quality_gate": m.get("quality_gate", "legacy"),
            "skill_vs_persistence": m.get("cv_skill_vs_persistence"),
            "interval_method": interval_method,
            "foresight":  foresight
        }

    # Añadir el ICA calculado de manera determinista en base a los contaminantes base
    for horizon in HORIZONS:
        add_deterministic_ica(results, horizon)

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
            interval_txt = f"[{lower:.1f} - {upper:.1f}]" if lower is not None and upper is not None else "[N/D]"
            log(f"  {meta['label']:<12} {zone_label:<6} {pred:>7.2f} {meta['unit']}  "
                f"{interval_txt}  {alert}")

    log(f"\n  D+2 · {pred_date.date() + pd.Timedelta(days=1)}")
    for cont in ["NO2", "PM10", "PM2.5", "ICA"]:
        meta = META[cont]
        for zone in ["zbe", "out"]:
            key = f"{cont}_{zone}_d2"
            if key not in results:
                continue
            result = results[key]
            alert = "[WARN] ALERTA" if result["prediction"] >= meta["alert"] else "[OK] OK"
            zone_label = "ZBE" if zone == "zbe" else "OUT"
            interval_txt = (
                f"[{result['lower']:.1f} - {result['upper']:.1f}]"
                if result.get("lower") is not None and result.get("upper") is not None
                else "[N/D]"
            )
            log(
                f"  {meta['label']:<12} {zone_label:<6} {result['prediction']:>7.2f} {meta['unit']}  "
                f"{interval_txt}  {alert}"
            )

    log()
    interval_methods = {result.get("interval_method", "aproximación normal") for result in results.values()}
    log(f"  Intervalo 90%: {', '.join(sorted(interval_methods))}")
    methods = {result.get("method", "lightgbm") for result in results.values()}
    method_label = ", ".join(sorted(methods))
    log(f"  Método de producción: {method_label}")
    source = "pron?stico Open-Meteo real" if with_forecast else "proxy hist?rico (parquet)"
    log(f"  Meteorolog?a d1: {source}")
    log("  Selección: prueba temporal reservada + gap de un día")


# ??? 6. GUARDAR JSON ??????????????????????????????????????????????????????????
def save_json(results: dict, pred_date: pd.Timestamp, run_type: str = "production",
              source_date: pd.Timestamp = None, weather_source: str = "proxy"):
    out = {
        "prediction_date": pred_date.date().isoformat(),
        "prediction_dates": {
            f"d{horizon}": (pred_date + pd.Timedelta(days=horizon - 1)).date().isoformat()
            for horizon in HORIZONS
        },
        "forecast_horizons_days": list(HORIZONS),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "model_version":   "forecast_v10" if any(
            value.get("model_version") == "forecast_v10" for value in results.values()
        ) else "v8",
        "run_type":        run_type,
        "source_date":     source_date.date().isoformat() if source_date is not None else None,
        "weather_source":  weather_source,
        "targets": results,
    }

    if run_type == "backtest":
        backtest_dir = PROCESSED_DIR / "backtests"
        backtest_dir.mkdir(exist_ok=True)
        out_path = backtest_dir / f"prediction_{pred_date.date().isoformat()}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"\n  [OK] Backtest guardado fuera del histórico operativo: {out_path}")
        return out

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

        # Registro append-only: nunca sustituir una predicción ya emitida.
        entry = {
            "prediction_date": pred_date.date().isoformat(),
            "prediction_dates": {
                f"d{horizon}": (pred_date + pd.Timedelta(days=horizon - 1)).date().isoformat()
                for horizon in HORIZONS
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_date": source_date.date().isoformat() if source_date is not None else None,
            "run_type": "production",
            "predictions": {k: round(v["prediction"], 2) for k, v in results.items()}
        }
        history.append(entry)

        # Mantener las últimas 180 ejecuciones conservando su fecha de generación.
        history = sorted(history, key=lambda x: x.get("generated_at", ""))[-180:]

        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"  [OK] Historial actualizado: {history_path} ({len(history)} entradas)")
    except Exception as e:
        log(f"  [WARN] No se pudo actualizar el historial de predicciones: {e}")

    return out


# ??? MAIN ?????????????????????????????????????????????????????????????????????
def main():
    global QUIET
    with_forecast = "--with-forecast" in sys.argv
    json_only     = "--json"          in sys.argv
    QUIET = json_only
    publish_date  = "--publish-date"  in sys.argv
    date_arg      = next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                          if a == "--date" and i+1 < len(sys.argv)), None)

    if not json_only:
        meteo_desc = "Open-Meteo real" if with_forecast else "proxy histórico"
        log("=" * 65)
        log("  PREDICT - Vitoria Air Quality")
        log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        log("  Modelo: forecast v10 validado · horizontes D+1 y D+2")
        log(f"  Meteorología D+1/D+2: {meteo_desc}")
        log("=" * 65)

    # 1. Cargar modelos
    if not json_only:
        section("1. Cargando modelos operativos")
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
        forecast_override = fetch_forecasts(pred_date)
        if not date_arg:
            save_forecast_snapshots(pred_date, forecast_override)

    # Aplicar pronóstico real a row si está disponible
    if forecast_override:
        row = row.copy()
        for feat, val in forecast_override.items():
            row[feat] = val

    # No mezclamos el pronóstico de D+1 con variables observadas de D.
    # Las features fc_* se activarán al reentrenar con snapshots históricos.

    # 4. Predecir (con medianas estacionales)
    results = predict(models, row, None, pred_date=pred_date)

    # El meta-modelo v2 se retiró: fue entrenado con otro modelo base y sus
    # residuos no son compatibles con forecast v10.

    # 5. Mostrar o volcar JSON
    run_type = "production" if publish_date or not date_arg else "backtest"
    source_date = row["date"].iloc[0]
    if not with_forecast:
        weather_source = "proxy"
    elif forecast_override:
        weather_source = "open-meteo"
    else:
        weather_source = "fallback"
    if json_only:
        out = save_json(results, pred_date, run_type, source_date, weather_source)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print_results(results, pred_date, with_forecast=bool(forecast_override))
        save_json(results, pred_date, run_type, source_date, weather_source)

        log()
        log("=" * 65)
        log("  [OK] PREDICCI?N COMPLETADA")
        log("=" * 65)


if __name__ == "__main__":
    main()
