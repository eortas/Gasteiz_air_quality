"""
predict.py
==================
Genera predicciones d1 (mañana) para los 8 targets de calidad del aire
usando los modelos LightGBM v8 entrenados.

Cada modelo carga sus propias features seleccionadas por permutation importance
(77-80 de las 256 totales) desde los archivos lgbm_v5_*_features.json.

Fuentes de datos para la predicción:
  - data/processed/features_daily.parquet  → última fila con features conocidas
  - Open-Meteo API (opcional)              → pronóstico meteorológico real d1

Salida:
  - Tabla en consola con predicciones + intervalos de confianza aproximados
  - data/processed/predictions_latest.json → para consumo por otros módulos

Uso:
    python src/ml/predict.py
    python src/ml/predict.py --with-forecast   # usa pronóstico real Open-Meteo
    python src/ml/predict.py --date 2026-01-15 # predice para un día histórico
    python src/ml/predict.py --json            # solo salida JSON (para pipelines)
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"

DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
    "ICA_zbe_d1", "ICA_out_d1",
]

# Métricas CV v5 — usadas para construir intervalos de confianza aproximados
CV_RMSE = {
    "NO2_zbe_d1":   3.915,
    "NO2_out_d1":   4.490,
    "PM10_zbe_d1":  4.239,
    "PM10_out_d1":  4.208,
    "PM2.5_zbe_d1": 2.680,
    "PM2.5_out_d1": 3.111,
    "ICA_zbe_d1":   6.722,
    "ICA_out_d1":   7.987,
}

# Unidades y umbrales de alerta (WHO 2021 guidelines, medias diarias)
META = {
    "NO2":   {"unit": "µg/m³", "alert": 25.0,  "label": "NO₂"},
    "PM10":  {"unit": "µg/m³", "alert": 45.0,  "label": "PM10"},
    "PM2.5": {"unit": "µg/m³", "alert": 15.0,  "label": "PM2.5"},
    "ICA":   {"unit": "µg/m³", "alert": 40.0,  "label": "ICA"},
}

LATITUDE  = 42.8467
LONGITUDE = -2.6716

FORECAST_VARS = [
    "temperature_2m", "precipitation", "rain", "snowfall",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "cloud_cover", "relative_humidity_2m", "boundary_layer_height",
    "weather_code", "sunshine_duration",
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg)

def section(title):
    log(); log("═" * 65); log(f"  {title}"); log("═" * 65)


# ─── 1. CARGAR MODELOS Y FEATURES ────────────────────────────────────────────
def load_models() -> dict:
    """Carga los 8 modelos v8 y sus listas de features seleccionadas."""
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
        models[target] = {"model": model, "features": features}

    if missing:
        log(f"\n  ❌ Archivos no encontrados:")
        for m in missing:
            log(f"     {m}")
        log(f"\n  → Ejecuta primero: python src/ml/train_model_v8.py")
        sys.exit(1)

    log(f"  ✅ {len(models)} modelos cargados")
    for target, m in models.items():
        log(f"     {target:<20} — {len(m['features'])} features")

    return models


# ─── 2. CARGAR ROW DE PREDICCIÓN ─────────────────────────────────────────────
def load_prediction_row(target_date: pd.Timestamp = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Carga la fila del parquet correspondiente al día de predicción.

    El parquet tiene features construidas con datos hasta el día D,
    y el target es D+1. Para predecir mañana (D+1) usamos la última fila (D).
    Para un día histórico (--date), usamos la fila del día anterior.
    """
    if not DATASET_PATH.exists():
        log(f"  ❌ No se encuentra {DATASET_PATH}")
        log(f"  → Ejecuta primero: python src/features/build_features.py")
        sys.exit(1)

    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # ── Verificar si hay features_latest.parquet más reciente ─────────────────
    # build_features_v6.py guarda todo el dataset (con features completas pero
    # sin d2/d3 targets) antes de filtrar. Esto permite predecir mañana real.
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
                log(f"  Usando features_latest.parquet: {pred_row_date.date()} → predice {pred_date.date()}")
            else:
                row = df.iloc[[-1]]
                pred_date = last_training_date + pd.Timedelta(days=1)
        else:
            row = df.iloc[[-1]]
            pred_date = last_training_date + pd.Timedelta(days=1)
    else:
        # Predecir un día histórico: buscar la fila del día anterior
        target_date = pd.Timestamp(target_date, tz="UTC")
        source_date = target_date - pd.Timedelta(days=1)
        row = df[df["date"] == source_date]
        if row.empty:
            # Intentar con la fila más cercana anterior
            row = df[df["date"] <= source_date].iloc[[-1]]
            if row.empty:
                log(f"  ❌ No hay datos disponibles para predecir el {target_date.date()}")
                sys.exit(1)
            log(f"  ⚠️  Fecha exacta no encontrada — usando fila más cercana: {row['date'].iloc[0].date()}")
        pred_date = target_date

    source_date = row["date"].iloc[0]
    log(f"  Fila fuente   : {source_date.date()} (datos conocidos hasta este día)")
    log(f"  Predicción    : {pred_date.date()} (mañana)")
    log(f"  Features disp.: {len(df.columns)} columnas en parquet")

    return row, pred_date


# ─── 3. PRONÓSTICO OPEN-METEO (OPCIONAL) ─────────────────────────────────────
def fetch_forecast_d1() -> dict:
    """Descarga pronóstico Open-Meteo para mañana y devuelve dict de features fc_*_d1."""
    try:
        import requests
    except ImportError:
        log("  ⚠️  requests no instalado — pip install requests")
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
        "forecast_days":  2,
        "timezone":       "UTC",
        "wind_speed_unit": "ms",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  ⚠️  Error Open-Meteo: {e} — usando proxy histórico del parquet")
        return {}

    df = pd.DataFrame(data.get("hourly", {}))
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df["date"]      = df["timestamp"].dt.floor("D")

    tomorrow = pd.to_datetime(pd.Timestamp.now(tz="Europe/Madrid").date()).tz_localize("UTC") + pd.Timedelta(days=1)
    df_tomorrow = df[df["date"] == tomorrow]

    if df_tomorrow.empty:
        log("  ⚠️  Sin datos de mañana en Open-Meteo")
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

    log(f"  ✅ Pronóstico descargado: {len(agg)} features fc_*_d1 para {tomorrow.date()}")
    return agg


# ─── 4. PREDECIR ──────────────────────────────────────────────────────────────
def predict(models: dict, row: pd.DataFrame, forecast_override: dict = None) -> dict:
    """
    Genera predicciones para los 8 targets usando sus respectivas features.
    Si forecast_override tiene features fc_*_d1 reales, las sustituye en la fila.
    """
    results = {}

    # Aplicar pronóstico real si está disponible
    if forecast_override:
        for feat, val in forecast_override.items():
            if feat in row.columns:
                row = row.copy()
                row[feat] = val

    for target, m in models.items():
        model    = m["model"]
        features = m["features"]

        # Construir vector de entrada con las features del modelo
        missing_features = [f for f in features if f not in row.columns]
        if missing_features:
            log(f"  ⚠️  {target}: {len(missing_features)} features no encontradas en parquet")
            log(f"       Primeras: {missing_features[:5]}")

        X = row.reindex(columns=features, fill_value=0).fillna(0)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)  # los contaminantes no pueden ser negativos

        rmse_cv = CV_RMSE.get(target, 0)
        results[target] = {
            "prediction": round(pred, 2),
            "lower":      round(max(0, pred - 1.28 * rmse_cv), 2),  # ~90% CI
            "upper":      round(pred + 1.28 * rmse_cv, 2),
            "rmse_cv":    rmse_cv,
        }

    return results


# ─── 5. MOSTRAR RESULTADOS ────────────────────────────────────────────────────
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
            alert = "⚠️  ALERTA" if pred >= meta["alert"] else "✅  OK"
            zone_label = "ZBE" if zone == "zbe" else "OUT"
            log(f"  {meta['label']:<12} {zone_label:<6} {pred:>7.2f} {meta['unit']}  "
                f"[{lower:.1f} – {upper:.1f}]  {alert}")

    log()
    log(f"  Intervalo de confianza: ±1.28 × RMSE_CV (~90%)")
    source = "pronóstico Open-Meteo real" if with_forecast else "proxy histórico (parquet)"
    log(f"  Meteorología d1: {source}")
    log(f"  Modelos: LightGBM v8 · TimeSeriesSplit 5 folds · permutation importance")


# ─── 6. GUARDAR JSON ──────────────────────────────────────────────────────────
def save_json(results: dict, pred_date: pd.Timestamp):
    out = {
        "prediction_date": pred_date.date().isoformat(),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "model_version":   "v5",
        "targets": results,
    }
    out_path = PROCESSED_DIR / "predictions_latest.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  ✅ JSON guardado: {out_path}")
    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    with_forecast = "--with-forecast" in sys.argv
    json_only     = "--json"          in sys.argv
    date_arg      = next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                          if a == "--date" and i+1 < len(sys.argv)), None)

    if not json_only:
        log("=" * 65)
        log("  PREDICT — Vitoria Air Quality")
        log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        log(f"  Modelo: LightGBM v8 · 8 targets d1")
        log(f"  Meteo d1: {'Open-Meteo real' if with_forecast else 'proxy histórico'}")
        log("=" * 65)

    # 1. Cargar modelos
    if not json_only:
        section("1. Cargando modelos v5")
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
        forecast_override = fetch_forecast_d1()

    # 4. Predecir
    results = predict(models, row, forecast_override)

    # 5. Mostrar o volcar JSON
    if json_only:
        out = save_json(results, pred_date)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print_results(results, pred_date, with_forecast=bool(forecast_override))
        save_json(results, pred_date)

        log()
        log("=" * 65)
        log("  ✅ PREDICCIÓN COMPLETADA")
        log("=" * 65)


if __name__ == "__main__":
    main()
