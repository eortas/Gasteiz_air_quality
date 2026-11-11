"""
Ejecuta el pipeline completo en orden:

  1. Ingesta        - descarga paralela de tráfico, calidad del aire y meteorología
  2. Transformación - agrega el CSV de tráfico de 15 min a 1 hora
  3. Compresión     - comprime los CSVs locales a .gz para reducir espacio
  4. Carga          - sube los últimos 90 días a Supabase (backfill)
  5. Features       - construye el dataset de features diarias (Parquet)
  6. Entrenamiento  - entrena los modelos LightGBM v8 (100 iter RandomSearch)
  7. Análisis causal- Event Study DiD + Synthetic Control (v9)
  8. Predicción     - genera predicciones d1 y guarda predictions_latest.json

Uso:
    python run_pipeline.py
    python run_pipeline.py --skip-training   # omite el entrenamiento (usa modelo existente)
    python run_pipeline.py --local-only      # sin Supabase (solo CSV local)
"""

import subprocess
import sys
import os
import gzip
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from loguru import logger

ROOT_DIR = Path(__file__).parent

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run_script(name: str, script: str, extra_args: list = None, cwd: Path = None) -> bool:
    """Ejecuta un script Python. Devuelve True si tiene éxito, False si falla."""
    cmd = [sys.executable, script] + (extra_args or [])
    logger.info(f"> {name}")
    result = subprocess.run(cmd, capture_output=False, cwd=cwd or ROOT_DIR)
    if result.returncode != 0:
        logger.error(f"[ERROR] Falló: {name}")
        return False
    logger.success(f"[OK] {name}")
    return True


def run_parallel(steps: list[tuple]) -> bool:
    """
    Ejecuta varios scripts en paralelo.
    steps: lista de (nombre, script, [args_extra])
    Devuelve True si todos tuvieron éxito.
    """
    ok = True
    with ThreadPoolExecutor(max_workers=len(steps)) as pool:
        futures = {
            pool.submit(run_script, name, script, args[0] if args else None): name
            for name, script, *args in steps
        }
        for future in as_completed(futures):
            if not future.result():
                ok = False
    return ok


def compress_csvs(data_dirs: list[Path]):
    """Comprime en .gz los CSVs que aún no estén comprimidos."""
    logger.info("> Compresión - comprimiendo CSVs locales")
    compressed = 0
    for d in data_dirs:
        for csv_file in d.glob("*.csv"):
            gz_file = csv_file.with_suffix(".csv.gz")
            # Siempre sobreescribimos para asegurar que los nuevos datos lleguen al .gz
            # que es el archivo que realmente se sube al repositorio.
            # if gz_file.exists():
            #     continue
            with open(csv_file, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            saved_mb = (csv_file.stat().st_size - gz_file.stat().st_size) / 1024 / 1024
            logger.debug(f"  {csv_file.name} -> {gz_file.name}  ({saved_mb:+.1f} MB)")
            compressed += 1
    logger.success(f"[OK] Compresión - {compressed} archivo(s) comprimidos")




def decompress_local_csvs(data_dirs: list[Path]):
    """Descomprime los .csv.gz que existan si el .csv no existe."""
    logger.info("> Descompresión - restaurando CSVs desde histórico Git")
    restored = 0
    for d in data_dirs:
        if not d.exists():
            continue
        for gz_file in d.glob("*.csv.gz"):
            csv_file = gz_file.with_suffix("") # quitar .gz
            if csv_file.exists() and csv_file.stat().st_mtime >= gz_file.stat().st_mtime:
                continue
            with gzip.open(gz_file, "rb") as f_in, open(csv_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            restored += 1
            logger.debug(f"  {gz_file.name} -> {csv_file.name}")
    if restored > 0:
        logger.success(f"[OK] Descompresión - {restored} archivo(s) restaurados")


def find_station_daily_csv() -> Path | None:
    """Localiza el CSV de datos diarios por estación para el análisis v9."""
    candidates = [
        ROOT_DIR / "data" / "processed" / "station_daily.csv",
        ROOT_DIR / "data" / "station_daily.csv",
        ROOT_DIR / "station_daily.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def write_pipeline_status():
    """Guarda un artefacto ligero con la frescura real de cada fuente.

    Evita publicar el dashboard sin saber si los datos de entrada son de hoy.
    No incluye los parquets: basta con las fechas máximas.
    """
    import pandas as pd

    processed = ROOT_DIR / "data" / "processed"
    models = ROOT_DIR / "models"
    status = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "prediction_date": None,
        "source_date": None,
        "station_daily_max": None,
        "traffic_yoy_max": None,
        "counterfactual_max": None,
        "model_trained_until": None,
        "drift_status": None,
        "drift_alerts": [],
        "qc_excluded": [],
    }

    try:
        pred = json.loads((processed / "predictions_latest.json").read_text(encoding="utf-8"))
        status["prediction_date"] = pred.get("prediction_date")
        status["source_date"] = pred.get("source_date")
        status["weather_source"] = pred.get("weather_source")
    except Exception as exc:
        logger.warning(f"[WARN]  pipeline_status: sin predictions_latest.json ({exc})")

    def _max_date_from_csv(path, column="date"):
        try:
            series = pd.read_csv(path, usecols=[column])[column]
            return pd.to_datetime(series, errors="coerce").max().date().isoformat()
        except Exception:
            return None

    station_csv = find_station_daily_csv()
    if station_csv:
        try:
            df = pd.read_csv(station_csv, index_col=0, parse_dates=True)
            status["station_daily_max"] = df.index.max().date().isoformat()
        except Exception:
            status["station_daily_max"] = _max_date_from_csv(station_csv)

    status["counterfactual_max"] = _max_date_from_csv(models / "counterfactual_gap_v8.csv")

    try:
        yoy = json.loads((processed / "peripheral_traffic_yoy.json").read_text(encoding="utf-8"))
        if yoy:
            status["traffic_yoy_max"] = yoy[-1].get("date")
    except Exception:
        pass

    try:
        manifest = json.loads((models / "forecast_v10_manifest.json").read_text(encoding="utf-8"))
        status["model_trained_until"] = manifest.get("data_end")
    except Exception:
        pass

    try:
        drift = json.loads((models / "drift_report.json").read_text(encoding="utf-8"))
        status["drift_status"] = drift.get("status")
        status["drift_alerts"] = [
            key for key, value in drift.get("targets", {}).items()
            if value.get("status") == "alert"
        ]
    except Exception:
        pass

    try:
        qc = json.loads((models / "station_qc.json").read_text(encoding="utf-8"))
        status["qc_excluded"] = qc.get("excluded", [])
        status["spillover_risk"] = [
            key for key, value in qc.get("spillover", {}).items()
            if value.get("spillover_risk")
        ]
    except Exception:
        pass

    (processed / "pipeline_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"[OK] pipeline_status.json escrito (datos hasta {status['source_date']})")


def sync_web_assets():
    """Copia a la raíz los PNG y el mapa que sirve index.html.

    build_v10 usa basePath="" y copia los plots junto al HTML generado
    (reports/html), pero la web se publica desde la raíz del repo.
    """
    dashboard_dir = ROOT_DIR / "reports" / "html"
    copied = 0
    for png in dashboard_dir.glob("*.png"):
        shutil.copy2(png, ROOT_DIR / png.name)
        copied += 1
    traffic_map = dashboard_dir / "traffic_map.html"
    if traffic_map.exists():
        shutil.copy2(traffic_map, ROOT_DIR / "traffic_map.html")
    logger.success(f"  [OK] {copied} PNG y traffic_map.html sincronizados en la raíz")


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def main():
    skip_training = "--skip-training" in sys.argv
    # Ahora el modo por defecto es local-only para evitar dependencia de Supabase
    local_only    = "--with-supabase" not in sys.argv

    ingestion_args = ["--local-only"] if local_only else []

    with_releases = "--with-releases" in sys.argv or os.getenv("GITHUB_ACTIONS") == "true"

    logger.info("=" * 60)
    logger.info("  PIPELINE VITORIA AIR QUALITY")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # ── 0. RESTORE DESDE STORAGE / GITHUB RELEASES ────────────────────────────
    if with_releases:
        logger.info("\n── FASE 0: Restore desde GitHub Releases")
        if not run_script("Download CSVs historicos", "src/ingestion/download_releases.py", []):
            logger.warning("[WARN]  Fallo la descarga desde Releases. Puede faltar historico.")
    else:
        logger.info("\n── FASE 0: Restore - OMITIDO (usando cache local)")
    
    # Restaura desde los .csv.gz que están en el repositorio (Git)
    decompress_local_csvs([
        ROOT_DIR / "data" / "raw" / "traffic",
        ROOT_DIR / "data" / "raw" / "air",
        ROOT_DIR / "data" / "raw" / "air" / "source",
        ROOT_DIR / "data" / "raw" / "weather",
    ])

    # ── 1. INGESTA (paralela) ─────────────────────────────────────────────────
    logger.info("\n── FASE 1: Ingesta")
    # Nota: download_air_quality ya no acepta --local-only (se asume siempre local)
    ok = run_parallel([
        ("Ingesta - Tráfico",         "src/ingestion/download_traffic.py",     ingestion_args),
        ("Ingesta - Calidad del aire", "src/ingestion/download_air_quality.py", []), 
        ("Ingesta - Meteorología",     "src/ingestion/download_weather.py",     ingestion_args),
        ("Ingesta - GV aire",         "src/ingestion/download_gv_air.py",       []),
    ])
    if not ok:
        logger.warning("[WARN]  Una o más fuentes de ingesta fallaron - el pipeline continúa con lo disponible.")

    # ── 2. PREPARACIÓN NO DESTRUCTIVA DEL AIRE ────────────────────────────────
    logger.info("\n── FASE 2: Preparación de datos aire")
    if not run_script("Preparar aire Kunak sin sobrescribir raw", "src/transformation/prepare_air_clean.py"):
        logger.error("[ERROR] No se pudo preparar el aire. Abortando para no predecir con datos incompletos.")
        sys.exit(1)

    # ── 3. TRANSFORMACIÓN ─────────────────────────────────────────────────────
    logger.info("\n── FASE 3: Transformación")
    current_year = str(datetime.now().year)
    if not run_script("Tráfico 15 min -> 1 hora", "src/ingestion/migrate_traffic_to_hourly.py", ["--year", current_year]):
        logger.error("[ERROR] La transformación de tráfico falló. Abortando.")
        sys.exit(1)

    # ── 3b. COMPRESIÓN ────────────────────────────────────────────────────────
    logger.info("\n── FASE 3b: Compresión")
    compress_csvs([
        ROOT_DIR / "data" / "raw" / "traffic",
        ROOT_DIR / "data" / "raw" / "air",
        ROOT_DIR / "data" / "raw" / "air" / "source",
        ROOT_DIR / "data" / "raw" / "weather",
    ])

    # ── 3c. UPLOAD A STORAGE ──────────────────────────────────────────────────
    if not local_only:
        logger.info("\n── FASE 3c: Upload de CSVs actualizados a Storage")
        if not run_script("Upload CSVs a Storage", "src/ingestion/upload_csv_storage.py", ["--force"]):
            logger.warning("[WARN]  Fallo la subida a Storage.")
    else:
        logger.info("\n── FASE 3c: Upload a Storage - OMITIDO (--local-only)")

    # ── 4. CARGA A SUPABASE ───────────────────────────────────────────────────
    if not local_only:
        logger.info("\n── FASE 4: Carga a Supabase")
        ok = run_parallel([
            ("Backfill Supabase - Tráfico",         "src/ingestion/download_traffic.py",     ["--backfill"]),
            ("Backfill Supabase - Calidad del aire", "src/ingestion/download_air_quality.py", ["--backfill"]),
            ("Backfill Supabase - Meteorología",     "src/ingestion/download_weather.py",     ["--backfill"]),
        ])
        if not ok:
            logger.warning("[WARN]  Algún backfill a Supabase falló - continuando.")
    else:
        logger.info("\n── FASE 4: Carga a Supabase - OMITIDA (--local-only)")

    # ── 5a. STATION DAILY CSV (input del QC y del análisis causal v9) ────────
    logger.info("\n── FASE 5a: Build station daily")
    if not run_script("Build station daily CSV", "src/features/build_station_daily.py"):
        logger.warning("[WARN]  build_station_daily falló - el análisis v9 puede no ejecutarse.")

    # ── 5b. CONTROL DE CALIDAD DE ESTACIONES ─────────────────────────────────
    logger.info("\n── FASE 5b: QC de estaciones (cambios de régimen y spillover)")
    if not run_script("Control de calidad de estaciones", "src/analysis/station_qc.py"):
        logger.warning("[WARN]  El QC de estaciones falló - el análisis causal usará todas las series.")

    # ── 5c. FEATURES (usa el QC para no contaminar las medias de zona) ───────
    logger.info("\n── FASE 5c: Feature engineering (corte 22:00 local)")
    if not run_script("Build features v6", "src/features/build_features_v6.py", ["--cutoff-hour", "22"]):
        logger.error("[ERROR] La construcción de features falló. Abortando.")
        sys.exit(1)

    # ── 6. ENTRENAMIENTO OPERATIVO + ANÁLISIS CAUSAL ───────
    if not skip_training:
        logger.info("\n── FASE 6: Entrenamiento forecast v10 D+1/D+2 con backtesting")
        if not run_script("Train forecast v10", "src/ml/train_forecast_v10.py"):
            logger.error("[ERROR] El entrenamiento operativo falló. Abortando.")
            sys.exit(1)
    else:
        logger.info("\n── FASE 6: Entrenamiento forecast v10 omitido")

    logger.info("\n── FASE 6a: Validación de artefactos forecast v10")
    validation_args = ["--allow-dataset-change"] if skip_training else []
    if not run_script("Validar forecast v10", "src/ml/validate_forecast_v10.py", validation_args):
        logger.error("[ERROR] Los artefactos operativos no son publicables. Abortando.")
        sys.exit(1)

    logger.info("\n── FASE 6b: Recálculo causal v8")
    if not run_script("Actualizar contrafactual v8 --skip-cv", "src/ml/train_model_v8.py", ["--skip-cv"]):
        logger.warning("[WARN] El recálculo contrafactual falló.")

    # ── 7. ANÁLISIS CAUSAL v9 (Event Study + Synthetic Control) ──────────────
    logger.info("\n── FASE 7: Análisis causal v9")
    station_csv = find_station_daily_csv()
    if station_csv:
        if not run_script(
            "Event Study DiD + Synthetic Control",
            "src/ml/train_model_v9.py",
            ["--station-data", str(station_csv)],
        ):
            logger.warning("[WARN]  El análisis causal v9 falló - continuando.")
    else:
        logger.warning("[WARN]  station_daily.csv no encontrado - omitiendo análisis v9.")
        logger.warning("    Ejecuta build_station_daily.py o coloca el CSV en data/processed/")

    logger.info("\n── FASE 8: Predicción D+1 y D+2 desde corte 22:00")
    if not run_script("Predecir D+1 y D+2", "src/ml/predict.py", ["--with-forecast", "--no-meta"]):
        logger.error("[ERROR] La predicción falló o los datos no están frescos. Abortando para no publicar una predicción antigua.")
        sys.exit(1)

    # ── 8b. META-MODELO (OMITIDO HASTA ESTABILIZACIÓN) ───────────────────────
    logger.info("\n── FASE 8b: Meta-Modelo - OMITIDO (modelo simple base)")

    # ── 8c. ANÁLISIS TRÁFICO YOY ─────────────────────────────────────────────
    logger.info("\n── FASE 8c: Análisis de tráfico YoY")
    if not run_script("Generar datos tráfico YoY", "src/analysis/generate_traffic_yoy_data.py"):
        logger.warning("[WARN]  La generación de datos de tráfico YoY falló.")

    # ── 8d. DRIFT DE LAS PREDICCIONES ────────────────────────────────────────
    logger.info("\n── FASE 8d: Monitor de drift (skill rolling vs persistencia)")
    if not run_script("Comprobar drift", "src/ml/check_drift.py"):
        logger.warning("[WARN]  El monitor de drift falló - no se actualiza drift_report.json.")

    # ── 8e. ESTADO DE FRESCURA PARA EL DASHBOARD ─────────────────────────────
    logger.info("\n── FASE 8e: Estado del pipeline")
    try:
        write_pipeline_status()
    except Exception as exc:
        logger.warning(f"[WARN]  No se pudo escribir pipeline_status.json: {exc}")

    # ── 9. DASHBOARD HTML ─────────────────────────────────────────────────────
    logger.info("\n── FASE 9: Dashboard")
    dated_name    = f"dashboard_{datetime.now().strftime('%Y-%m-%d')}.html"
    dashboard_dir = ROOT_DIR / "reports" / "html"
    # 9a. Dashboard de Calidad del Aire (v10)
    if not run_script(
        f"Build dashboard -> {dated_name}",
        str(dashboard_dir / "build_v10.py"),
        ["--output", dated_name],
        cwd=dashboard_dir,
    ):
        logger.warning("[WARN]  La generación del dashboard falló.")
    else:
        logger.info(f"  -> reports/html/{dated_name}")
        # Copiar a index.html para que la web (Vercel/Pages) se actualice
        try:
            shutil.copy2(dashboard_dir / dated_name, ROOT_DIR / "index.html")
            logger.success("  [OK] Dashboard copiado a la raíz (index.html)")
        except Exception as e:
            logger.error(f"  [ERROR] No se pudo copiar a la raíz: {e}")

    # 9b. Dashboard de Tráfico (Mapa de Sensores)
    if not run_script(
        "Build traffic map dashboard",
        str(dashboard_dir / "build_traffic_map.py"),
        cwd=dashboard_dir,
    ):
        logger.warning("[WARN]  La generación del mapa de tráfico falló.")
    else:
        logger.info("  -> reports/html/traffic_map.html")

    # 9c. Sincronizar en la raíz los assets que sirve index.html
    try:
        sync_web_assets()
    except Exception as exc:
        logger.error(f"  [ERROR] No se pudieron sincronizar los assets web: {exc}")

    logger.info("\n" + "=" * 60)
    logger.success("  PIPELINE COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
