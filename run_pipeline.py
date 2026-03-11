"""
Ejecuta el pipeline completo en orden:

  1. Ingesta        — descarga paralela de tráfico, calidad del aire y meteorología
  2. Transformación — agrega el CSV de tráfico de 15 min a 1 hora
  3. Compresión     — comprime los CSVs locales a .gz para reducir espacio
  4. Carga          — sube los últimos 90 días a Supabase (backfill)
  5. Features       — construye el dataset de features diarias (Parquet)
  6. Entrenamiento  — entrena los modelos LightGBM v8 (100 iter RandomSearch)
  7. Análisis causal— Event Study DiD + Synthetic Control (v9)
  8. Predicción     — genera predicciones d1 y guarda predictions_latest.json

Uso:
    python run_pipeline.py
    python run_pipeline.py --skip-training   # omite el entrenamiento (usa modelo existente)
    python run_pipeline.py --local-only      # sin Supabase (solo CSV local)
"""

import subprocess
import sys
import os
import gzip
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
    logger.info(f"▶ {name}")
    result = subprocess.run(cmd, capture_output=False, cwd=cwd or ROOT_DIR)
    if result.returncode != 0:
        logger.error(f"✗ Falló: {name}")
        return False
    logger.success(f"✓ {name}")
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
    logger.info("▶ Compresión — comprimiendo CSVs locales")
    compressed = 0
    for d in data_dirs:
        for csv_file in d.glob("*.csv"):
            gz_file = csv_file.with_suffix(".csv.gz")
            if gz_file.exists():
                continue
            with open(csv_file, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            saved_mb = (csv_file.stat().st_size - gz_file.stat().st_size) / 1024 / 1024
            logger.debug(f"  {csv_file.name} → {gz_file.name}  ({saved_mb:+.1f} MB)")
            compressed += 1
    logger.success(f"✓ Compresión — {compressed} archivo(s) comprimidos")


def find_station_daily_csv() -> Path | None:
    """Localiza el CSV de datos diarios por estación para el análisis v9."""
    candidates = [
        ROOT_DIR / "data" / "processed" / "station_daily.csv",
        ROOT_DIR / "data" / "station_daily.csv",
        ROOT_DIR / "station_daily.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def main():
    skip_training = "--skip-training" in sys.argv
    local_only    = "--local-only"    in sys.argv

    ingestion_args = ["--local-only"] if local_only else []

    logger.info("=" * 60)
    logger.info("  PIPELINE VITORIA AIR QUALITY")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # ── 0. RESTORE DESDE STORAGE ──────────────────────────────────────────────
    if not local_only:
        logger.info("\n── FASE 0: Restore desde Supabase Storage")
        if not run_script("Download CSVs historicos", "src/ingestion/download_csv_storage.py", []):
            logger.warning("⚠️  Fallo la descarga desde Storage. Puede faltar historico.")
    else:
        logger.info("\n── FASE 0: Restore — OMITIDO (--local-only)")

    # ── 1. INGESTA (paralela) ─────────────────────────────────────────────────
    logger.info("\n── FASE 1: Ingesta")
    ok = run_parallel([
        ("Ingesta — Tráfico",         "src/ingestion/download_traffic.py",     ingestion_args),
        ("Ingesta — Calidad del aire", "src/ingestion/download_air_quality.py", ingestion_args),
        ("Ingesta — Meteorología",     "src/ingestion/download_weather.py",     ingestion_args),
    ])
    if not ok:
        logger.warning("⚠️  Una o más fuentes de ingesta fallaron — el pipeline continúa con lo disponible.")

    # ── 2. LIMPIEZA ───────────────────────────────────────────────────────────
    logger.info("\n── FASE 2: Limpieza de datos")
    if not run_script("Limpieza datos aire (clean_raw_data)", "src/transformation/clean_raw_data.py"):
        logger.warning("⚠️  La limpieza de datos falló — continuando con datos sin limpiar.")

    # ── 3. TRANSFORMACIÓN ─────────────────────────────────────────────────────
    logger.info("\n── FASE 3: Transformación")
    current_year = str(datetime.now().year)
    if not run_script("Tráfico 15 min → 1 hora", "src/ingestion/migrate_traffic_to_hourly.py", ["--year", current_year]):
        logger.error("✗ La transformación de tráfico falló. Abortando.")
        sys.exit(1)

    # ── 3b. COMPRESIÓN ────────────────────────────────────────────────────────
    logger.info("\n── FASE 3b: Compresión")
    compress_csvs([
        ROOT_DIR / "data" / "raw" / "traffic",
        ROOT_DIR / "data" / "raw" / "air",
        ROOT_DIR / "data" / "raw" / "weather",
    ])

    # ── 3c. UPLOAD A STORAGE ──────────────────────────────────────────────────
    if not local_only:
        logger.info("\n── FASE 3c: Upload de CSVs actualizados a Storage")
        if not run_script("Upload CSVs a Storage", "src/ingestion/upload_csv_storage.py", ["--force"]):
            logger.warning("⚠️  Fallo la subida a Storage.")
    else:
        logger.info("\n── FASE 3c: Upload a Storage — OMITIDO (--local-only)")

    # ── 4. CARGA A SUPABASE ───────────────────────────────────────────────────
    if not local_only:
        logger.info("\n── FASE 4: Carga a Supabase")
        ok = run_parallel([
            ("Backfill Supabase — Tráfico",         "src/ingestion/download_traffic.py",     ["--backfill"]),
            ("Backfill Supabase — Calidad del aire", "src/ingestion/download_air_quality.py", ["--backfill"]),
            ("Backfill Supabase — Meteorología",     "src/ingestion/download_weather.py",     ["--backfill"]),
        ])
        if not ok:
            logger.warning("⚠️  Algún backfill a Supabase falló — continuando.")
    else:
        logger.info("\n── FASE 4: Carga a Supabase — OMITIDA (--local-only)")

    # ── 5. FEATURES ───────────────────────────────────────────────────────────
    logger.info("\n── FASE 5: Feature engineering")
    if not run_script("Build features v6", "src/features/build_features_v6.py"):
        logger.error("✗ La construcción de features falló. Abortando.")
        sys.exit(1)

    # ── 5b. STATION DAILY CSV (input para análisis causal v9) ────────────────
    logger.info("\n── FASE 5b: Build station daily")
    if not run_script("Build station daily CSV", "src/features/build_station_daily.py"):
        logger.warning("⚠️  build_station_daily falló — el análisis v9 puede no ejecutarse.")

    # ── 6. ENTRENAMIENTO v8 (100 iter RandomSearch, siempre con --tune) ───────
    if not skip_training:
        logger.info("\n── FASE 6: Entrenamiento v8 (100 iteraciones RandomSearch)")
        if not run_script("Train model v8 --tune", "src/ml/train_model_v8.py", ["--tune"]):
            logger.error("✗ El entrenamiento falló. Abortando.")
            sys.exit(1)
    else:
        logger.info("\n── FASE 6: Entrenamiento — OMITIDO (--skip-training)")

    # ── 7. ANÁLISIS CAUSAL v9 (Event Study + Synthetic Control) ──────────────
    logger.info("\n── FASE 7: Análisis causal v9")
    station_csv = find_station_daily_csv()
    if station_csv:
        if not run_script(
            "Event Study DiD + Synthetic Control",
            "src/ml/plot_causal_v9.py",
            ["--station-data", str(station_csv)],
        ):
            logger.warning("⚠️  El análisis causal v9 falló — continuando.")
    else:
        logger.warning("⚠️  station_daily.csv no encontrado — omitiendo análisis v9.")
        logger.warning("    Ejecuta build_station_daily.py o coloca el CSV en data/processed/")

    # ── 8. PREDICCIÓN (d1) ───────────────────────────────────────────────────
    logger.info("\n── FASE 8: Predicción d1")
    if not run_script("Predecir mañana", "src/ml/predict.py", ["--with-forecast"]):
        logger.warning("⚠️  La predicción falló.")

    # ── 9. DASHBOARD HTML ─────────────────────────────────────────────────────
    logger.info("\n── FASE 9: Dashboard")
    dated_name    = f"dashboard_{datetime.now().strftime('%Y-%m-%d')}.html"
    dashboard_dir = ROOT_DIR / "reports" / "html"
    if not run_script(
        f"Build dashboard → {dated_name}",
        str(dashboard_dir / "build_v10.py"),
        ["--output", dated_name],
        cwd=dashboard_dir,
    ):
        logger.warning("⚠️  La generación del dashboard falló.")
    else:
        logger.info(f"  → reports/html/{dated_name}")

    logger.info("\n" + "=" * 60)
    logger.success("  PIPELINE COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()