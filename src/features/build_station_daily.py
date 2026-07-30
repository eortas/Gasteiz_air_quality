"""
build_station_daily.py
======================
Genera data/processed/station_daily.csv desde la capa limpia de Kunak.

El CSV resultante tiene una fila por día y una columna por cada combinación
estación × contaminante (ej. PAUL_NO2, BEATO_PM10, ...).

Es el input que necesitan:
  - src/ml/train_model_v9.py  (Event Study DiD + Synthetic Control)

Se ejecuta automáticamente desde run_pipeline.py (Fase 5b),
justo después de build_features_v6.py.

Uso:
    python src/features/build_station_daily.py
    python src/features/build_station_daily.py --start 2024-03-01
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH   = PROCESSED_DIR / "station_daily.csv"
AIR_CLEAN_PATH = PROCESSED_DIR / "air_clean_hourly.parquet"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONTAMINANTS  = ["NO2", "PM10", "PM2.5", "ICA"]
ZBE_STATIONS  = ["PAUL", "FUEROS"]
OUT_STATIONS  = ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"]
ALL_STATIONS  = ZBE_STATIONS + OUT_STATIONS

MIN_AIR_HOURS = 18


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg)


def section(title):
    log(); log("=" * 60); log(f"  {title}"); log("=" * 60)


# ─── CARGA ────────────────────────────────────────────────────────────────────
def load_clean_air(start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    section("1. Cargando aire limpio de Kunak")

    if not AIR_CLEAN_PATH.exists():
        log(f"  [ERROR] No se encuentra {AIR_CLEAN_PATH}")
        log("  Ejecuta primero: python src/transformation/prepare_air_clean.py")
        sys.exit(1)

    df = pd.read_parquet(AIR_CLEAN_PATH)
    log(f"\n  Filas brutas    : {len(df):,}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.tz_convert("Europe/Madrid").dt.normalize().dt.tz_localize(None)

    # Filtrar contaminantes relevantes
    df = df[df["contaminante"].isin(CONTAMINANTS)].copy()

    # Filtrar estaciones conocidas
    df = df[df["estacion"].isin(ALL_STATIONS)].copy()

    # Conservamos NaN de apagones y solo aceptamos estaciones-día con cobertura suficiente.
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "estacion", "contaminante"])
    removed = before - len(df)
    if removed > 0:
        log(f"  Duplicados eliminados: {removed:,}")

    # Filtro de fecha de inicio
    if start_date is not None:
        df = df[df["date"] >= start_date.tz_localize(None)]
        log(f"  Filtrando desde {start_date.date()}")

    log(f"  Filas tras limpieza : {len(df):,}")
    log(f"  Estaciones          : {sorted(df['estacion'].unique())}")
    log(f"  Contaminantes       : {sorted(df['contaminante'].unique())}")
    log(f"  Rango               : {df['date'].min().date()} -> {df['date'].max().date()}")

    return df


# ─── AGREGADO DIARIO ──────────────────────────────────────────────────────────
def build_daily_pivot(df: pd.DataFrame) -> pd.DataFrame:
    section("2. Agregando a media diaria")

    df["valid_hour"] = df.get("valid_hour", df["valor"].notna()).astype(bool)
    daily = (
        df.groupby(["date", "estacion", "contaminante"])
        .agg(valor=("valor", "mean"), valid_hours=("valid_hour", "sum"))
        .reset_index()
    )
    daily.loc[daily["valid_hours"] < MIN_AIR_HOURS, "valor"] = np.nan

    # Nombre de columna: PAUL_NO2, BEATO_PM10, etc.
    daily["col"] = (
        daily["estacion"] + "_"
        + daily["contaminante"].str.replace(".", "", regex=False).str.replace(" ", "", regex=False)
    )

    pivot = daily.pivot_table(
        index="date", columns="col", values="valor", aggfunc="mean"
    ).reset_index()
    pivot.columns.name = None

    # Reindexar a rango continuo de fechas
    date_range = pd.date_range(pivot["date"].min(), pivot["date"].max(), freq="D")
    pivot = (
        pivot.set_index("date")
        .reindex(date_range)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    data_cols = [c for c in pivot.columns if c != "date"]

    log(f"  Días en el pivote   : {len(pivot)}")
    log(f"  Columnas            : {len(data_cols)}")
    log("  Apagones y huecos   : conservados como NaN (sin interpolación)")

    # Cobertura por columna
    log(f"\n  Cobertura por columna:")
    log(f"  {'Columna':<25}  {'Cobertura':>10}  {'Primer dato':>12}")
    for col in sorted(data_cols):
        pct   = pivot[col].notna().mean() * 100
        first = pivot.loc[pivot[col].notna(), "date"].min()
        first_str = str(first.date()) if pd.notna(first) else "-"
        log(f"  {col:<25}  {pct:>9.1f}%  {first_str:>12}")

    return pivot


# ─── ESTADÍSTICAS PRE/POST ZBE ────────────────────────────────────────────────
def print_zbe_stats(pivot: pd.DataFrame):
    section("3. Estadísticas pre/post ZBE")

    # Importar desde config central (Fix auditoría #17)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import ZBE_DATE_NAIVE as ZBE_DATE
    pre      = pivot[pivot["date"] < ZBE_DATE]
    post     = pivot[pivot["date"] >= ZBE_DATE]

    log(f"  Pre-ZBE  : {len(pre)} días ({pre['date'].min().date()} -> {pre['date'].max().date()})")
    log(f"  Post-ZBE : {len(post)} días ({post['date'].min().date()} -> {post['date'].max().date()})")

    log(f"\n  {'Columna':<25}  {'Pre media':>10}  {'Post media':>11}  {'Cambio':>8}")
    log(f"  {'-'*25}  {'-'*10}  {'-'*11}  {'-'*8}")

    # Mostrar solo ZBE stations para los 4 contaminantes principales
    for cont in ["NO2", "PM10", "PM25", "ICA"]:
        for station in ALL_STATIONS:
            col = f"{station}_{cont}"
            if col not in pivot.columns:
                continue
            pm  = pre[col].mean()
            pom = post[col].mean()
            if pd.isna(pm) or pd.isna(pom):
                continue
            chg = (pom - pm) / pm * 100 if pm > 0 else 0
            arrow = "v" if chg < 0 else "^"
            log(f"  {col:<25}  {pm:>10.2f}  {pom:>11.2f}  {chg:>+7.1f}% {arrow}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None,
                        help="Fecha de inicio (YYYY-MM-DD). Por defecto usa todos los datos.")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start) if args.start else None

    log("=" * 60)
    log("  BUILD STATION DAILY - Vitoria Air Quality")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  Output: {OUTPUT_PATH}")
    log("=" * 60)

    df    = load_clean_air(start_date)
    pivot = build_daily_pivot(df)
    print_zbe_stats(pivot)

    # Guardar
    section("4. Guardando")
    pivot.to_csv(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    log(f"  [OK] {OUTPUT_PATH}")
    log(f"     {len(pivot)} días × {len(pivot.columns)-1} columnas  ({size_mb:.2f} MB)")
    log(f"\n  Siguiente paso: python src/ml/train_model_v9.py")

    log("\n" + "=" * 60)
    log("  [OK] COMPLETADO")
    log("=" * 60)


if __name__ == "__main__":
    main()
