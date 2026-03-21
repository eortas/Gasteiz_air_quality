"""
migrate_traffic_to_hourly.py
=============================
Convierte los CSVs de tráfico existentes de granularidad 15 min a 1 hora.
Sobreescribe los ficheros originales con los datos agregados.

Uso:
    python src/ingestion/migrate_traffic_to_hourly.py           # solo año actual
    python src/ingestion/migrate_traffic_to_hourly.py --all    # todos los años
    python src/ingestion/migrate_traffic_to_hourly.py --year 2024
    python src/ingestion/migrate_traffic_to_hourly.py --dry-run
"""

import pandas as pd
import sys
from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent.parent
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega un DataFrame de tráfico de 15 min a 1 hora.
    Acepta columnas en formato camelCase (startDate/endDate)
    o snake_case (start_date/end_date).
      - volume   : suma (total de vehículos en la hora)
      - occupancy: media (% medio de ocupación)
      - load     : media (carga media)
    """
    # Normalizar nombres de columna a snake_case
    df = df.rename(columns={
        "startDate": "start_date",
        "endDate":   "end_date",
    })

    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   utc=True, errors="coerce")
    df = df.dropna(subset=["start_date"])

    # Truncar al inicio de la hora
    df["hour"] = df["start_date"].dt.floor("h")

    agg = (
        df.groupby(["hour", "code"])
        .agg(
            volume    = ("volume",    "sum"),
            occupancy = ("occupancy", "mean"),
            load      = ("load",      "mean"),
        )
        .reset_index()
    )

    agg["start_date"] = agg["hour"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    agg["end_date"]   = (agg["hour"] + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    agg = agg.drop(columns=["hour"])

    agg["occupancy"] = agg["occupancy"].round(4)
    agg["load"]      = agg["load"].round(4)

    return agg[["code", "start_date", "end_date", "volume", "occupancy", "load"]]


def main():
    from datetime import datetime
    dry_run = "--dry-run" in sys.argv
    process_all = "--all" in sys.argv

    # --year 2024  →  solo ese año
    year_arg = None
    if "--year" in sys.argv:
        idx = sys.argv.index("--year")
        if idx + 1 < len(sys.argv):
            year_arg = int(sys.argv[idx + 1])

    target_year = year_arg or (None if process_all else datetime.now().year)

    print("=" * 60)
    print("MIGRACION TRAFICO: 15 MIN -> 1 HORA")
    print("=" * 60)
    if dry_run:
        print("  MODO DRY RUN — no se sobreescribe nada")
    if target_year:
        print(f"  Procesando solo año: {target_year}")
    print()

    all_files = sorted(TRAFFIC_DIR.glob("trafico_[0-9][0-9][0-9][0-9].csv"))
    if target_year:
        files = [f for f in all_files if str(target_year) in f.name]
    else:
        files = all_files

    if not files:
        print(f"  No se encontraron ficheros en {TRAFFIC_DIR}")
        sys.exit(1)

    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

        df = pd.read_csv(f, dtype={"code": str})

        # Mostrar columnas detectadas en el primer fichero
        date_col = "startDate" if "startDate" in df.columns else "start_date"
        rows_before = len(df)

        df_hourly  = aggregate_to_hourly(df)
        rows_after = len(df_hourly)
        reduction  = (1 - rows_after / rows_before) * 100

        print(f"{rows_before:,} filas -> {rows_after:,} filas (reduccion {reduction:.0f}%)")

        if not dry_run:
            df_hourly.to_csv(f, index=False, encoding="utf-8")
            new_size = f.stat().st_size / 1024 / 1024
            print(f"    Guardado: {new_size:.1f} MB")

    print()
    if dry_run:
        print("Dry run completado — ejecuta sin --dry-run para aplicar los cambios")
    else:
        print("Migracion completada.")
        print()
        print("Siguientes pasos:")
        print("  1. Ejecutar el schema actualizado en Supabase SQL Editor")
        print("  2. Recargar Supabase con datos horarios:")
        print("     python src/ingestion/download_traffic.py --backfill")
        print("  3. Subir CSVs actualizados a Supabase Storage:")
        print("     python src/ingestion/upload_csv_storage.py --traffic-only --force")


if __name__ == "__main__":
    main()