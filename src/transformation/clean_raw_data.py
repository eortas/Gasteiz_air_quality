"""
clean_raw_data.py
==================
Limpia los CSVs crudos de calidad del aire antes de ejecutar build_features.py.

Problemas detectados:
  - 139 días con >50% lecturas en 0 (sensores caídos) → convertir a NaN
  - Ceros parciales en NO2 distribuidos todo el año → convertir a NaN
  - Ceros puntuales en ICA (16-22 por estación) → convertir a NaN
  - Interpolación lineal con límite 3h para huecos cortos
  - Forward/backward fill con límite 6h para huecos medianos

Los CSVs originales se guardan en data/raw/air/backup/ antes de modificar.

Uso:
    python src/transformation/clean_raw_data.py
    python src/transformation/clean_raw_data.py --dry-run   # solo muestra cambios, no guarda
"""

import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
# Ubicación: src/transformation/clean_raw_data.py
# __file__ → src/transformation/ → parent = src/ → parent = vitoria-air-quality/
ROOT_DIR   = Path(__file__).parent.parent.parent
AIR_DIR     = ROOT_DIR / "data" / "raw" / "air"
BACKUP_DIR  = AIR_DIR / "backup"
WEATHER_DIR = ROOT_DIR / "data" / "raw" / "weather"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Contaminantes donde 0 es físicamente imposible en entorno urbano
ZERO_IMPOSSIBLE = ["NO2", "ICA", "PM10", "PM2.5"]

# Si un día tiene más de este % de lecturas en 0 → sensor caído → NaN todo el día
SENSOR_DOWN_THRESHOLD = 0.5

# Límites de interpolación (en número de registros horarios)
INTERP_LIMIT_LINEAR = 3   # hasta 3h → interpolación lineal
FILL_LIMIT_FFILL    = 6   # hasta 6h → forward fill
FILL_LIMIT_BFILL    = 6   # hasta 6h → backward fill

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg)

def section(title):
    log(); log("═" * 65); log(f"  {title}"); log("═" * 65)


# ─── 1. BACKUP ────────────────────────────────────────────────────────────────
def make_backup(dry_run=False):
    section("1. Backup de CSVs originales")

    if dry_run:
        log("  [DRY RUN] Se saltaría el backup")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(AIR_DIR.glob("kunak_*.csv"))

    for f in files:
        dest = BACKUP_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            log(f"  ✅ Backup: {f.name}")
        else:
            log(f"  ⏭  Ya existe backup: {f.name}")

    log(f"\n  Backups en: {BACKUP_DIR}")


# ─── 2. CARGAR TODOS LOS CSVs ─────────────────────────────────────────────────
def load_all(files):
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["_source_file"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    # Eliminar duplicados en caso de solapamiento de ingestas
    n_antes = len(df)
    df = df.drop_duplicates(subset=["timestamp", "estacion", "contaminante"], keep="last")
    if len(df) < n_antes:
        print(f"  Info: Se eliminaron {n_antes - len(df):,} filas duplicadas en la carga.")
    
    return df.reset_index(drop=True)


# ─── 3. LIMPIAR CEROS ────────────────────────────────────────────────────────
def clean_zeros(df, dry_run=False):
    section("2. Limpieza de ceros imposibles")

    df = df.copy()
    df["date"] = df["timestamp"].dt.floor("D")
    total_ceros = 0
    total_sensor_caido = 0

    for cont in ZERO_IMPOSSIBLE:
        mask_cont = df["contaminante"] == cont
        subset = df[mask_cont].copy()

        if len(subset) == 0:
            continue

        # ── Detectar días con sensor caído (>50% ceros) ──
        daily_stats = subset.groupby(["estacion", "date"]).agg(
            total=("valor", "count"),
            ceros=("valor", lambda x: (x == 0).sum())
        ).reset_index()
        daily_stats["pct_ceros"] = daily_stats["ceros"] / daily_stats["total"]
        sensor_down = daily_stats[daily_stats["pct_ceros"] > SENSOR_DOWN_THRESHOLD]

        n_sensor_down = 0
        for _, row in sensor_down.iterrows():
            mask_day = (
                (df["contaminante"] == cont) &
                (df["estacion"] == row["estacion"]) &
                (df["date"] == row["date"])
            )
            n_sensor_down += mask_day.sum()
            if not dry_run:
                df.loc[mask_day, "valor"] = np.nan

        # ── Convertir todos los ceros restantes a NaN ──
        mask_zero = mask_cont & (df["valor"] == 0)
        n_zeros = mask_zero.sum()

        if not dry_run:
            df.loc[mask_zero, "valor"] = np.nan

        log(f"  {cont}:")
        log(f"    Días sensor caído   : {len(sensor_down):>4} días  ({n_sensor_down:>5} registros → NaN)")
        log(f"    Ceros restantes     : {n_zeros:>5} registros → NaN")
        log(f"    Total convertidos   : {n_sensor_down + n_zeros:>5} registros")

        total_ceros += n_zeros
        total_sensor_caido += n_sensor_down

    log(f"\n  Total registros → NaN : {total_ceros + total_sensor_caido:,}")
    return df


# ─── 4. INTERPOLACIÓN ────────────────────────────────────────────────────────
def interpolate_gaps(df, dry_run=False):
    section("3. Interpolación de huecos")

    if dry_run:
        # En dry run solo contamos cuántos NaN hay por estación/contaminante
        nulos = df[df["contaminante"].isin(ZERO_IMPOSSIBLE)].groupby(
            ["estacion", "contaminante"]
        )["valor"].apply(lambda x: x.isna().sum())
        log("  NaN a interpolar por estación/contaminante:")
        log(nulos[nulos > 0].to_string())
        return df

    df = df.copy()
    total_filled = 0

    for estacion in df["estacion"].unique():
        for cont in ZERO_IMPOSSIBLE:
            mask = (df["estacion"] == estacion) & (df["contaminante"] == cont)
            if mask.sum() == 0:
                continue

            s = df.loc[mask, "valor"].copy()
            n_nan_before = s.isna().sum()

            if n_nan_before == 0:
                continue

            # Paso 1: interpolación lineal para huecos cortos (≤3 registros)
            s = s.interpolate(method="linear", limit=INTERP_LIMIT_LINEAR)

            # Paso 2: forward fill para huecos medianos (≤6 registros)
            s = s.ffill(limit=FILL_LIMIT_FFILL)

            # Paso 3: backward fill para el inicio de la serie
            s = s.bfill(limit=FILL_LIMIT_BFILL)

            n_nan_after = s.isna().sum()
            filled = n_nan_before - n_nan_after
            total_filled += filled

            df.loc[mask, "valor"] = s

            if n_nan_after > 0:
                log(f"  ⚠️  {estacion}/{cont}: {n_nan_after} NaN sin rellenar "
                    f"(huecos >6h — se dejarán como NaN)")

    log(f"\n  Total registros interpolados: {total_filled:,}")
    return df


# ─── 5. VALIDACIÓN FINAL ─────────────────────────────────────────────────────
def validate(df):
    section("4. Validación post-limpieza")

    log(f"  {'Estación':<12} {'Contaminante':<12} {'Nulos':>6} {'%Nulos':>7} "
        f"{'Min':>7} {'Max':>8} {'Media':>8}")
    log(f"  {'-'*12} {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    for estacion in sorted(df["estacion"].unique()):
        for cont in ZERO_IMPOSSIBLE:
            mask = (df["estacion"] == estacion) & (df["contaminante"] == cont)
            s = df.loc[mask, "valor"]
            if len(s) == 0:
                continue
            nulos = s.isna().sum()
            pct = nulos / len(s) * 100
            log(f"  {estacion:<12} {cont:<12} {nulos:>6} {pct:>6.1f}% "
                f"{s.min():>7.2f} {s.max():>8.2f} {s.mean():>8.2f}")

    # Ceros residuales
    log()
    ceros_residuales = ((df["contaminante"].isin(ZERO_IMPOSSIBLE)) &
                        (df["valor"] == 0)).sum()
    if ceros_residuales > 0:
        log(f"  ⚠️  Ceros residuales: {ceros_residuales} — revisar manualmente")
    else:
        log(f"  ✅ Sin ceros residuales en contaminantes críticos")


# ─── 6. GUARDAR CSVs LIMPIOS ──────────────────────────────────────────────────
# ─── 7. LIMPIEZA WEATHER CSVs ─────────────────────────────────────────────────────
def clean_weather_csvs(dry_run=False):
    """Interpola NaN numéricos en los CSVs de meteorología para evitar el error
    'Out of range float values are not JSON compliant' al subir a Supabase."""
    section("6. Limpieza de CSVs de meteorología (NaN / inf)")

    files = sorted(WEATHER_DIR.glob("weather_*.csv"))
    if not files:
        log(f"  \u26a0\ufe0f  No se encontraron weather_*.csv en {WEATHER_DIR}")
        return

    total_fixed = 0
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            continue

        nan_before = df[num_cols].isna().sum().sum()
        inf_count  = np.isinf(df[num_cols].values).sum()

        if nan_before == 0 and inf_count == 0:
            log(f"  \u23ed\ufe0f  {f.name}: sin NaN/inf — omitido")
            continue

        if not dry_run:
            # Sustituir inf por NaN
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            # Interpolación lineal → huecos cortos (hasta 6 registros horarios)
            df[num_cols] = df[num_cols].interpolate(method="linear", limit=6)
            # Forward/backward fill para huecos que queden en los extremos
            df[num_cols] = df[num_cols].ffill(limit=12).bfill(limit=12)

        nan_after  = df[num_cols].isna().sum().sum() if not dry_run else nan_before
        fixed      = nan_before + inf_count - nan_after
        total_fixed += fixed

        log(f"  {'[DRY RUN] ' if dry_run else ''}\u2705 {f.name}: "
            f"{nan_before} NaN + {inf_count} inf → {nan_after} NaN residuales "
            f"({fixed} valores corregidos)")

        if not dry_run:
            df.to_csv(f, index=False)

    log(f"\n  Total valores corregidos en weather: {total_fixed:,}")
    if total_fixed > 0 and not dry_run:
        log("  ℹ️  Los CSVs se han sobrescrito. El backfill a Supabase usará datos limpios.")


# ─── 6. GUARDAR CSVs LIMPIOS ──────────────────────────────────────────────────
def save_clean(df, dry_run=False):
    section("5. Guardando CSVs limpios")

    if dry_run:
        log("  [DRY RUN] No se guardan cambios")
        return

    # Guardar por archivo original — eliminar columnas auxiliares al escribir
    for fname in sorted(df["_source_file"].unique()):
        subset = df[df["_source_file"] == fname].drop(
            columns=["date", "_source_file"], errors="ignore"
        )
        out_path = AIR_DIR / fname
        subset.to_csv(out_path, index=False)
        log(f"  ✅ Guardado: {fname} ({len(subset):,} filas)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv

    log("=" * 65)
    log("  CLEAN RAW DATA — Vitoria Air Quality")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  Modo: {'DRY RUN (sin cambios)' if dry_run else 'ESCRITURA'}")
    log("=" * 65)

    files = sorted(AIR_DIR.glob("kunak_*.csv"))
    if not files:
        log(f"  ❌ No se encuentran CSVs en {AIR_DIR}")
        sys.exit(1)

    log(f"\n  Archivos encontrados: {len(files)}")
    for f in files:
        log(f"    {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")

    make_backup(dry_run=dry_run)

    df = load_all(files)
    log(f"\n  Filas totales cargadas: {len(df):,}")

    df = clean_zeros(df, dry_run=dry_run)
    df = interpolate_gaps(df, dry_run=dry_run)

    validate(df)

    if not dry_run:
        save_clean(df, dry_run=dry_run)

    clean_weather_csvs(dry_run=dry_run)

    log("\n" + "=" * 65)
    if dry_run:
        log("  ✅ DRY RUN COMPLETADO — ejecuta sin --dry-run para aplicar cambios")
    else:
        log("  ✅ LIMPIEZA COMPLETADA")
        log("  Siguiente paso: python build_features.py")
    log("=" * 65)


if __name__ == "__main__":
    main()