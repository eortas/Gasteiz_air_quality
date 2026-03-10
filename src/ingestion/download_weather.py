"""
download_weather.py
====================
Descarga datos meteorológicos históricos de Vitoria-Gasteiz
desde Open-Meteo Historical Weather API (gratuita, sin API key)
y los guarda en DOS sitios simultáneamente:
  1. CSVs locales  → data/raw/weather/weather_YYYY.csv  (cache local)
  2. Supabase      → tabla weather_measurements          (ultimos 90 dias)

En cada ejecucion:
  - Descarga los meses nuevos desde el checkpoint
  - Sube a Supabase solo los ultimos 90 dias
  - Purga automaticamente registros con mas de 90 dias en Supabase

Uso (desde la raiz del proyecto vitoria-air-quality/):
    pip install requests pandas supabase python-dotenv python-dateutil
    python src/ingestion/download_weather.py

    # Solo CSV local (sin Supabase):
    python src/ingestion/download_weather.py --local-only

    # Carga inicial de los ultimos 90 dias desde CSVs locales a Supabase:
    python src/ingestion/download_weather.py --backfill

Requiere .env en la raiz del proyecto:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_secret_...
"""

import requests
import pandas as pd
import json
import math
import sys
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path

try:
    from dateutil.relativedelta import relativedelta
except ImportError:
    print("ERROR: pip install python-dateutil")
    sys.exit(1)

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT_DIR / "data" / "raw" / "weather"
CHECKPOINT  = DATA_DIR / "checkpoint_weather.json"
ENV_FILE    = ROOT_DIR / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
VITORIA_LAT   = 42.8467
VITORIA_LON   = -2.6726
TIMEZONE      = "Europe/Madrid"
START_DATE    = date(2024, 3, 1)
END_DATE      = date.today() - timedelta(days=1)
DELAY_S       = 0.5
BATCH_SIZE    = 500
SUPABASE_DAYS = 90
SUPABASE_FROM = datetime.now() - timedelta(days=SUPABASE_DAYS)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "is_day",
    "sunshine_duration",
    "vapour_pressure_deficit",
    "boundary_layer_height",
]

DAILY_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
]


# ─── SUPABASE ─────────────────────────────────────────────────────────────────
def get_supabase_client():
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("  SUPABASE_URL / SUPABASE_KEY no encontrados en .env")
            print("  Continuando solo con CSV local...")
            return None
        client = create_client(url, key)
        print("  Supabase conectado correctamente")
        return client
    except ImportError:
        print("  Libreria 'supabase' no instalada -> solo CSV local")
        return None
    except Exception as e:
        print(f"  Error conectando Supabase: {e} -> solo CSV local")
        return None


def sanitize_records(records: list) -> list:
    """Reemplaza float NaN e inf por None para cumplir con JSON."""
    out = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def upsert_batch_supabase(client, records: list, retries: int = 3) -> bool:
    if client is None or not records:
        return True
    records = sanitize_records(records)  # garantía final anti-NaN
    for attempt in range(1, retries + 1):
        try:
            client.table("weather_measurements").upsert(
                records,
                on_conflict="timestamp"
            ).execute()
            return True
        except Exception as e:
            print(f"\n  Error Supabase batch (intento {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def purge_old_supabase(client, days: int = SUPABASE_DAYS):
    if client is None:
        return
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    try:
        client.table("weather_measurements").delete().lt("timestamp", cutoff).execute()
        print(f"  Supabase purgado — eliminados registros anteriores a {cutoff[:10]}")
    except Exception as e:
        print(f"\n  Error purgando Supabase: {e}")


def get_supabase_checkpoint(client):
    if client is None:
        return None
    try:
        result = (client.table("weather_measurements")
                  .select("timestamp")
                  .order("timestamp", desc=True)
                  .limit(1)
                  .execute())
        if result.data:
            ts_str = result.data[0]["timestamp"]
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
        return None
    except Exception as e:
        print(f"  No se pudo consultar checkpoint de Supabase: {e}")
        return None


# ─── CHECKPOINT LOCAL ─────────────────────────────────────────────────────────
def load_checkpoint(client) -> date:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        last = date.fromisoformat(data["last_completed"])
        next_day = last + timedelta(days=1)
        print(f"  Reanudando desde {next_day} (checkpoint local)")
        return next_day

    supabase_last = get_supabase_checkpoint(client)
    if supabase_last:
        next_day = supabase_last.date() + timedelta(days=1)
        print(f"  Reanudando desde {next_day} (checkpoint Supabase)")
        return next_day

    print(f"  Primera ejecucion — arrancando desde {START_DATE}")
    return START_DATE


def save_checkpoint(d: date):
    CHECKPOINT.write_text(
        json.dumps({"last_completed": d.isoformat()}, ensure_ascii=False),
        encoding="utf-8"
    )


# ─── FETCH DE UN MES ──────────────────────────────────────────────────────────
def fetch_month(year: int, month: int) -> pd.DataFrame:
    first_day = date(year, month, 1)
    last_day  = (first_day + relativedelta(months=1)) - timedelta(days=1)
    last_day  = min(last_day, END_DATE)

    if first_day > END_DATE:
        return pd.DataFrame()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":           VITORIA_LAT,
        "longitude":          VITORIA_LON,
        "start_date":         first_day.isoformat(),
        "end_date":           last_day.isoformat(),
        "hourly":             ",".join(HOURLY_VARS),
        "daily":              ",".join(DAILY_VARS),
        "timezone":           TIMEZONE,
        "models":             "best_match",
        "wind_speed_unit":    "kmh",
        "precipitation_unit": "mm",
        "temperature_unit":   "celsius",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"\n  Error fetch {year}-{month:02d}: {e}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times  = hourly.pop("time", [])

    # Parsear timestamps en UTC para evitar problemas con cambios de hora
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(times, utc=True)
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(TIMEZONE)

    df["latitude"]    = data.get("latitude", VITORIA_LAT)
    df["longitude"]   = data.get("longitude", VITORIA_LON)
    df["elevation_m"] = data.get("elevation", None)
    df["date"]        = df["timestamp"].dt.date

    # Merge con resumen diario
    daily_raw   = data.get("daily", {})
    daily_times = daily_raw.pop("time", [])
    if daily_times:
        df_daily = pd.DataFrame(daily_raw)
        df_daily["date"] = pd.to_datetime(daily_times).date
        df_daily.columns = [
            f"daily_{c}" if c != "date" else c
            for c in df_daily.columns
        ]
        df = df.merge(df_daily, on="date", how="left")

    df.drop(columns=["date"], inplace=True)
    return df


# ─── GUARDAR CSV LOCAL ────────────────────────────────────────────────────────
def append_to_csv(df: pd.DataFrame, year: int):
    if df.empty:
        return
    filepath     = DATA_DIR / f"weather_{year}.csv"
    write_header = not filepath.exists()
    df.to_csv(filepath, mode="a", header=write_header, index=False, encoding="utf-8")


# ─── DESCARGA PRINCIPAL ───────────────────────────────────────────────────────
def main(local_only: bool = False):
    print("=" * 60)
    print("DESCARGA METEOROLOGIA — VITORIA-GASTEIZ (Open-Meteo)")
    print("=" * 60)
    print(f"  Directorio de datos : {DATA_DIR}")
    print()

    client  = None if local_only else get_supabase_client()
    current = load_checkpoint(client)

    total_days = (END_DATE - START_DATE).days
    remaining  = (END_DATE - current).days

    print(f"  Rango completo      : {START_DATE} -> {END_DATE} ({total_days} dias)")
    print(f"  Por descargar       : {current} -> {END_DATE} ({remaining} dias)")
    print(f"  Supabase            : mantiene ultimos {SUPABASE_DAYS} dias")
    print(f"  Destino             : CSV local {'+ Supabase' if client else '(solo local)'}")
    print()

    if remaining <= 0:
        print("Descarga ya completada.")
        purge_old_supabase(client)
        return

    # Generar rangos mensuales
    months = []
    ref = date(current.year, current.month, 1)
    while ref <= END_DATE:
        months.append((ref.year, ref.month))
        ref += relativedelta(months=1)

    print(f"  Meses a procesar    : {len(months)}")
    print()

    supabase_buf = []

    for year, month in months:
        print(f"  Procesando {year}-{month:02d}...", end=" ", flush=True)
        df = fetch_month(year, month)

        if df.empty:
            print("sin datos")
            continue

        # Filtrar desde current (primer mes puede ser parcial)
        df_filtered = df[df["timestamp"].dt.date >= current].copy()
        print(f"{len(df_filtered)} filas")

        # Interpolar NaN en columnas numéricas (huecos cortos de la API)
        # Se aplica ANTES de dividir en CSV y Supabase para que ambos se beneficien.
        num_cols = df_filtered.select_dtypes(include="number").columns
        df_filtered[num_cols] = df_filtered[num_cols].interpolate(method="linear", limit=6)
        df_filtered[num_cols] = df_filtered[num_cols].fillna(method="bfill").fillna(method="ffill")

        # 1. CSV local — guardar timestamp como string legible
        df_csv = df_filtered.copy()
        df_csv["timestamp"] = df_csv["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_csv["timestamp_local"] = df_filtered["timestamp_local"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        append_to_csv(df_csv, year)

        # 2. Buffer Supabase — solo ultimos 90 dias
        cutoff_dt = pd.Timestamp(SUPABASE_FROM, tz="UTC")
        df_supa = df_filtered[df_filtered["timestamp"] >= cutoff_dt].copy()

        if not df_supa.empty:
            df_supa["timestamp"] = df_supa["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            df_supa["timestamp_local"] = df_supa["timestamp_local"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            supabase_buf.extend(df_supa.to_dict(orient="records"))

        if len(supabase_buf) >= BATCH_SIZE:
            upsert_batch_supabase(client, supabase_buf)
            supabase_buf = []

        # Checkpoint al ultimo dia del mes
        last_day_processed = df_filtered["timestamp"].max()
        if pd.notna(last_day_processed):
            save_checkpoint(pd.Timestamp(last_day_processed).date())

        time.sleep(DELAY_S)

    # Vaciar buffer final
    if supabase_buf:
        upsert_batch_supabase(client, supabase_buf)

    purge_old_supabase(client)

    # ─── RESUMEN ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DESCARGA COMPLETADA")
    print("=" * 60)
    total_size = 0
    for f in sorted(DATA_DIR.glob("weather_[0-9]*.csv")):
        size_mb    = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  Total local: {total_size:.1f} MB")

    if client:
        final = get_supabase_checkpoint(client)
        print(f"  Supabase — ultimo registro: {final}")
        print(f"  Supabase — ventana: ultimos {SUPABASE_DAYS} dias")
    print()


# ─── BACKFILL DESDE CSV LOCAL ─────────────────────────────────────────────────
def backfill_from_csv(days: int = SUPABASE_DAYS):
    """Sube a Supabase los ultimos N dias desde los CSVs locales."""
    print("=" * 60)
    print("BACKFILL WEATHER CSV LOCAL → SUPABASE")
    print("=" * 60)

    client = get_supabase_client()
    if client is None:
        print("Sin conexion a Supabase.")
        return

    cutoff = datetime.now() - timedelta(days=days)

    supabase_last = get_supabase_checkpoint(client)
    resume_from   = supabase_last if supabase_last else cutoff
    print(f"  Rango    : {resume_from.date()} -> {END_DATE}")
    print()

    all_frames = []
    for f in sorted(DATA_DIR.glob("weather_[0-9]*.csv")):
        df = pd.read_csv(f)
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df[df["timestamp"] >= pd.Timestamp(resume_from, tz="UTC")]
        if not df.empty:
            all_frames.append(df)
            print(f"  {f.name} — {len(df):,} registros en rango")

    if not all_frames:
        print("No se encontraron registros en ese rango.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    # Interpolar huecos residuales antes de subir a Supabase
    num_cols = combined.select_dtypes(include="number").columns
    combined[num_cols] = combined[num_cols].interpolate(method="linear", limit=6)
    combined[num_cols] = combined[num_cols].fillna(method="bfill").fillna(method="ffill")
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    records = combined.to_dict(orient="records")
    total   = len(records)

    print(f"\n  Total registros a subir: {total:,}")
    print()

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        upsert_batch_supabase(client, batch)
        print(
            f"\r  Subidos: {min(i + BATCH_SIZE, total):,}/{total:,} "
            f"({min(i + BATCH_SIZE, total)/total*100:.1f}%)",
            end="", flush=True
        )

    print(f"\n\n  Backfill completado — {total:,} registros subidos")
    purge_old_supabase(client, days=days)

    final = get_supabase_checkpoint(client)
    print(f"  Supabase — ultimo registro: {final}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill_from_csv(days=SUPABASE_DAYS)
    elif "--local-only" in sys.argv:
        main(local_only=True)
    else:
        main(local_only=False)