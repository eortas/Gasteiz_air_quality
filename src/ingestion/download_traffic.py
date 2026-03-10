"""
download_traffic.py
====================
Descarga el historico completo de trafico de Vitoria-Gasteiz (2024-hoy)
y lo guarda en DOS sitios simultaneamente:
  1. CSVs locales  → data/raw/traffic/trafico_YYYY.csv  (agregado por hora)
  2. Supabase      → tabla traffic_measurements          (ultimos 90 dias, por hora)

Granularidad: 1 hora (agregado desde los 15 min originales de la API)
  - volume   : suma de vehículos en la hora
  - occupancy: media del % de ocupación
  - load     : media de la carga

En cada ejecucion:
  - Descarga los dias nuevos desde el checkpoint
  - Agrega a 1h antes de guardar CSV y subir a Supabase
  - Sube a Supabase solo los ultimos 90 dias
  - Purga automaticamente registros con mas de 90 dias en Supabase

Uso (desde la raiz del proyecto vitoria-air-quality/):
    pip install requests pandas supabase python-dotenv
    python src/ingestion/download_traffic.py

    # Solo CSV local (sin Supabase):
    python src/ingestion/download_traffic.py --local-only

    # Carga inicial de los ultimos 90 dias desde CSVs locales a Supabase:
    python src/ingestion/download_traffic.py --backfill

Requiere .env en la raiz del proyecto:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_secret_...
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys
import os

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw" / "traffic"
CHECKPOINT = DATA_DIR / "checkpoint.json"
ENV_FILE   = ROOT_DIR / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
BASE_URL      = "https://www.vitoria-gasteiz.org/c11-01w/traffic"
START_DATE    = datetime(2024, 3, 1)
END_DATE      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
DELAY_SECONDS = 1.0
BATCH_SIZE    = 500
SUPABASE_DAYS = 90
SUPABASE_FROM = END_DATE - timedelta(days=SUPABASE_DAYS)


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


def upsert_sensors_supabase(client, sensors_df: pd.DataFrame):
    if client is None:
        return
    try:
        records = sensors_df.to_dict(orient="records")
        client.table("traffic_sensors").upsert(records, on_conflict="code").execute()
        print(f"  {len(records)} sensores sincronizados en Supabase")
    except Exception as e:
        print(f"  Error subiendo sensores a Supabase: {e}")


def upsert_batch_supabase(client, records: list, retries: int = 3) -> bool:
    if client is None or not records:
        return True
    for attempt in range(1, retries + 1):
        try:
            formatted = []
            for r in records:
                if r is None:
                    continue
                formatted.append({
                    "code":       str(r["code"]),
                    "start_date": r["start_date"],
                    "end_date":   r["end_date"],
                    "volume":     r.get("volume"),
                    "occupancy":  r.get("occupancy"),
                    "load":       r.get("load"),
                })
            if not formatted:
                return True
            client.table("traffic_measurements").upsert(
                formatted,
                on_conflict="code,start_date"
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
    cutoff = (END_DATE - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    try:
        client.table("traffic_measurements").delete().lt("start_date", cutoff).execute()
        print(f"  Supabase purgado — eliminados registros anteriores a {cutoff[:10]}")
    except Exception as e:
        print(f"\n  Error purgando Supabase: {e}")


def get_supabase_checkpoint(client):
    if client is None:
        return None
    try:
        result = (client.table("traffic_measurements")
                  .select("start_date")
                  .order("start_date", desc=True)
                  .limit(1)
                  .execute())
        if result.data:
            date_str = result.data[0]["start_date"]
            return datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        return None
    except Exception as e:
        print(f"  No se pudo consultar checkpoint de Supabase: {e}")
        return None


# ─── INVENTARIO DE SENSORES ───────────────────────────────────────────────────
def get_sensors(client) -> pd.DataFrame:
    sensors_file = DATA_DIR / "sensors.csv"

    if sensors_file.exists():
        print("  Inventario de sensores ya descargado (local)")
        df = pd.read_csv(sensors_file, dtype={"code": str})
        upsert_sensors_supabase(client, df)
        return df

    print("  Descargando inventario de sensores...", end=" ", flush=True)
    r = requests.get(
        BASE_URL,
        params={"action": "inventory", "format": "JSON"},
        timeout=15
    )
    items = r.json()["list"]

    sensors = []
    for item in items:
        if item is not None and item.get("type") == "MS":
            coords = item.get("geometry", {}).get("coordinates", [None, None])
            sensors.append({
                "code":     str(item["code"]),
                "name":     item.get("name", ""),
                "provider": item.get("provider", ""),
                "lon":      coords[0],
                "lat":      coords[1],
            })

    df = pd.DataFrame(sensors)
    df.to_csv(sensors_file, index=False, encoding="utf-8")
    print(f"OK — {len(df)} sensores guardados")
    upsert_sensors_supabase(client, df)
    return df


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────
def load_checkpoint(client) -> datetime:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(data["last_completed"])
        next_day = last + timedelta(days=1)
        print(f"  Reanudando desde {next_day.date()} (checkpoint local)")
        return next_day

    supabase_last = get_supabase_checkpoint(client)
    if supabase_last:
        next_day = supabase_last.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        print(f"  Reanudando desde {next_day.date()} (checkpoint Supabase)")
        return next_day

    print(f"  Primera ejecucion — arrancando desde {START_DATE.date()}")
    return START_DATE


def save_checkpoint(date: datetime):
    CHECKPOINT.write_text(
        json.dumps({"last_completed": date.isoformat()}, ensure_ascii=False),
        encoding="utf-8"
    )


# ─── FETCH DE UN DIA ──────────────────────────────────────────────────────────
def fetch_day(date: datetime):
    params = {
        "action":    "data",
        "format":    "JSON",
        "startDate": date.strftime("%Y-%m-%dT00:00:00"),
        "endDate":   date.strftime("%Y-%m-%dT23:59:59"),
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        if not r.text.strip():
            return []
        raw = r.json().get("list", [])
        return [item for item in raw if item is not None]
    except Exception as e:
        print(f"\n  Error en {date.date()}: {e}")
        return None


# ─── AGREGACION A 1 HORA ──────────────────────────────────────────────────────
def aggregate_to_hourly(records: list) -> list:
    """
    Agrega registros de 15 min a 1 hora.
    - volume   : suma de vehículos
    - occupancy: media del % de ocupación
    - load     : media de la carga
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    df["startDate"] = pd.to_datetime(df["startDate"], utc=True, errors="coerce")
    df = df.dropna(subset=["startDate"])
    df["hour"] = df["startDate"].dt.floor("h")

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
    agg["occupancy"]  = agg["occupancy"].round(4)
    agg["load"]       = agg["load"].round(4)
    agg               = agg.drop(columns=["hour"])

    return agg[["code", "start_date", "end_date", "volume", "occupancy", "load"]].to_dict(orient="records")


# ─── GUARDAR CSV LOCAL ────────────────────────────────────────────────────────
def append_to_csv(records: list, year: int):
    if not records:
        return
    df           = pd.DataFrame(records)
    filepath     = DATA_DIR / f"trafico_{year}.csv"
    write_header = not filepath.exists()
    df.to_csv(filepath, mode="a", header=write_header, index=False, encoding="utf-8")


# ─── DESCARGA PRINCIPAL ───────────────────────────────────────────────────────
def main(local_only: bool = False):
    print("=" * 60)
    print("DESCARGA HISTORICO TRAFICO — VITORIA-GASTEIZ")
    print("=" * 60)
    print(f"  Directorio de datos : {DATA_DIR}")
    print()

    client     = None if local_only else get_supabase_client()
    sensors_df = get_sensors(client)
    print(f"  Sensores MS         : {len(sensors_df)}")

    current    = load_checkpoint(client)
    total_days = (END_DATE - START_DATE).days
    remaining  = (END_DATE - current).days

    print(f"  Rango completo      : {START_DATE.date()} -> {END_DATE.date()} ({total_days} dias)")
    print(f"  Por descargar       : {current.date()} -> {END_DATE.date()} ({remaining} dias)")
    print(f"  Granularidad        : 1 hora (agregado desde 15 min)")
    print(f"  Supabase            : mantiene ultimos {SUPABASE_DAYS} dias")
    print(f"  Destino             : CSV local {'+ Supabase' if client else '(solo local)'}")
    print()

    if remaining <= 0:
        print("Descarga ya completada.")
        purge_old_supabase(client)
        return

    est_min = remaining * DELAY_SECONDS / 60
    print(f"  Tiempo estimado     : ~{est_min:.0f} min ({est_min/60:.1f} h)")
    print("  Ctrl+C para pausar — se reanuda automaticamente")
    print()

    errors       = 0
    supabase_buf = []

    try:
        while current < END_DATE:
            done = (current - START_DATE).days
            pct  = done / total_days * 100
            print(
                f"\r[{pct:5.1f}%] {current.strftime('%Y-%m-%d')} "
                f"| errores: {errors} | buf: {len(supabase_buf):4d}",
                end="", flush=True
            )

            records = fetch_day(current)

            if records is None:
                errors += 1
                if errors > 5:
                    print("\n\nDemasiados errores consecutivos. Revisa la conexion.")
                    break
                time.sleep(10)
                continue

            errors = 0
            ms_records = [r for r in records if r is not None and r.get("type") == "MS"]

            # Agregar a 1 hora
            hourly_records = aggregate_to_hourly(ms_records)

            # 1. CSV local
            append_to_csv(hourly_records, current.year)

            # 2. Buffer Supabase (solo ultimos 90 dias)
            if current >= SUPABASE_FROM:
                supabase_buf.extend(hourly_records)
            if len(supabase_buf) >= BATCH_SIZE:
                upsert_batch_supabase(client, supabase_buf)
                supabase_buf = []

            save_checkpoint(current)
            current += timedelta(days=1)
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n\nPausado por el usuario. Guardando buffer pendiente...")
        if supabase_buf:
            upsert_batch_supabase(client, supabase_buf)
        last_completed = current - timedelta(days=1)
        print(f"  Ultimo dia completado : {last_completed.date()}")
        print("  Vuelve a ejecutar el script para reanudar.")
        sys.exit(0)

    # Vaciar buffer final
    if supabase_buf:
        upsert_batch_supabase(client, supabase_buf)

    purge_old_supabase(client)

    print("\n")
    print("=" * 60)
    print("DESCARGA COMPLETADA")
    print("=" * 60)
    total_size = 0
    for f in sorted(DATA_DIR.glob("trafico_*.csv")):
        if "geo" in f.name:
            continue
        size_mb    = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"  Total local: {total_size:.1f} MB")

    if client:
        final = get_supabase_checkpoint(client)
        print(f"  Supabase — ultimo registro: {final}")
        print(f"  Supabase — ventana: ultimos {SUPABASE_DAYS} dias, granularidad 1h")
    print()


# ─── BACKFILL DESDE CSV LOCAL ─────────────────────────────────────────────────
def backfill_from_csv(days: int = SUPABASE_DAYS):
    """Sube a Supabase los ultimos N dias desde los CSVs locales (ya agregados a 1h)."""
    print("=" * 60)
    print("BACKFILL TRAFICO CSV LOCAL → SUPABASE")
    print("=" * 60)

    client = get_supabase_client()
    if client is None:
        print("Sin conexion a Supabase.")
        return

    cutoff = END_DATE - timedelta(days=days)

    supabase_last = get_supabase_checkpoint(client)
    if supabase_last:
        resume_from = supabase_last.replace(hour=0, minute=0, second=0)
        resume_from = max(resume_from, cutoff)
        print(f"  Reanudando desde {resume_from.date()} (ultimo en Supabase)")
    else:
        resume_from = cutoff
        print(f"  Primera carga desde {resume_from.date()}")

    print(f"  Rango    : {resume_from.date()} -> {END_DATE.date()}")
    print()

    all_frames = []
    for f in sorted(DATA_DIR.glob("trafico_[0-9]*.csv")):
        if "geo" in f.name:
            continue
        df = pd.read_csv(f, dtype={"code": str})
        if "start_date" not in df.columns:
            continue
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
        df = df[df["start_date"] >= pd.Timestamp(resume_from, tz="UTC")]
        if not df.empty:
            all_frames.append(df)
            print(f"  {f.name} — {len(df):,} registros en rango")

    if not all_frames:
        print("No se encontraron registros en ese rango.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined["start_date"] = combined["start_date"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    if "end_date" in combined.columns:
        combined["end_date"] = pd.to_datetime(
            combined["end_date"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    combined = combined.replace([float("inf"), float("-inf")], None)
    combined = combined.where(pd.notna(combined), other=None)
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


# ─── MERGE CON COORDENADAS ────────────────────────────────────────────────────
def merge_with_sensors():
    sensors_file = DATA_DIR / "sensors.csv"
    if not sensors_file.exists():
        print(f"No se encuentra {sensors_file}.")
        sys.exit(1)

    print("Combinando trafico con coordenadas de sensores...")
    sensors = pd.read_csv(sensors_file, dtype={"code": str})

    for f in sorted(DATA_DIR.glob("trafico_*.csv")):
        if "geo" in f.name:
            continue
        print(f"  {f.name}...", end=" ", flush=True)
        df     = pd.read_csv(f, dtype={"code": str})
        merged = df.merge(sensors[["code", "lat", "lon"]], on="code", how="left")
        out    = DATA_DIR / f.name.replace("trafico_", "trafico_geo_")
        merged.to_csv(out, index=False, encoding="utf-8")
        pct_geo = merged["lat"].notna().mean() * 100
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"OK — {len(merged):,} registros, {pct_geo:.1f}% geolocalizados, {size_mb:.1f} MB")

    print("\nArchivos con coordenadas listos en data/raw/traffic/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_with_sensors()
    elif "--backfill" in sys.argv:
        backfill_from_csv(days=SUPABASE_DAYS)
    elif "--local-only" in sys.argv:
        main(local_only=True)
    else:
        main(local_only=False)