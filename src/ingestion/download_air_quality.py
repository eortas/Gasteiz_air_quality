"""
download_air_quality.py
=======================
Descarga el historico completo de calidad del aire de Vitoria-Gasteiz
desde la API Kunak del Ayuntamiento (kunakcloud.com/websites/aytoVitoria.html)
y lo guarda en DOS sitios simultaneamente:
  1. CSVs locales  → data/raw/air/kunak_YYYY.csv     (cache local)
  2. Supabase      → tabla air_measurements           (ultimos 90 dias)

En cada ejecucion:
  - Descarga los meses nuevos desde el checkpoint
  - Sube a Supabase solo los ultimos 90 dias
  - Purga automaticamente registros con mas de 90 dias en Supabase

Contaminantes descargados:
  NO2 (1h), PM10 (24h), PM2.5 (24h), ICA, Temperatura,
  Humedad relativa, Presion, Velocidad viento, Direccion viento

Uso (desde la raiz del proyecto vitoria-air-quality/):
    pip install requests pandas supabase python-dotenv playwright
    playwright install chromium
    python src/ingestion/download_air_quality.py

    # Solo CSV local (sin Supabase):
    python src/ingestion/download_air_quality.py --local-only

    # Carga inicial de los ultimos 90 dias desde CSVs locales a Supabase:
    python src/ingestion/download_air_quality.py --backfill

Requiere .env en la raiz del proyecto:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_secret_...
"""

import asyncio
import urllib.parse
import json
import sys
import os
import time
import requests as req
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from playwright.async_api import async_playwright

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw" / "air"
CHECKPOINT = DATA_DIR / "checkpoint_air.json"
ENV_FILE   = ROOT_DIR / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
URL_BASE      = "https://kunakcloud.com/dashboards/services"
START_DATE    = datetime(2024, 3, 1)
END_DATE      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
BATCH_SIZE    = 500
DELAY_S       = 0.5
SUPABASE_DAYS = 90
SUPABASE_FROM = END_DATE - timedelta(days=SUPABASE_DAYS)

ESTACIONES = {
    1247: "FUEROS",
    1248: "BEATO",
    2085: "LANDAZURI",
    2086: "PAUL",
    3460: "HUETOS",
    3461: "ZUMABIDE",
}

FECHA_INICIO_ESTACION = {
    3460: datetime(2024, 12, 1),
    3461: datetime(2024, 12, 1),
}

CONTAMINANTES = {
    "2395": "ICA",
    "1253": "NO2",
    "1259": "PM10",
    "1260": "PM2.5",
    "50":   "temperatura",
    "706":  "humedad",
    "901":  "presion",
    "909":  "viento_vel",
    "911":  "viento_dir",
}


# ─── SUPABASE ─────────────────────────────────────────────────────────────────
def get_supabase_client():
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("  SUPABASE_URL / SUPABASE_KEY no encontrados en .env")
            print(f"  Buscando .env en: {ENV_FILE}")
            print("  Continuando solo con CSV local...")
            return None
        client = create_client(url, key)
        print("  Supabase conectado correctamente")
        return client
    except ImportError:
        print("  Libreria 'supabase' no instalada -> solo CSV local")
        print("  Instala con: pip install supabase python-dotenv")
        return None
    except Exception as e:
        print(f"  Error conectando Supabase: {e} -> solo CSV local")
        return None


def upsert_batch_supabase(client, records: list, retries: int = 3) -> bool:
    if client is None or not records:
        return True
    for attempt in range(1, retries + 1):
        try:
            client.table("air_measurements").upsert(
                records,
                on_conflict="timestamp,estacion_id,contaminante"
            ).execute()
            return True
        except Exception as e:
            print(f"\n  Error Supabase batch (intento {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def purge_old_supabase(client, days: int = SUPABASE_DAYS):
    """Borra de Supabase los registros con mas de N dias de antiguedad."""
    if client is None:
        return
    cutoff = (END_DATE - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00+00:00")
    try:
        client.table("air_measurements").delete().lt("timestamp", cutoff).execute()
        print(f"  Supabase purgado — eliminados registros anteriores a {cutoff[:10]}")
    except Exception as e:
        print(f"\n  Error purgando Supabase: {e}")


def get_supabase_checkpoint(client):
    """Obtiene el ultimo mes completado en Supabase."""
    if client is None:
        return None
    try:
        result = (client.table("air_measurements")
                  .select("timestamp")
                  .order("timestamp", desc=True)
                  .limit(1)
                  .execute())
        if result.data:
            ts = result.data[0]["timestamp"]
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
            if dt.month == 12:
                return datetime(dt.year + 1, 1, 1)
            return datetime(dt.year, dt.month + 1, 1)
        return None
    except Exception as e:
        print(f"  No se pudo consultar checkpoint de Supabase: {e}")
        return None


# ─── CHECKPOINT LOCAL ─────────────────────────────────────────────────────────
def load_checkpoint(client) -> datetime:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        last_month = datetime.fromisoformat(data["last_completed_month"])
        resume = max(last_month, START_DATE)
        print(f"  Reanudando desde {resume.strftime('%Y-%m')} (checkpoint local)")
        return resume

    supabase_last = get_supabase_checkpoint(client)
    if supabase_last:
        resume = max(supabase_last, START_DATE)
        print(f"  Reanudando desde {resume.strftime('%Y-%m')} (checkpoint Supabase)")
        return resume

    print(f"  Primera ejecucion — arrancando desde {START_DATE.strftime('%Y-%m')}")
    return START_DATE


def save_checkpoint(month_start: datetime):
    CHECKPOINT.write_text(
        json.dumps(
            {"last_completed_month": month_start.isoformat()},
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ─── TOKEN PLAYWRIGHT ─────────────────────────────────────────────────────────
async def get_token() -> str | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()
        token   = None

        async def interceptar(request):
            nonlocal token
            if "getWidgetNavDevicesBasicData" in request.url:
                body   = request.post_data or ""
                params = urllib.parse.parse_qs(body)
                if "token" in params:
                    token = params["token"][0]

        page.on("request", interceptar)
        await page.goto(
            "https://kunakcloud.com/websites/aytoVitoria.html",
            wait_until="networkidle",
            timeout=30000
        )
        await page.wait_for_timeout(2000)
        await browser.close()
        return token


# ─── FETCH DE UN MES ──────────────────────────────────────────────────────────
def fetch_mes(token: str, device_id: int, from_date: str, to_date: str) -> dict | None:
    try:
        r = req.post(
            f"{URL_BASE}/getWidgetNavDevicesFromToChartData",
            data={
                "token":       token,
                "deviceId":    str(device_id),
                "pollutants":  json.dumps(list(CONTAMINANTES.keys())),
                "fromDate":    from_date,
                "toDate":      to_date,
                "groupBy":     "1h",
            },
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 403:
            return None  # token caducado
        else:
            print(f"\n  Error {r.status_code}: {r.text[:80]}")
            return {}
    except Exception as e:
        print(f"\n  Excepcion en fetch: {e}")
        return {}


def json_to_records(data: dict, device_id: int, estacion: str) -> list:
    if not data or "datesReadsCharts" not in data:
        return []

    reads = json.loads(data["datesReadsCharts"])
    filas = []

    for pollutant_id, mediciones in reads.items():
        nombre = CONTAMINANTES.get(str(pollutant_id), f"p_{pollutant_id}")
        for ts_ms, valor, unidad in mediciones:
            filas.append({
                "timestamp":    pd.to_datetime(ts_ms, unit="ms", utc=True).isoformat(),
                "estacion_id":  device_id,
                "estacion":     estacion,
                "contaminante": nombre,
                "valor":        valor,
                "unidad":       unidad,
            })

    return filas


# ─── GENERADOR DE RANGOS MENSUALES ────────────────────────────────────────────
def generar_rangos(start: datetime, end: datetime) -> list:
    rangos = []
    fecha  = start
    while fecha <= end:
        primer_dia = fecha.strftime("%Y-%m-%d")
        if fecha.month == 12:
            ultimo_mes = datetime(fecha.year, 12, 31)
        else:
            ultimo_mes = datetime(fecha.year, fecha.month + 1, 1) - timedelta(days=1)
        ultimo_rango = min(ultimo_mes, end)
        rangos.append((fecha, primer_dia, ultimo_rango.strftime("%Y-%m-%d")))
        if fecha.month == 12:
            fecha = datetime(fecha.year + 1, 1, 1)
        else:
            fecha = datetime(fecha.year, fecha.month + 1, 1)
    return rangos


# ─── GUARDAR CSV LOCAL ────────────────────────────────────────────────────────
def append_to_csv(records: list, year: int):
    if not records:
        return
    df           = pd.DataFrame(records)
    filepath     = DATA_DIR / f"kunak_{year}.csv"
    write_header = not filepath.exists()
    df.to_csv(filepath, mode="a", header=write_header, index=False, encoding="utf-8")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main(local_only: bool = False):
    print("=" * 60)
    print("DESCARGA CALIDAD DEL AIRE — VITORIA-GASTEIZ (KUNAK)")
    print("=" * 60)
    print(f"  Directorio de datos : {DATA_DIR}")
    print()

    client = None if local_only else get_supabase_client()

    resume_from = load_checkpoint(client)
    rangos      = generar_rangos(resume_from, END_DATE)

    total_meses     = len(rangos) * len(ESTACIONES)
    meses_ya_hechos = 0

    print(f"  Rango completo      : {START_DATE.strftime('%Y-%m')} -> {END_DATE.strftime('%Y-%m')}")
    print(f"  Reanudando desde    : {resume_from.strftime('%Y-%m')}")
    print(f"  Meses x estacion    : {len(rangos)} x {len(ESTACIONES)} = {total_meses}")
    print(f"  Supabase            : mantiene ultimos {SUPABASE_DAYS} dias")
    print(f"  Destino             : CSV local {'+ Supabase' if client else '(solo local)'}")
    print()

    if not rangos:
        print("Descarga ya completada.")
        purge_old_supabase(client)
        return

    print("  Obteniendo token Kunak...", end=" ", flush=True)
    token = await get_token()
    if not token:
        print("ERROR — no se pudo obtener token")
        return
    print(f"OK ({token[:20]}...)")
    print()

    supabase_buf = []
    errors       = 0

    for mes_dt, from_date, to_date in rangos:
        es_mes_completo = mes_dt.month != END_DATE.month or mes_dt.year != END_DATE.year

        for device_id, nombre in ESTACIONES.items():
            meses_ya_hechos += 1
            pct = meses_ya_hechos / total_meses * 100

            fecha_inicio_est = FECHA_INICIO_ESTACION.get(device_id, START_DATE)
            if mes_dt < fecha_inicio_est:
                print(f"\r[{pct:5.1f}%] {from_date} {nombre:<12} SKIP (sin datos)", end="", flush=True)
                continue

            print(f"\r[{pct:5.1f}%] {from_date} {nombre:<12}", end="", flush=True)

            data = fetch_mes(token, device_id, from_date, to_date)

            # Token caducado → renovar y reintentar
            if data is None:
                print(" token caducado, renovando...", end="", flush=True)
                token = await get_token()
                if not token:
                    print("\n  ERROR: no se pudo renovar token")
                    errors += 1
                    continue
                data = fetch_mes(token, device_id, from_date, to_date)

            if not data:
                errors += 1
                print(f" ERROR ({errors})", end="", flush=True)
                continue

            records = json_to_records(data, device_id, nombre)
            if not records:
                print(" vacio", end="", flush=True)
                continue

            errors = 0
            print(f" {len(records):>5} filas", end="", flush=True)

            # CSV local
            append_to_csv(records, mes_dt.year)

            # Buffer Supabase — solo ultimos 90 dias
            if mes_dt >= SUPABASE_FROM:
                supabase_buf.extend(records)
            if len(supabase_buf) >= BATCH_SIZE:
                upsert_batch_supabase(client, supabase_buf)
                supabase_buf = []

            time.sleep(DELAY_S)

        if es_mes_completo:
            save_checkpoint(mes_dt)

    # Vaciar buffer final
    if supabase_buf:
        upsert_batch_supabase(client, supabase_buf)

    # Checkpoint del mes en curso
    save_checkpoint(END_DATE.replace(day=1))

    # Purgar registros antiguos
    purge_old_supabase(client)

    # ─── RESUMEN ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("DESCARGA COMPLETADA")
    print("=" * 60)
    total_size = 0
    for f in sorted(DATA_DIR.glob("kunak_*.csv")):
        size_mb    = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        nrows      = sum(1 for _ in open(f, encoding="utf-8")) - 1
        print(f"  {f.name}: {nrows:,} filas, {size_mb:.1f} MB")
    print(f"  Total local: {total_size:.1f} MB")

    if client:
        final = get_supabase_checkpoint(client)
        print(f"  Supabase — ultimo registro: {final}")
        print(f"  Supabase — ventana: ultimos {SUPABASE_DAYS} dias")

    print()
    print("Siguiente paso: EDA")
    print("  python src/eda/eda_aire.py")


# ─── BACKFILL DESDE CSV LOCAL ─────────────────────────────────────────────────
def backfill_from_csv(days: int = SUPABASE_DAYS):
    """
    Sube a Supabase los ultimos N dias desde los CSVs locales.
    Reanuda desde el ultimo registro ya subido para no repetir trabajo.
    """
    print("=" * 60)
    print("BACKFILL AIR QUALITY CSV LOCAL → SUPABASE")
    print("=" * 60)

    client = get_supabase_client()
    if client is None:
        print("Sin conexion a Supabase.")
        return

    cutoff = END_DATE - timedelta(days=days)

    # Reanudar desde el ultimo registro ya subido
    supabase_last = get_supabase_checkpoint(client)
    if supabase_last:
        # get_supabase_checkpoint devuelve el primer dia del mes siguiente
        # retroceder un mes para no perder el mes parcialmente subido
        resume_from = supabase_last - timedelta(days=30)
        resume_from = max(resume_from, cutoff)
        print(f"  Reanudando desde {resume_from.strftime('%Y-%m-%d')} (ultimo en Supabase)")
    else:
        resume_from = cutoff
        print(f"  Primera carga desde {resume_from.strftime('%Y-%m-%d')}")

    print(f"  Rango    : {resume_from.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}")
    print()

    all_frames = []
    for f in sorted(DATA_DIR.glob("kunak_*.csv")):
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
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
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


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill_from_csv(days=SUPABASE_DAYS)
    elif "--local-only" in sys.argv:
        asyncio.run(main(local_only=True))
    else:
        asyncio.run(main(local_only=False))