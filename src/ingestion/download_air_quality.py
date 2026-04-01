"""
download_air_quality.py
=======================
Descarga el historico completo de calidad del aire de Vitoria-Gasteiz
desde la API Kunak del Ayuntamiento (kunakcloud.com/websites/aytoVitoria.html)
y lo guarda en CSVs locales → data/raw/air/kunak_YYYY.csv (cache local).

En cada ejecucion:
  - Descarga los meses nuevos desde el checkpoint

Contaminantes descargados:
  NO2 (1h), PM10 (24h), PM2.5 (24h), ICA, Temperatura,
  Humedad relativa, Presion, Velocidad viento, Direccion viento

Uso (desde la raiz del proyecto vitoria-air-quality/):
    pip install requests pandas python-dotenv playwright
    playwright install chromium
    python src/ingestion/download_air_quality.py

Requiere playwright para capturar los cookies de sesion requeridos.
"""

import asyncio
import json
import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from playwright.async_api import async_playwright

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT_DIR / "data" / "raw" / "air"
CHECKPOINT = DATA_DIR / "checkpoint_air.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
URL_BASE      = "https://kunakcloud.com/dashboards/services"
START_DATE    = datetime(2024, 3, 1)
END_DATE      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
DELAY_S       = 0.5

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

# ─── MANEJO DE SESION PLAYWRIGHT ──────────────────────────────────────────────
async def get_browser_session(playwright):
    """
    Abre headless Chromium, navega al dashboard, acepta cookies si es necesario
    y captura la cookie de sesión (usualmente `token` en localStorage o devuelta
    en requests de red). Retorna el (token, browser, context).
    """
    print("  Iniciando Playwright para obtener sesión válida...")
    browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    page = await context.new_page()

    token = None

    def capture_token(request):
        nonlocal token
        if "getWidget" in request.url and request.method == "POST":
            post_data = request.post_data
            if post_data and "token=" in post_data:
                for param in post_data.split("&"):
                    if param.startswith("token="):
                        from urllib.parse import unquote
                        token = unquote(param.split("=")[1])
                        break

    page.on("request", capture_token)

    try:
        await page.goto("https://kunakcloud.com/websites/aytoVitoria.html", wait_until="networkidle", timeout=60000)
        await asyncio.sleep(2)
        # click cookiebot accept if exists
        try:
            btn = page.locator("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")
            if await btn.is_visible(timeout=5000):
                await btn.click()
                await asyncio.sleep(1)
        except:
            pass
        # Esperar a que el dashboard dispare las peticiones getWidget (token capturado)
        for _ in range(30):
            if token: break
            await asyncio.sleep(1)
    except Exception as e:
        print(f"\n  [WARN] Error cargando página: {e}")

    await page.close()
    return token, browser, context


# ─── FETCH DE UN MES (usa el contexto del navegador con sus cookies) ──────────
async def fetch_mes_async(browser_context, token: str, device_id: int, from_date: str, to_date: str) -> dict | None:
    """
    Fetch datos usando el contexto del navegador para incluir cookies de sesión.
    Esto evita que el servidor devuelva HTML en lugar de JSON.
    """
    try:
        resp = await browser_context.request.post(
            f"{URL_BASE}/getWidgetNavFromToChartData",
            form={
                "token":      token,
                "id":         str(device_id),
                "pollutants": json.dumps(list(CONTAMINANTES.keys())),
                "fromDate":   from_date,
                "toDate":     to_date,
                "groupBy":    "1h",
            },
            timeout=30000
        )
        if resp.status == 200:
            text = await resp.text()
            try:
                return json.loads(text)
            except:
                print(f"\n  [WARN] Respuesta 200 pero no es JSON: {text[:120]}...")
                return {}
        elif resp.status == 403:
            return None  # token caducado → renovar sesión
        else:
            print(f"\n  Error {resp.status}")
            return {}
    except Exception as e:
        print(f"\n  Excepción en fetch: {e}")
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
        ultimo_dia = min(ultimo_mes, end).strftime("%Y-%m-%d")
        rangos.append((fecha, primer_dia, ultimo_dia))
        fecha = ultimo_mes + timedelta(days=1)
    return rangos


# ─── GESTION CHECKPOINT ───────────────────────────────────────────────────────
def get_local_checkpoint() -> datetime:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, "r") as f:
                data = json.load(f)
                return datetime.fromisoformat(data["last_completed_month"])
        except json.JSONDecodeError:
            pass
    return START_DATE

def update_local_checkpoint(mes_dt: datetime):
    with open(CHECKPOINT, "w") as f:
        json.dump({"last_completed_month": mes_dt.isoformat()}, f)


# ─── FUNCION PRINCIPAL ────────────────────────────────────────────────────────
async def main(manual_token: str = None):
    print("=============================================")
    print("INGESTA LOCAL DE CALIDAD DEL AIRE")
    print("=============================================")
    
    last_run = get_local_checkpoint()
    rangos   = generar_rangos(last_run, END_DATE)

    if not rangos:
        print("Todo actualizado. No hay nada nuevo que descargar.")
        return

    total_meses = len(rangos) * len(ESTACIONES)
    meses_ya_hechos = 0
    errors = 0

    async with async_playwright() as playwright:
        token = manual_token
        browser = None
        browser_ctx = None

        if not token:
            token, browser, browser_ctx = await get_browser_session(playwright)
            if not token:
                print("=============================================")
                print("FATAL: No se pudo obtener el token autómata.")
                print("=============================================")
                return
            print(f"OK ({token[:20]}...)")
        print()

        for mes_dt, from_date, to_date in rangos:
            es_mes_completo = mes_dt.month != END_DATE.month or mes_dt.year != END_DATE.year
            df_mes = []

            for device_id, nombre in ESTACIONES.items():
                meses_ya_hechos += 1
                pct = meses_ya_hechos / total_meses * 100

                fecha_inicio_est = FECHA_INICIO_ESTACION.get(device_id, START_DATE)
                if mes_dt < fecha_inicio_est:
                    print(f"\r[{pct:5.1f}%] {from_date} {nombre:<12} SKIP (sin datos)", end="", flush=True)
                    continue

                print(f"\r[{pct:5.1f}%] {from_date} {nombre:<12}", end="", flush=True)

                if browser_ctx:
                    data = await fetch_mes_async(browser_ctx, token, device_id, from_date, to_date)
                else:
                    # Fallback modo token manual sin browser
                    import requests as req_sync
                    r = req_sync.post(f"{URL_BASE}/getWidgetNavFromToChartData",
                        data={"token": token, "id": str(device_id),
                              "pollutants": json.dumps(list(CONTAMINANTES.keys())),
                              "fromDate": from_date, "toDate": to_date, "groupBy": "1h"},
                        headers={"Referer": "https://kunakcloud.com/websites/aytoVitoria.html"},
                        timeout=30)
                    try: data = r.json() if r.status_code == 200 else (None if r.status_code == 403 else {})
                    except: data = {}

                # Token caducado → renovar sesión completa
                if data is None:
                    print(" sesión caducada, renovando...", end="", flush=True)
                    if browser:
                        await browser.close()
                    token, browser, browser_ctx = await get_browser_session(playwright)
                    if not token:
                        print("\n  ERROR: no se pudo renovar sesión")
                        errors += 1
                        continue
                    data = await fetch_mes_async(browser_ctx, token, device_id, from_date, to_date)

                if not data:
                    errors += 1
                    continue

                filas = json_to_records(data, device_id, nombre)
                if filas:
                    df = pd.DataFrame(filas)
                    df_mes.append(df)
                    time.sleep(DELAY_S)

            print()

            # Guardar el mes en el CSV correspondiente al AÑO
            if df_mes:
                df_final = pd.concat(df_mes, ignore_index=True)
                csv_path = DATA_DIR / f"kunak_{mes_dt.year}.csv"
                
                # Si el CSV existe, agregamos. Usamos drop_duplicates() global para evitar superposicion.
                if csv_path.exists():
                    df_existente = pd.read_csv(csv_path)
                    df_final = pd.concat([df_existente, df_final], ignore_index=True)
                
                df_final.drop_duplicates(subset=["timestamp", "estacion_id", "contaminante"], keep="last", inplace=True)
                df_final.sort_values(["timestamp", "estacion_id", "contaminante"], inplace=True)
                df_final.to_csv(csv_path, index=False)
                
            if es_mes_completo and errors == 0:
                siguiente_mes = mes_dt.replace(day=28) + timedelta(days=4)
                update_local_checkpoint(siguiente_mes.replace(day=1))

        if browser:
            await browser.close()

    print("\n\n=============================================")
    print("RESUMEN LOCAL:")
    total_local_kb = 0
    for file in sorted(DATA_DIR.glob("kunak_*.csv")):
        df = pd.read_csv(file)
        size_mb = file.stat().st_size / (1024 * 1024)
        total_local_kb += file.stat().st_size
        print(f"  {file.name}: {len(df):,} filas, {size_mb:.1f} MB")
    print(f"  Total local: {total_local_kb / (1024*1024):.1f} MB")
    print("=============================================")
    if errors > 0:
        print(f"ATENCION: Hubo {errors} errores. Verifica tu conexion o API.")
    print("Siguiente paso: EDA")
    print("  python src/eda/eda_aire.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos de calidad del aire (Kunak)")
    parser.add_argument("--token", type=str, help="Token manual de Kunak")
    args = parser.parse_args()

    asyncio.run(main(manual_token=args.token))