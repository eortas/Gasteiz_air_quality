"""
download_gv_air.py
==================
Descarga las estaciones de tráfico del Gobierno Vasco en Vitoria-Gasteiz y
guarda una serie diaria en `data/processed/gv_daily.csv`.

Estaciones incluidas (Fase 0):
  75 3 DE MARZO   -> NO2, PM10, PM2.5, CO, SO2
  77 LOS HERRAN   -> NO2, PM10, PM2.5
  78 AV. GASTEIZ  -> NO2, PM10, PM2.5, CO

Farmacia (85) se excluye: solo mide O3 y meteorología.

La API es pública, sin autenticación:
  https://api.euskadi.eus/air-quality

Uso:
    python src/ingestion/download_gv_air.py
    python src/ingestion/download_gv_air.py --full
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "gv_daily.csv"
RAW_PATH = ROOT_DIR / "data" / "raw" / "gv" / "gv_daily.csv"

BASE_URL = "https://api.euskadi.eus/air-quality/measurements/daily/stations"
STATIONS = {
    75: "3MARZO",
    77: "HERRAN",
    78: "AVGASTEIZ",
}
POLLUTANTS = {
    "NO2": "no2",
    "PM10": "pm10",
    "PM2,5": "pm25",
}
FIRST_YEAR = 2019
MAX_RETRIES = 4


def fetch_year(station_id: int, year: int) -> list:
    url = (
        f"{BASE_URL}/{station_id}/from/{year}-01-01/to/{year}-12-31"
        "?lang=SPANISH"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=90)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                f"GV {station_id} {year}: HTTP {response.status_code} "
                f"(intento {attempt}/{MAX_RETRIES})"
            )
        except Exception as exc:
            logger.warning(
                f"GV {station_id} {year}: {exc} "
                f"(intento {attempt}/{MAX_RETRIES})"
            )
        time.sleep(2 * attempt)
    return []


def parse_records(station_id: int, station_name: str, records: list) -> list:
    rows = []
    for record in records:
        stations = record.get("station") or []
        if not stations:
            continue
        measurements = {
            item.get("name"): item.get("value")
            for item in stations[0].get("measurements", [])
        }
        row = {
            "date": record.get("date"),
            "station_id": station_id,
            "station": station_name,
        }
        for api_name, column in POLLUTANTS.items():
            row[column] = measurements.get(api_name)
        rows.append(row)
    return rows


def download(full: bool) -> pd.DataFrame:
    existing = pd.DataFrame()
    if OUTPUT_PATH.exists() and not full:
        try:
            existing = pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
            logger.info(f"Histórico GV existente: {len(existing)} filas")
        except Exception as exc:
            logger.warning(f"No se pudo leer {OUTPUT_PATH}: {exc}")

    today = date.today()
    # Solo días cerrados: el día en curso aún no está consolidado.
    end_date = today - timedelta(days=1)
    rows = []

    for station_id, station_name in STATIONS.items():
        station_existing = existing[existing["station_id"] == station_id] if not existing.empty else pd.DataFrame()
        if not station_existing.empty:
            max_date = pd.to_datetime(station_existing["date"]).max().date()
            start_year = max(max_date.year, FIRST_YEAR)
            start_date = max(date(start_year, 1, 1), max_date - timedelta(days=7))
        else:
            start_date = date(FIRST_YEAR, 1, 1)

        if start_date > end_date:
            logger.info(f"GV {station_name}: ya está al día ({start_date})")
            continue

        for year in range(start_date.year, end_date.year + 1):
            records = fetch_year(station_id, year)
            parsed = parse_records(station_id, station_name, records)
            rows.extend(parsed)
            logger.info(f"GV {station_name} {year}: {len(parsed)} días")
            time.sleep(0.3)

    if rows:
        new = pd.DataFrame(rows)
        new["date"] = pd.to_datetime(new["date"])
        combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
        combined = (
            combined.drop_duplicates(subset=["date", "station_id"], keep="last")
            .sort_values(["date", "station_id"])
            .reset_index(drop=True)
        )
        combined = combined[combined["date"].dt.date <= end_date]
        # Nos quedamos con el histórico completo de cada estación.
        combined = combined[combined["date"].dt.year >= FIRST_YEAR]
    else:
        combined = existing

    if combined.empty:
        logger.error("No se ha podido descargar ningún dato del GV.")
        return combined

    combined.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_PATH, index=False, date_format="%Y-%m-%d")

    max_date = combined["date"].max().date()
    logger.success(
        f"[OK] GV daily: {len(combined)} filas, {combined['station'].nunique()} estaciones, "
        f"hasta {max_date} -> {OUTPUT_PATH}"
    )
    if max_date < end_date - timedelta(days=2):
        logger.warning(f"GV retrasado: último dato {max_date}, esperado {end_date}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Descarga desde 2019.")
    args = parser.parse_args()
    logger.info("Descargando estaciones del Gobierno Vasco (Vitoria-Gasteiz)...")
    download(full=args.full)


if __name__ == "__main__":
    main()
