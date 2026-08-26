"""Reconstruye pronósticos meteorológicos D+1 y D+2 sin fuga temporal.

Usamos Previous Runs de Open-Meteo. ``previous_day1`` y ``previous_day2``
representan lo que se conocía uno y dos días antes del día objetivo.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "weather_forecast_history.jsonl"

LATITUDE = 42.8467
LONGITUDE = -2.6716
TIMEZONE = "Europe/Madrid"
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

FORECAST_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "sunshine_duration",
    "weather_code",
]

SUM_VARS = {"precipitation", "rain", "snowfall", "sunshine_duration"}
HORIZONS = (1, 2)


def aggregate_day(day_data: pd.DataFrame, horizon: int) -> dict:
    """Agregamos las 24 horas del día objetivo igual que en producción."""
    features = {}

    for variable in FORECAST_VARS:
        column = f"{variable}_previous_day{horizon}"
        if column not in day_data.columns:
            continue
        values = pd.to_numeric(day_data[column], errors="coerce")
        if values.notna().sum() == 0:
            continue

        if variable in SUM_VARS:
            value = values.sum()
        elif variable == "wind_direction_10m":
            radians = np.deg2rad(values.dropna())
            value = np.degrees(
                np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
            ) % 360
        else:
            value = values.mean()

        features[f"fc_{variable}_d{horizon}"] = float(value)

    suffix = f"d{horizon}"
    temperature = features.get(f"fc_temperature_2m_{suffix}")
    humidity = features.get(f"fc_relative_humidity_2m_{suffix}")
    wind_speed = features.get(f"fc_wind_speed_10m_{suffix}")
    wind_direction = features.get(f"fc_wind_direction_10m_{suffix}")

    if temperature is not None and humidity is not None:
        features[f"fc_dew_point_{suffix}"] = temperature - ((100.0 - humidity) / 5.0)
        features[f"fc_HDD_{suffix}"] = max(0.0, 15.0 - temperature)

    if wind_speed is not None and wind_direction is not None:
        radians = np.deg2rad(wind_direction)
        features[f"fc_wind_u_{suffix}"] = float(-wind_speed * np.sin(radians))
        features[f"fc_wind_v_{suffix}"] = float(-wind_speed * np.cos(radians))

    return features


def download_chunk(start_date: date, end_date: date) -> list[dict]:
    """Descargamos un tramo corto para no superar el límite de la API."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(
            f"{name}_previous_day{horizon}"
            for horizon in HORIZONS
            for name in FORECAST_VARS
        ),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "models": "icon_seamless",
    }
    for attempt in range(5):
        try:
            response = requests.get(API_URL, params=params, timeout=90)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    else:
        response.raise_for_status()

    hourly = pd.DataFrame(response.json().get("hourly", {}))
    if hourly.empty:
        return []

    hourly["target_date"] = pd.to_datetime(hourly["time"]).dt.date
    records = []
    for target_date, day_data in hourly.groupby("target_date"):
        for horizon in HORIZONS:
            features = aggregate_day(day_data, horizon)
            if not features:
                continue
            records.append(
                {
                    "generated_at": datetime.combine(
                        target_date - timedelta(days=horizon),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    ).isoformat(),
                    "target_date": target_date.isoformat(),
                    "source": "open-meteo-previous-runs",
                    "lead_days": horizon,
                    "features": features,
                }
            )
    return records


def record_key(record: dict) -> str:
    """Identificamos cada emisión por fecha objetivo y horizonte."""
    horizon = int(record.get("lead_days", 1))
    return f"{record['target_date']}|d{horizon}"


def load_live_snapshots() -> dict[str, dict]:
    """Conservamos los snapshots reales creados por el pipeline."""
    snapshots = {}
    if not OUTPUT_PATH.exists():
        return snapshots

    for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            generated_at = pd.Timestamp(record["generated_at"])
            target_start = pd.Timestamp(
                record["target_date"], tz=TIMEZONE
            ).tz_convert("UTC")
            if (
                record.get("source") != "open-meteo-previous-runs"
                and generated_at <= target_start
            ):
                snapshots[record_key(record)] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2024-08-10")
    parser.add_argument("--end-date", default=(date.today() - timedelta(days=1)).isoformat())
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if end_date < start_date:
        raise ValueError("--end-date debe ser igual o posterior a --start-date")

    chunks = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=60), end_date)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)

    print(f"Descargando {len(chunks)} tramos meteorológicos")
    records = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        downloads = executor.map(lambda dates: download_chunk(*dates), chunks)
        for chunk_records in downloads:
            for record in chunk_records:
                records[record_key(record)] = record

    # Los snapshots ejecutados realmente tienen prioridad sobre la reconstrucción.
    records.update(load_live_snapshots())
    ordered = [records[key] for key in sorted(records)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    print(f"Guardados {len(ordered)} pronósticos en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
