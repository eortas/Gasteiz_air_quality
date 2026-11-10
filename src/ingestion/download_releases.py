import requests
import os
import json
from pathlib import Path
from loguru import logger

REPO = "eortas/Gasteiz_air_quality"
RELEASE_TAG = "latest-data"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

ROOT_DIR = Path(__file__).parent.parent.parent

FILES_TO_DOWNLOAD = {
    "data/raw/traffic": ["trafico_2024.csv.gz", "trafico_2025.csv.gz", "trafico_2026.csv.gz", "sensors.csv.gz"],
    "data/raw/weather": ["weather_2024.csv.gz", "weather_2025.csv.gz", "weather_2026.csv.gz"]
}

AIR_MANIFEST = "air_source_manifest.json"

MODEL_TARGETS = [
    f"{pollutant}_{zone}_d{horizon}"
    for horizon in (1, 2)
    for pollutant in ("NO2", "PM10", "PM2.5")
    for zone in ("zbe", "out")
]

def download_file(url: str, path: Path) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            return False
        with open(path, "wb") as out_file:
            for chunk in resp.iter_content(chunk_size=8192):
                out_file.write(chunk)
        return True
    except Exception:
        return False

def main():
    logger.info("Iniciando descarga de CSVs desde GitHub Releases...")
    for folder, files in FILES_TO_DOWNLOAD.items():
        dir_path = ROOT_DIR / folder
        dir_path.mkdir(parents=True, exist_ok=True)
        for f in files:
            file_path = dir_path / f
            url = f"{BASE_URL}/{f}"
            if download_file(url, file_path):
                logger.success(f"OK - {f}")
            else:
                logger.warning(f"Omitido - {f} (aún no existe en la release)")

    manifest_path = ROOT_DIR / "data" / "processed" / AIR_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not download_file(f"{BASE_URL}/{AIR_MANIFEST}", manifest_path):
        logger.warning("La release no contiene un manifiesto de Kunak inmutable; se omite el aire.")
        return

    source_dir = ROOT_DIR / "data" / "raw" / "air" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["kunak_2024.csv.gz", "kunak_2025.csv.gz", "kunak_2026.csv.gz"]:
        if download_file(f"{BASE_URL}/{filename}", source_dir / filename):
            logger.success(f"OK - fuente Kunak {filename}")
        else:
            logger.warning(f"Omitido - fuente Kunak {filename}")

    # Artefactos del modelo: se guardan en Releases para no versionar ~75 MB
    # de joblib en el repositorio. Si no existen, se reentrena.
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded_models = 0
    for target in MODEL_TARGETS:
        filename = f"forecast_v10_{target}.joblib"
        if download_file(f"{BASE_URL}/{filename}", models_dir / filename):
            downloaded_models += 1
    if downloaded_models == len(MODEL_TARGETS):
        logger.success(f"OK - {downloaded_models} artefactos forecast v10")
    elif downloaded_models:
        logger.warning(f"Artefactos incompletos: {downloaded_models}/{len(MODEL_TARGETS)}")
    else:
        logger.warning("La release no contiene artefactos v10; se usará el modelo local o se reentrenará.")

if __name__ == "__main__":
    main()
