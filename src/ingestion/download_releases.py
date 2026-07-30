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

if __name__ == "__main__":
    main()
