import requests
import os
from pathlib import Path
from loguru import logger

REPO = "eortas/Gasteiz_air_quality"
RELEASE_TAG = "latest-data"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

ROOT_DIR = Path(__file__).parent.parent.parent

FILES_TO_DOWNLOAD = {
    "data/raw/traffic": ["trafico_2024.csv.gz", "trafico_2025.csv.gz", "trafico_2026.csv.gz", "sensors.csv.gz"],
    "data/raw/air": ["kunak_2024.csv.gz", "kunak_2025.csv.gz", "kunak_2026.csv.gz"],
    "data/raw/weather": ["weather_2024.csv.gz", "weather_2025.csv.gz", "weather_2026.csv.gz"]
}

def main():
    logger.info("Iniciando descarga de CSVs desde GitHub Releases...")
    for folder, files in FILES_TO_DOWNLOAD.items():
        dir_path = ROOT_DIR / folder
        dir_path.mkdir(parents=True, exist_ok=True)
        for f in files:
            file_path = dir_path / f
            url = f"{BASE_URL}/{f}"
            try:
                resp = requests.get(url, stream=True)
                if resp.status_code == 200:
                    with open(file_path, "wb") as out_file:
                        for chunk in resp.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                    logger.success(f"OK - {f}")
                else:
                    logger.warning(f"Omitido (HTTP {resp.status_code}) - {f} (aún no existe en la release)")
            except Exception as e:
                logger.error(f"Error descargando {f}: {e}")

if __name__ == "__main__":
    main()
