"""
upload_csv_storage.py
======================
Comprime los CSVs historicos de trafico, calidad del aire y meteorologia
y los sube a Supabase Storage.

Uso (una sola vez desde tu maquina local):
    python src/ingestion/upload_csv_storage.py

    # Solo trafico:
    python src/ingestion/upload_csv_storage.py --traffic-only

    # Solo aire:
    python src/ingestion/upload_csv_storage.py --air-only

    # Solo meteorologia:
    python src/ingestion/upload_csv_storage.py --weather-only

    # Forzar resubida aunque ya existan:
    python src/ingestion/upload_csv_storage.py --force

Requiere .env en la raiz del proyecto:
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=sb_secret_...

Buckets necesarios en Supabase Storage (crear manualmente en el dashboard):
    - csv-traffic   (para los CSVs de trafico)
    - csv-air       (para los CSVs de calidad del aire)
    - csv-weather   (para los CSVs de meteorologia)
"""

import gzip
import shutil
import sys
import os
import io
from pathlib import Path
from datetime import datetime

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent.parent
TRAFFIC_DIR  = ROOT_DIR / "data" / "raw" / "traffic"
AIR_DIR      = ROOT_DIR / "data" / "raw" / "air"
WEATHER_DIR  = ROOT_DIR / "data" / "raw" / "weather"
ENV_FILE     = ROOT_DIR / ".env"

BUCKET_TRAFFIC = "csv-traffic"
BUCKET_AIR     = "csv-air"
BUCKET_WEATHER = "csv-weather"


# ─── SUPABASE ─────────────────────────────────────────────────────────────────
def get_supabase_client():
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("ERROR: SUPABASE_URL / SUPABASE_KEY no encontrados en .env")
            sys.exit(1)
        client = create_client(url, key)
        print("  Supabase conectado correctamente")
        return client
    except ImportError:
        print("ERROR: pip install supabase python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR conectando Supabase: {e}")
        sys.exit(1)


def list_storage_files(client, bucket: str) -> set:
    """Lista los ficheros ya subidos en un bucket."""
    try:
        result = client.storage.from_(bucket).list()
        return {f["name"] for f in result if f.get("name")}
    except Exception as e:
        print(f"  Error listando bucket {bucket}: {e}")
        return set()


def upload_compressed(client, bucket: str, filepath: Path, force: bool = False) -> bool:
    """
    Comprime un CSV con gzip y lo sube a Supabase Storage.
    Devuelve True si se subio correctamente.
    """
    dest_name = filepath.name + ".gz"

    if not force:
        existing = list_storage_files(client, bucket)
        if dest_name in existing:
            print(f"  {dest_name} ya existe en Storage — saltando (usa --force para resubir)")
            return True

    original_mb = filepath.stat().st_size / 1024 / 1024
    print(f"  Comprimiendo {filepath.name} ({original_mb:.1f} MB)...", end=" ", flush=True)

    buf = io.BytesIO()
    with open(filepath, "rb") as f_in:
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
            shutil.copyfileobj(f_in, gz)

    buf.seek(0)
    compressed_data = buf.read()
    compressed_mb   = len(compressed_data) / 1024 / 1024
    ratio           = (1 - compressed_mb / original_mb) * 100
    print(f"{compressed_mb:.1f} MB (reduccion {ratio:.0f}%)")

    print(f"  Subiendo {dest_name} a bucket '{bucket}'...", end=" ", flush=True)
    try:
        client.storage.from_(bucket).upload(
            path=dest_name,
            file=compressed_data,
            file_options={"content-type": "application/gzip", "upsert": "true"},
        )
        print("OK")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def upload_directory(client, bucket: str, data_dir: Path, pattern: str, force: bool):
    """Sube todos los CSVs de un directorio que coincidan con el patron."""
    if not data_dir.exists():
        print(f"  Directorio no encontrado: {data_dir}")
        return

    files = sorted(data_dir.glob(pattern))
    if not files:
        print(f"  No se encontraron ficheros con patron '{pattern}' en {data_dir}")
        return

    total_original = sum(f.stat().st_size for f in files) / 1024 / 1024
    print(f"  {len(files)} ficheros encontrados ({total_original:.1f} MB total original)")
    print()

    ok = 0
    for f in files:
        if upload_compressed(client, bucket, f, force=force):
            ok += 1
        print()

    print(f"  Resultado: {ok}/{len(files)} ficheros subidos correctamente")


def print_storage_summary(client):
    """Muestra un resumen del estado actual de los buckets."""
    print("\n" + "=" * 60)
    print("ESTADO ACTUAL DE SUPABASE STORAGE")
    print("=" * 60)
    for bucket in [BUCKET_TRAFFIC, BUCKET_AIR, BUCKET_WEATHER]:
        try:
            files = client.storage.from_(bucket).list()
            total = 0
            print(f"\nBucket: {bucket}")
            for f in files:
                if f.get("name"):
                    size_mb = (f.get("metadata", {}) or {}).get("size", 0) / 1024 / 1024
                    total  += size_mb
                    updated = f.get("updated_at", "")[:10]
                    print(f"  {f['name']:<35} {size_mb:6.1f} MB  ({updated})")
            print(f"  {'TOTAL':<35} {total:6.1f} MB")
        except Exception as e:
            print(f"  Error leyendo bucket {bucket}: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    force        = "--force"        in sys.argv
    traffic_only = "--traffic-only" in sys.argv
    air_only     = "--air-only"     in sys.argv
    weather_only = "--weather-only" in sys.argv

    if traffic_only:
        air_only = weather_only = False
    elif air_only:
        traffic_only = weather_only = False
    elif weather_only:
        traffic_only = air_only = False

    print("=" * 60)
    print("SUBIDA DE CSVs HISTORICOS A SUPABASE STORAGE")
    print("=" * 60)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Modo      : {'forzar resubida' if force else 'solo nuevos'}")
    print()

    client = get_supabase_client()
    print()

    if not air_only and not weather_only:
        print(f"── TRAFICO ({TRAFFIC_DIR}) ──────────────────────────────")
        upload_directory(
            client, BUCKET_TRAFFIC, TRAFFIC_DIR,
            pattern="trafico_[0-9][0-9][0-9][0-9].csv",
            force=force
        )
        print()

    if not traffic_only and not weather_only:
        print(f"── CALIDAD DEL AIRE ({AIR_DIR}) ──────────────────────")
        upload_directory(
            client, BUCKET_AIR, AIR_DIR,
            pattern="kunak_[0-9][0-9][0-9][0-9].csv",
            force=force
        )
        print()

    if not traffic_only and not air_only:
        print(f"── METEOROLOGIA ({WEATHER_DIR}) ──────────────────────")
        upload_directory(
            client, BUCKET_WEATHER, WEATHER_DIR,
            pattern="weather_[0-9][0-9][0-9][0-9].csv",
            force=force
        )
        print()

    print_storage_summary(client)
    print()
    print("Siguiente paso: configura el GitHub Action de entrenamiento")
    print("  .github/workflows/train_model.yml")


if __name__ == "__main__":
    main()