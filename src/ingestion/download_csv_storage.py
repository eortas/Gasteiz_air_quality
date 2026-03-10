"""
download_csv_storage.py
========================
Descarga los CSVs historicos comprimidos desde Supabase Storage
y los descomprime en data/raw/ para usar en el entrenamiento.

Usado principalmente por GitHub Actions antes de entrenar el modelo.
También util para recuperar el historico en una maquina nueva.

Uso:
    python src/ingestion/download_csv_storage.py

    # Solo trafico:
    python src/ingestion/download_csv_storage.py --traffic-only

    # Solo aire:
    python src/ingestion/download_csv_storage.py --air-only

    # Forzar redownload aunque ya existan localmente:
    python src/ingestion/download_csv_storage.py --force

Requiere:
    SUPABASE_URL y SUPABASE_KEY en .env o como variables de entorno
    (GitHub Actions las inyecta automaticamente desde secrets)
"""

import gzip
import sys
import os
import io
from pathlib import Path
from datetime import datetime

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"
AIR_DIR     = ROOT_DIR / "data" / "raw" / "air"
ENV_FILE    = ROOT_DIR / ".env"

BUCKET_TRAFFIC = "csv-traffic"
BUCKET_AIR     = "csv-air"

TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)
AIR_DIR.mkdir(parents=True, exist_ok=True)


# ─── SUPABASE ─────────────────────────────────────────────────────────────────
def get_supabase_client():
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("ERROR: SUPABASE_URL / SUPABASE_KEY no encontrados")
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


# ─── DESCARGA Y DESCOMPRESION ─────────────────────────────────────────────────
def download_and_decompress(client, bucket: str, dest_dir: Path, force: bool = False):
    """
    Lista todos los .gz del bucket, los descarga y descomprime en dest_dir.
    Saltea los que ya existen localmente (a menos que force=True).
    """
    print(f"  Listando ficheros en bucket '{bucket}'...")
    try:
        files = client.storage.from_(bucket).list()
    except Exception as e:
        print(f"  ERROR listando bucket: {e}")
        return False

    gz_files = [f for f in files if f.get("name", "").endswith(".gz")]

    if not gz_files:
        print(f"  No se encontraron ficheros .gz en '{bucket}'")
        return True

    print(f"  {len(gz_files)} ficheros encontrados en Storage")
    print()

    ok = 0
    for f in gz_files:
        gz_name   = f["name"]
        csv_name  = gz_name[:-3]  # quitar .gz
        dest_path = dest_dir / csv_name

        # Comprobar si ya existe localmente
        if dest_path.exists() and not force:
            size_mb = dest_path.stat().st_size / 1024 / 1024
            print(f"  {csv_name} ya existe ({size_mb:.1f} MB) — saltando")
            ok += 1
            continue

        print(f"  Descargando {gz_name}...", end=" ", flush=True)
        try:
            data = client.storage.from_(bucket).download(gz_name)
            compressed_mb = len(data) / 1024 / 1024
            print(f"{compressed_mb:.1f} MB descargados, descomprimiendo...", end=" ", flush=True)
        except Exception as e:
            print(f"ERROR descargando: {e}")
            continue

        # Descomprimir
        try:
            with gzip.open(io.BytesIO(data), "rb") as gz_in:
                csv_data = gz_in.read()

            dest_path.write_bytes(csv_data)
            final_mb = len(csv_data) / 1024 / 1024
            print(f"OK → {csv_name} ({final_mb:.1f} MB)")
            ok += 1
        except Exception as e:
            print(f"ERROR descomprimiendo: {e}")
            # Limpiar fichero parcial
            if dest_path.exists():
                dest_path.unlink()

    print()
    print(f"  Resultado: {ok}/{len(gz_files)} ficheros disponibles localmente")
    return ok == len(gz_files)


def print_local_summary(traffic_dir: Path, air_dir: Path):
    """Muestra un resumen de los ficheros disponibles localmente."""
    print("\n" + "=" * 60)
    print("DATOS DISPONIBLES LOCALMENTE")
    print("=" * 60)

    print(f"\nTrafico ({traffic_dir}):")
    total = 0
    for f in sorted(traffic_dir.glob("trafico_[0-9]*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        total  += size_mb
        print(f"  {f.name:<35} {size_mb:6.1f} MB")
    print(f"  {'TOTAL':<35} {total:6.1f} MB")

    print(f"\nCalidad del aire ({air_dir}):")
    total = 0
    for f in sorted(air_dir.glob("kunak_[0-9]*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        total  += size_mb
        print(f"  {f.name:<35} {size_mb:6.1f} MB")
    print(f"  {'TOTAL':<35} {total:6.1f} MB")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    force        = "--force"        in sys.argv
    traffic_only = "--traffic-only" in sys.argv
    air_only     = "--air-only"     in sys.argv

    print("=" * 60)
    print("DESCARGA DE CSVs HISTORICOS DESDE SUPABASE STORAGE")
    print("=" * 60)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Modo      : {'forzar redownload' if force else 'solo nuevos'}")
    print()

    client = get_supabase_client()
    print()

    success = True

    if not air_only:
        print(f"── TRAFICO → {TRAFFIC_DIR} ──────────────────────────────")
        ok = download_and_decompress(client, BUCKET_TRAFFIC, TRAFFIC_DIR, force=force)
        success = success and ok
        print()

    if not traffic_only:
        print(f"── CALIDAD DEL AIRE → {AIR_DIR} ──────────────────────")
        ok = download_and_decompress(client, BUCKET_AIR, AIR_DIR, force=force)
        success = success and ok
        print()

    print_local_summary(TRAFFIC_DIR, AIR_DIR)

    if not success:
        print("\nALERTA: algunos ficheros no se pudieron descargar")
        sys.exit(1)

    print("\nDatos listos para entrenamiento.")


if __name__ == "__main__":
    main()
