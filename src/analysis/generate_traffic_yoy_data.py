import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"

sys.path.insert(0, str(ROOT_DIR / "src"))
from config import ZBE_DATE_STR as ZBE_DATE

# Sensores del perímetro ZBE
PERIPHERAL_SENSORS = [
    "02164", "02197", "03012", "03014", "03016", "03241", "03017", "03242",
    "03125", "03120", "03024", "03027", "03023", "03025", "02036", "02158",
    "02148", "02159", "02042",
]

TIMEZONE = "Europe/Madrid"
# El CSV transformado es horario (24 lecturas/día); exigimos 20 para descartar
# días parciales sin confundirlos con una caída real de tráfico.
MIN_READINGS_PER_DAY = 20
# Sin sensores comunes suficientes, la media no es comparable entre años
MIN_COMMON_SENSORS = 15


def load_sensor_daily() -> pd.DataFrame:
    """Agrega volumen diario por sensor usando el día local de Vitoria."""
    frames = []
    for year in [2024, 2025, 2026]:
        path = TRAFFIC_DIR / f"trafico_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["code", "start_date", "volume"])
        df["code"] = df["code"].astype(str).str.zfill(5)
        df = df[df["code"].isin(PERIPHERAL_SENSORS)]
        if df.empty:
            continue
        df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", utc=True)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["code", "start_date"])
    df["local_date"] = df["start_date"].dt.tz_convert(TIMEZONE).dt.date

    daily = (
        df.groupby(["code", "local_date"])
        .agg(volume=("volume", "sum"), readings=("volume", "size"))
        .reset_index()
    )
    # Descartamos días parciales: evita que el día intradía o un apagón de
    # sensor se lean como una caída real de tráfico.
    daily = daily[daily["readings"] >= MIN_READINGS_PER_DAY]
    return daily


def generate_yoy():
    print("Iniciando generación de datos YoY de tráfico perimetral...")
    daily = load_sensor_daily()
    if daily.empty:
        print("[ERROR] No se encontraron datos de tráfico.")
        return

    pivot = daily.pivot(index="local_date", columns="code", values="volume").sort_index()

    zbe_date = pd.Timestamp(ZBE_DATE).date()
    results = []
    for dt in pivot.index:
        if dt < zbe_date:
            continue
        # Mismo día de la semana (-52 semanas = -364 días)
        prev_dt = dt - pd.Timedelta(days=364)
        if prev_dt not in pivot.index:
            continue

        current_row = pivot.loc[dt]
        previous_row = pivot.loc[prev_dt]
        # Solo sensores medidos en ambos días: comparación manzana con manzana
        common = current_row.dropna().index.intersection(previous_row.dropna().index)
        if len(common) < MIN_COMMON_SENSORS:
            continue

        results.append({
            "date": dt.isoformat(),
            "current": round(float(current_row[common].mean()), 1),
            "previous": round(float(previous_row[common].mean()), 1),
            "n_sensors": int(len(common)),
            "previous_phase": "pre_zbe" if prev_dt < zbe_date else "year1_zbe",
        })

    output_path = ROOT_DIR / "data" / "processed" / "peripheral_traffic_yoy.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] Datos YoY guardados en {output_path}")
    print(f"     Total días procesados: {len(results)}")
    print(f"     Sensores comunes mínimos por día: {MIN_COMMON_SENSORS}")


if __name__ == "__main__":
    generate_yoy()
