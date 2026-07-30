"""Prepara datos de Kunak para modelado sin modificar la descarga original."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).parent.parent.parent
SOURCE_DIR = ROOT_DIR / "data" / "raw" / "air" / "source"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "air_clean_hourly.parquet"
MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "air_source_manifest.json"

ZERO_IMPOSSIBLE = ["NO2", "ICA", "PM10", "PM2.5"]
MIN_HOURLY_COVERAGE = 18
MAX_INTERPOLATION_HOURS = 3


def load_source() -> pd.DataFrame:
    files = sorted(SOURCE_DIR.glob("kunak_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay descargas originales en {SOURCE_DIR}. "
            "Ejecuta download_air_quality.py --refresh-history."
        )

    df = pd.concat([pd.read_csv(file, low_memory=False) for file in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["timestamp", "estacion", "contaminante"])
    df = df.drop_duplicates(["timestamp", "estacion", "contaminante"], keep="last")
    return df.sort_values(["estacion", "contaminante", "timestamp"]).reset_index(drop=True)


def fill_short_gaps(group: pd.DataFrame) -> pd.DataFrame:
    """Interpola solo huecos interiores de hasta tres horas, nunca apagones."""
    group = group.copy().sort_values("timestamp")
    values = group["valor"].copy()
    missing = values.isna() & ~group["sensor_down"]
    run_id = missing.ne(missing.shift()).cumsum()

    for _, positions in group[missing].groupby(run_id[missing]).groups.items():
        idx = list(positions)
        if len(idx) > MAX_INTERPOLATION_HOURS:
            continue
        first_pos = group.index.get_loc(idx[0])
        last_pos = group.index.get_loc(idx[-1])
        if first_pos == 0 or last_pos == len(group) - 1:
            continue
        left = values.iloc[first_pos - 1]
        right = values.iloc[last_pos + 1]
        if pd.isna(left) or pd.isna(right):
            continue
        values.loc[idx] = np.linspace(left, right, len(idx) + 2)[1:-1]
        group.loc[idx, "is_imputed"] = True

    group["valor"] = values
    return group


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.floor("D")
    df["sensor_down"] = False
    df["is_imputed"] = False

    relevant = df["contaminante"].isin(ZERO_IMPOSSIBLE)
    daily = (
        df.loc[relevant]
        .groupby(["estacion", "contaminante", "date"])["valor"]
        .agg(total="count", zeros=lambda values: (values == 0).sum())
    )
    sensor_down = daily[daily["zeros"] / daily["total"] > 0.5].index
    down_index = pd.MultiIndex.from_frame(df[["estacion", "contaminante", "date"]])
    df.loc[down_index.isin(sensor_down), "sensor_down"] = True

    df.loc[df["sensor_down"], "valor"] = np.nan
    df.loc[relevant & (df["valor"] == 0), "valor"] = np.nan

    groups = []
    for _, group in df.groupby(["estacion", "contaminante"], group_keys=False):
        groups.append(fill_short_gaps(group))
    result = pd.concat(groups, ignore_index=True)
    result["valid_hour"] = result["valor"].notna() & ~result["sensor_down"]
    return result.sort_values(["timestamp", "estacion", "contaminante"]).reset_index(drop=True)


def main():
    df = clean(load_source())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps({
        "source": "Kunak Ayuntamiento de Vitoria-Gasteiz",
        "format_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(df)),
        "min_hourly_coverage": MIN_HOURLY_COVERAGE,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    daily = (
        df.groupby(["date", "estacion", "contaminante"])
        .agg(valid_hours=("valid_hour", "sum"), sensor_down=("sensor_down", "max"))
    )
    enough = (daily["valid_hours"] >= MIN_HOURLY_COVERAGE).sum()
    print(f"[OK] {OUTPUT_PATH}: {len(df):,} lecturas")
    print(f"     estación-días con cobertura >= {MIN_HOURLY_COVERAGE}h: {enough:,}/{len(daily):,}")
    print(f"     lecturas imputadas: {int(df['is_imputed'].sum()):,}")
    print(f"     lecturas de apagón conservadas como NaN: {int((df['sensor_down'] & df['valor'].isna()).sum()):,}")


if __name__ == "__main__":
    main()
