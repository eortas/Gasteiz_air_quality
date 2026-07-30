"""Reconstruye predicciones históricas sin usar datos futuros.

Cada predicción se entrena con las filas anteriores al día de origen y usa
únicamente las variables disponibles en ese corte. No modifica los modelos de
producción ni predictions_latest.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "features_daily.parquet"
HISTORY_PATH = PROCESSED_DIR / "predictions_history.json"

TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
]

LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas que puede conocer el modelo en el día de origen."""
    excluded_raw = {
        "NO2_zbe", "NO2_out", "PM10_zbe", "PM10_out",
        "PM2.5_zbe", "PM2.5_out", "ICA_zbe", "ICA_out",
    }
    columns = []
    for column in df.columns:
        if column == "date" or column.startswith("target_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        # A las 22:00 las concentraciones parciales del mismo día son válidas.
        columns.append(column)
    return columns


def select_features(train: pd.DataFrame, feature_cols: list[str], target: pd.Series) -> list[str]:
    """Elige un esquema fijo, sin mirar el objetivo de los días futuros."""
    usable = [column for column in feature_cols if train[column].notna().mean() > 0.5]
    return usable[:80]


def predict_target(df: pd.DataFrame, feature_cols: list[str], source_date: pd.Timestamp,
                   target_name: str) -> float:
    """Entrena hasta antes de source_date y predice target_name para el día siguiente."""
    target_col = f"target_{target_name}"
    current_col = target_name.replace("_d1", "")
    train = df[(df["date"] < source_date) & df[target_col].notna() & df[current_col].notna()].copy()
    row = df[df["date"] == source_date].copy()
    if train.empty or row.empty:
        raise ValueError(f"No hay datos para {target_name} en {source_date.date()}")

    residual = train[target_col] - train[current_col]
    selected = select_features(train, feature_cols, residual)
    medians = train[selected].median().fillna(0)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[selected].fillna(medians), residual)
    predicted_delta = float(model.predict(row[selected].fillna(medians))[0])
    return round(max(0.0, float(row[current_col].iloc[0]) + predicted_delta), 2)


def add_ica(predictions: dict) -> None:
    """Añade ICA con el mismo cálculo determinista del pipeline operativo."""
    import sys
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from config import compute_ica_subindex

    for zone in ["zbe", "out"]:
        values = [
            predictions[f"NO2_{zone}_d1"],
            predictions[f"PM10_{zone}_d1"],
            predictions[f"PM2.5_{zone}_d1"],
        ]
        ica = max(
            compute_ica_subindex("NO2", values[0]),
            compute_ica_subindex("PM10", values[1]),
            compute_ica_subindex("PM2.5", values[2]),
        )
        predictions[f"ICA_{zone}_d1"] = round(float(ica), 2)


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    else:
        history = []

    dates = pd.date_range("2026-07-24", "2026-07-30", freq="D", tz="UTC")
    new_entries = []
    for prediction_date in dates:
        source_date = prediction_date - pd.Timedelta(days=1)
        predictions = {
            target: predict_target(df, feature_cols, source_date, target)
            for target in TARGETS
        }
        add_ica(predictions)
        available_at = source_date.tz_convert("Europe/Madrid").replace(
            hour=22, minute=0, second=0
        )
        new_entries.append({
            "prediction_date": prediction_date.date().isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "available_at": available_at.isoformat(),
            "source_date": source_date.date().isoformat(),
            "run_type": "backfill_no_leakage",
            "model_version": "v8_expanding_replay",
            "predictions": predictions,
        })
        print(f"OK {prediction_date.date()}: {predictions}")

    replay_dates = {entry["prediction_date"] for entry in new_entries}
    history = [entry for entry in history if entry.get("prediction_date") not in replay_dates]
    history.extend(new_entries)
    history.sort(key=lambda entry: (entry.get("prediction_date", ""), entry.get("generated_at", "")))
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Historial actualizado: {len(new_entries)} predicciones reconstruidas")


if __name__ == "__main__":
    main()
