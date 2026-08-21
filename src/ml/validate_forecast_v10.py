"""Valida el contrato y los artefactos antes de publicar forecast v10."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

TARGETS = [
    "NO2_zbe_d1",
    "NO2_out_d1",
    "PM10_zbe_d1",
    "PM10_out_d1",
    "PM2.5_zbe_d1",
    "PM2.5_out_d1",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate():
    errors = []
    contract = json.loads(
        (PROCESSED_DIR / "feature_contract.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (MODELS_DIR / "forecast_v10_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (MODELS_DIR / "forecast_v10_metrics.json").read_text(encoding="utf-8")
    )
    training_data = pd.read_parquet(
        PROCESSED_DIR / "features_daily.parquet", columns=["date"]
    )
    latest = pd.read_parquet(PROCESSED_DIR / "features_latest.parquet")

    if contract.get("cutoff_hour_local") != 22:
        errors.append("El corte operativo no es 22:00")
    if contract.get("target_window") != "full_local_day":
        errors.append("El objetivo no representa el día local completo")
    if not manifest.get("ready_for_production"):
        errors.append("El manifiesto no supera el control técnico")
    current_data_end = pd.to_datetime(training_data["date"], utc=True).max().date()
    trained_data_end = pd.Timestamp(manifest["data_end"]).date()
    if current_data_end < trained_data_end:
        errors.append("El dataset disponible es anterior al usado para entrenar")
    if current_data_end == trained_data_end and manifest.get(
        "dataset_sha256"
    ) != file_sha256(PROCESSED_DIR / "features_daily.parquet"):
        errors.append("La huella del dataset de entrenamiento no coincide")
    if latest.empty:
        errors.append("features_latest.parquet está vacío")

    for target in TARGETS:
        artifact_path = MODELS_DIR / f"forecast_v10_{target}.joblib"
        metric = metrics.get(f"target_{target}", {})
        if not artifact_path.exists():
            errors.append(f"Falta el artefacto de {target}")
            continue

        artifact = joblib.load(artifact_path)
        artifact_manifest = manifest.get("artifacts", {}).get(target, {})
        features = artifact.get("features", [])
        missing_features = [feature for feature in features if feature not in latest.columns]
        weights = artifact.get("weights", {})
        weight_values = [
            float(weights.get("lightgbm", -1)),
            float(weights.get("extra_trees", -1)),
            float(weights.get("ridge", -1)),
        ]

        if artifact.get("model_version") != "forecast_v10":
            errors.append(f"Versión incorrecta en {target}")
        if artifact_manifest.get("sha256") != file_sha256(artifact_path):
            errors.append(f"La huella del artefacto no coincide en {target}")
        if artifact.get("quality_gate") != "passed":
            errors.append(f"Control de calidad no aprobado en {target}")
        if not features or missing_features:
            errors.append(f"Features incompatibles en {target}: {missing_features[:3]}")
        if any(weight < 0 for weight in weight_values) or sum(weight_values) > 1.000001:
            errors.append(f"Pesos inválidos en {target}")
        if float(artifact.get("interval_half_width_90", 0)) <= 0:
            errors.append(f"Intervalo inválido en {target}")
        if metric.get("quality_gate") != "passed":
            errors.append(f"Métricas no aprobadas en {target}")

        if features and not missing_features and not latest.empty:
            medians = artifact.get("medians", {})
            X = latest.tail(1).reindex(columns=features).fillna(medians).fillna(0).astype(float)
            predictions = [
                artifact["lightgbm"].predict(X)[0],
                artifact["extra_trees"].predict(X)[0],
                artifact["ridge"].predict(X)[0],
            ]
            if not np.isfinite(predictions).all():
                errors.append(f"Inferencia no finita en {target}")

    if errors:
        print("FORECAST V10 NO VÁLIDO")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"FORECAST V10 VÁLIDO: {len(TARGETS)}/{len(TARGETS)} objetivos, "
        f"{len(latest)} filas disponibles"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(validate())
    except Exception as error:
        print(f"FORECAST V10 NO VÁLIDO: {error}")
        sys.exit(1)
