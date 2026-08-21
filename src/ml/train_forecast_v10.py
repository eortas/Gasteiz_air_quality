"""Entrena el modelo operativo D+1 con backtesting y control de calidad."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
DATASET_PATH = PROCESSED_DIR / "features_daily.parquet"
METRICS_PATH = MODELS_DIR / "forecast_v10_metrics.json"
MANIFEST_PATH = MODELS_DIR / "forecast_v10_manifest.json"
REPORT_PATH = MODELS_DIR / "training_report_forecast_v10.txt"

TARGETS = [
    "NO2_zbe_d1",
    "NO2_out_d1",
    "PM10_zbe_d1",
    "PM10_out_d1",
    "PM2.5_zbe_d1",
    "PM2.5_out_d1",
]

CURRENT_AIR = {
    "NO2_zbe", "NO2_out", "PM10_zbe", "PM10_out",
    "PM2.5_zbe", "PM2.5_out", "ICA_zbe", "ICA_out",
}
D1_CALENDAR = {
    "d1_day_of_week", "d1_month", "d1_day_of_year", "d1_is_weekend",
    "d1_dow_sin", "d1_dow_cos", "d1_doy_sin", "d1_doy_cos",
}
WEATHER_TERMS = (
    "temperature", "precipitation", "rain", "snowfall", "wind_",
    "humidity", "cloud", "sunshine", "boundary", "weather_code",
    "dew_point", "ventilation", "pressure", "HDD", "temp_",
)

FINAL_TEST_FRACTION = 0.15
MIN_TEST_DAYS = 90
BACKTEST_WINDOWS = 5
BACKTEST_WINDOW_DAYS = 45
MIN_TRAIN_DAYS = 330
CV_GAP_DAYS = 1
MIN_CV_SKILL = 0.03
MIN_TEST_SKILL = 0.00
MIN_TEST_COVERAGE = 0.85
INTERVAL_CALIBRATION_QUANTILE = 0.95
RANDOM_STATE = 42


def log(message=""):
    print(message)
    REPORT_LINES.append(str(message))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(actual, prediction):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(prediction)) ** 2)))


def mae(actual, prediction):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(prediction))))


def r2(actual, prediction):
    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    denominator = np.sum((actual - actual.mean()) ** 2)
    return float(1 - np.sum((actual - prediction) ** 2) / denominator) if denominator else 0.0


def mape(actual, prediction):
    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    valid = actual > 5.0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs((actual[valid] - prediction[valid]) / actual[valid])) * 100)


def directional_accuracy(actual, prediction, baseline):
    actual_change = np.asarray(actual) - np.asarray(baseline)
    predicted_change = np.asarray(prediction) - np.asarray(baseline)
    actual_direction = np.where(actual_change > 0.25, 1, np.where(actual_change < -0.25, -1, 0))
    predicted_direction = np.where(
        predicted_change > 0.25, 1, np.where(predicted_change < -0.25, -1, 0)
    )
    return float(np.mean(actual_direction == predicted_direction))


def add_d1_calendar(df):
    """Añadimos el calendario del día objetivo sin consultar datos futuros."""
    result = df.copy()
    target_date = pd.to_datetime(result["date"], utc=True) + pd.Timedelta(days=1)
    result["d1_day_of_week"] = target_date.dt.dayofweek
    result["d1_month"] = target_date.dt.month
    result["d1_day_of_year"] = target_date.dt.dayofyear
    result["d1_is_weekend"] = (target_date.dt.dayofweek >= 5).astype(int)
    result["d1_dow_sin"] = np.sin(2 * np.pi * target_date.dt.dayofweek / 7)
    result["d1_dow_cos"] = np.cos(2 * np.pi * target_date.dt.dayofweek / 7)
    result["d1_doy_sin"] = np.sin(2 * np.pi * target_date.dt.dayofyear / 365.25)
    result["d1_doy_cos"] = np.cos(2 * np.pi * target_date.dt.dayofyear / 365.25)
    return result


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe {DATASET_PATH}")

    contract_path = PROCESSED_DIR / "feature_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("cutoff_hour_local") != 22:
        raise ValueError("El modelo operativo exige un dataset construido con corte local 22:00")
    if contract.get("target_window") != "full_local_day":
        raise ValueError("El objetivo debe ser la media del día local completo")

    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = add_d1_calendar(df.sort_values("date").reset_index(drop=True))
    if df["date"].duplicated().any():
        raise ValueError("El dataset contiene fechas duplicadas")
    if len(df) < MIN_TRAIN_DAYS + MIN_TEST_DAYS:
        raise ValueError("No hay suficiente historia para entrenar y reservar un test final")
    return df, contract


def select_features(df, target, include_d1_calendar):
    """Conservamos variables operativas compactas y relacionadas con el objetivo."""
    pollutant = target.split("_")[0]
    selected = []
    for column in df.columns:
        if column == "date" or column.startswith("target_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if df[column].notna().mean() <= 0.50:
            continue

        recent_air = (
            column.startswith(f"{pollutant}_")
            and any(
                term in column
                for term in ["_lag_1d", "_lag_2d", "_lag_3d", "_lag_7d", "_roll_"]
            )
        )
        traffic = (
            column.startswith("traffic_")
            or column in {"exp_traffic_volume_d1", "exp_traffic_occupancy_d1"}
        )
        weather = column.startswith("fc_") or any(term in column for term in WEATHER_TERMS)
        calendar = include_d1_calendar and column in D1_CALENDAR
        coverage = column.startswith("air_stations_valid_")

        if column in CURRENT_AIR or recent_air or traffic or weather or calendar or coverage:
            selected.append(column)

    if not include_d1_calendar:
        selected = [column for column in selected if column not in D1_CALENDAR]
    return selected


def make_models():
    """Creamos dos modelos distintos para reducir errores no correlacionados."""
    lightgbm = LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=100.0))
    return lightgbm, extra_trees, ridge


def rolling_splits(n_rows):
    """Generamos cinco ventanas recientes con un día de separación."""
    splits = []
    train_end = MIN_TRAIN_DAYS
    while train_end + CV_GAP_DAYS + BACKTEST_WINDOW_DAYS <= n_rows:
        train_index = np.arange(train_end)
        validation_index = np.arange(
            train_end + CV_GAP_DAYS,
            train_end + CV_GAP_DAYS + BACKTEST_WINDOW_DAYS,
        )
        splits.append((train_index, validation_index))
        train_end += BACKTEST_WINDOW_DAYS
    if len(splits) < BACKTEST_WINDOWS:
        raise ValueError("No hay suficientes ventanas para el backtesting")
    return splits[-BACKTEST_WINDOWS:]


def choose_weights(actual_change, lightgbm_delta, extra_delta, ridge_delta):
    """Ajustamos pesos no negativos sin amplificar la corrección total."""
    best_weights = (0.0, 0.0, 0.0)
    best_rmse = rmse(actual_change, np.zeros_like(actual_change))
    for lightgbm_weight in np.arange(0, 1.01, 0.05):
        for extra_weight in np.arange(0, 1.01 - lightgbm_weight, 0.05):
            maximum_ridge = 1.0 - lightgbm_weight - extra_weight
            for ridge_weight in np.arange(0, maximum_ridge + 0.001, 0.05):
                prediction = (
                    lightgbm_weight * lightgbm_delta
                    + extra_weight * extra_delta
                    + ridge_weight * ridge_delta
                )
                candidate_rmse = rmse(actual_change, prediction)
                if candidate_rmse < best_rmse:
                    best_rmse = candidate_rmse
                    best_weights = (
                        float(lightgbm_weight), float(extra_weight), float(ridge_weight)
                    )
    return best_weights


def conformal_quantile(errors, level=INTERVAL_CALIBRATION_QUANTILE):
    """Calculamos el cuantil finito conservador para el intervalo conformal."""
    errors = np.sort(np.asarray(errors))
    rank = int(np.ceil((len(errors) + 1) * level)) - 1
    return float(errors[min(max(rank, 0), len(errors) - 1)])


def run_backtest(development, target, features):
    target_column = f"target_{target}"
    current_column = target.replace("_d1", "")
    fold_results = []

    for fold_number, (train_index, validation_index) in enumerate(rolling_splits(len(development)), 1):
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        medians = train[features].median().fillna(0)
        lightgbm, extra_trees, ridge = make_models()
        train_delta = train[target_column] - train[current_column]
        lightgbm.fit(train[features].fillna(medians), train_delta)
        extra_trees.fit(train[features].fillna(medians), train_delta)
        ridge.fit(train[features].fillna(medians), train_delta)
        fold_results.append({
            "fold": fold_number,
            "actual": validation[target_column].to_numpy(),
            "baseline": validation[current_column].to_numpy(),
            "lightgbm": lightgbm.predict(validation[features].fillna(medians)),
            "extra_trees": extra_trees.predict(validation[features].fillna(medians)),
            "ridge": ridge.predict(validation[features].fillna(medians)),
            "start": validation["date"].iloc[0].date().isoformat(),
            "end": validation["date"].iloc[-1].date().isoformat(),
        })

    calibration = fold_results[:2]
    evaluation = fold_results[2:]
    calibration_actual = np.concatenate([fold["actual"] - fold["baseline"] for fold in calibration])
    calibration_lightgbm = np.concatenate([fold["lightgbm"] for fold in calibration])
    calibration_extra = np.concatenate([fold["extra_trees"] for fold in calibration])
    calibration_ridge = np.concatenate([fold["ridge"] for fold in calibration])
    evaluation_weights = choose_weights(
        calibration_actual, calibration_lightgbm, calibration_extra, calibration_ridge
    )

    evaluation_actual = np.concatenate([fold["actual"] for fold in evaluation])
    evaluation_baseline = np.concatenate([fold["baseline"] for fold in evaluation])
    evaluation_lightgbm = np.concatenate([fold["lightgbm"] for fold in evaluation])
    evaluation_extra = np.concatenate([fold["extra_trees"] for fold in evaluation])
    evaluation_ridge = np.concatenate([fold["ridge"] for fold in evaluation])
    evaluation_prediction = (
        evaluation_baseline
        + evaluation_weights[0] * evaluation_lightgbm
        + evaluation_weights[1] * evaluation_extra
        + evaluation_weights[2] * evaluation_ridge
    )

    all_actual = np.concatenate([fold["actual"] for fold in fold_results])
    all_baseline = np.concatenate([fold["baseline"] for fold in fold_results])
    all_lightgbm = np.concatenate([fold["lightgbm"] for fold in fold_results])
    all_extra = np.concatenate([fold["extra_trees"] for fold in fold_results])
    all_ridge = np.concatenate([fold["ridge"] for fold in fold_results])
    production_weights = choose_weights(
        all_actual - all_baseline, all_lightgbm, all_extra, all_ridge
    )
    all_prediction = (
        all_baseline
        + production_weights[0] * all_lightgbm
        + production_weights[1] * all_extra
        + production_weights[2] * all_ridge
    )
    fold_rmse = []
    for fold in fold_results:
        fold_prediction = (
            fold["baseline"]
            + production_weights[0] * fold["lightgbm"]
            + production_weights[1] * fold["extra_trees"]
            + production_weights[2] * fold["ridge"]
        )
        fold_rmse.append(rmse(fold["actual"], fold_prediction))

    return {
        "weights": production_weights,
        "evaluation_weights": evaluation_weights,
        "actual": evaluation_actual,
        "baseline": evaluation_baseline,
        "prediction": evaluation_prediction,
        "all_actual": all_actual,
        "all_prediction": all_prediction,
        "interval_half_width_90": conformal_quantile(
            np.abs(evaluation_actual - evaluation_prediction)
        ),
        "fold_rmse": fold_rmse,
        "fold_ranges": [
            {"fold": fold["fold"], "start": fold["start"], "end": fold["end"]}
            for fold in fold_results
        ],
    }


def feature_importance(lightgbm, extra_trees, ridge, features, weights):
    lightgbm_importance = np.asarray(lightgbm.feature_importances_, dtype=float)
    extra_importance = np.asarray(extra_trees.feature_importances_, dtype=float)
    if lightgbm_importance.sum():
        lightgbm_importance /= lightgbm_importance.sum()
    if extra_importance.sum():
        extra_importance /= extra_importance.sum()
    ridge_importance = np.abs(np.asarray(ridge.named_steps["ridge"].coef_, dtype=float))
    if ridge_importance.sum():
        ridge_importance /= ridge_importance.sum()
    combined = (
        weights[0] * lightgbm_importance
        + weights[1] * extra_importance
        + weights[2] * ridge_importance
    )
    ranking = np.argsort(combined)[::-1][:20]
    return [
        {"feature": features[index], "importance": round(float(combined[index]), 6)}
        for index in ranking
    ]


def train_target(df, target):
    target_column = f"target_{target}"
    current_column = target.replace("_d1", "")
    valid = df[target_column].notna() & df[current_column].notna()
    data = df.loc[valid].reset_index(drop=True)
    test_size = max(MIN_TEST_DAYS, int(len(data) * FINAL_TEST_FRACTION))
    development = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()

    variants = []
    for include_calendar in [False, True]:
        features = select_features(data, target, include_calendar)
        backtest = run_backtest(development, target, features)
        backtest_rmse = rmse(backtest["actual"], backtest["prediction"])
        persistence_rmse = rmse(backtest["actual"], backtest["baseline"])
        variants.append({
            "include_d1_calendar": include_calendar,
            "features": features,
            "backtest": backtest,
            "cv_rmse": backtest_rmse,
            "cv_persistence_rmse": persistence_rmse,
            "selection_score": backtest_rmse + 0.10 * np.std(backtest["fold_rmse"]),
        })

    selected = min(variants, key=lambda variant: variant["selection_score"])
    features = selected["features"]
    backtest = selected["backtest"]
    weights = backtest["weights"]

    development_medians = development[features].median().fillna(0)
    lightgbm_test, extra_test, ridge_test = make_models()
    development_delta = development[target_column] - development[current_column]
    lightgbm_test.fit(development[features].fillna(development_medians), development_delta)
    extra_test.fit(development[features].fillna(development_medians), development_delta)
    ridge_test.fit(development[features].fillna(development_medians), development_delta)
    test_prediction = (
        test[current_column].to_numpy()
        + weights[0] * lightgbm_test.predict(test[features].fillna(development_medians))
        + weights[1] * extra_test.predict(test[features].fillna(development_medians))
        + weights[2] * ridge_test.predict(test[features].fillna(development_medians))
    )
    test_actual = test[target_column].to_numpy()
    test_baseline = test[current_column].to_numpy()

    cv_rmse = selected["cv_rmse"]
    cv_persistence_rmse = selected["cv_persistence_rmse"]
    test_rmse = rmse(test_actual, test_prediction)
    test_persistence_rmse = rmse(test_actual, test_baseline)
    cv_skill = 1 - cv_rmse / cv_persistence_rmse
    test_skill = 1 - test_rmse / test_persistence_rmse
    interval_half_width = backtest["interval_half_width_90"]
    test_coverage = float(np.mean(np.abs(test_actual - test_prediction) <= interval_half_width))
    quality_passed = (
        cv_skill >= MIN_CV_SKILL
        and test_skill >= MIN_TEST_SKILL
        and test_coverage >= MIN_TEST_COVERAGE
    )
    production_method = "ensemble" if quality_passed else "persistence"

    final_medians = data[features].median().fillna(0)
    final_lightgbm, final_extra, final_ridge = make_models()
    full_delta = data[target_column] - data[current_column]
    final_lightgbm.fit(data[features].fillna(final_medians), full_delta)
    final_extra.fit(data[features].fillna(final_medians), full_delta)
    final_ridge.fit(data[features].fillna(final_medians), full_delta)

    artifact = {
        "model_version": "forecast_v10",
        "target": target,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_until": data["date"].iloc[-1].date().isoformat(),
        "features": features,
        "medians": {column: float(final_medians[column]) for column in features},
        "lightgbm": final_lightgbm,
        "extra_trees": final_extra,
        "ridge": final_ridge,
        "weights": {
            "lightgbm": weights[0],
            "extra_trees": weights[1],
            "ridge": weights[2],
        },
        "production_method": production_method,
        "prediction_mode": "residual_ensemble",
        "interval_method": "conservative_rolling_conformal_90",
        "interval_half_width_90": interval_half_width,
        "quality_gate": "passed" if quality_passed else "fallback_persistence",
    }
    artifact_path = MODELS_DIR / f"forecast_v10_{target}.joblib"
    joblib.dump(artifact, artifact_path)

    cv_prediction = backtest["prediction"]
    cv_actual = backtest["actual"]
    cv_baseline = backtest["baseline"]
    cv_coverage = float(
        np.mean(np.abs(backtest["all_actual"] - backtest["all_prediction"]) <= interval_half_width)
    )
    metrics = {
        "model_version": "forecast_v10",
        "production_method": production_method,
        "quality_gate": artifact["quality_gate"],
        "prediction_mode": "residual_ensemble",
        "feature_variant": "with_d1_calendar" if selected["include_d1_calendar"] else "without_d1_calendar",
        "n_features": len(features),
        "weights": artifact["weights"],
        "cv_rmse": round(cv_rmse, 4),
        "cv_mae": round(mae(cv_actual, cv_prediction), 4),
        "cv_r2": round(r2(cv_actual, cv_prediction), 4),
        "cv_mape": round(mape(cv_actual, cv_prediction), 2),
        "cv_persistence_rmse": round(cv_persistence_rmse, 4),
        "cv_skill_vs_persistence": round(cv_skill, 4),
        "cv_directional_accuracy": round(
            directional_accuracy(cv_actual, cv_prediction, cv_baseline), 4
        ),
        "cv_fold_rmse": [round(value, 4) for value in backtest["fold_rmse"]],
        "cv_fold_rmse_std": round(float(np.std(backtest["fold_rmse"])), 4),
        "test_rmse": round(test_rmse, 4),
        "test_mae": round(mae(test_actual, test_prediction), 4),
        "test_r2": round(r2(test_actual, test_prediction), 4),
        "test_persistence_rmse": round(test_persistence_rmse, 4),
        "test_skill_vs_persistence": round(test_skill, 4),
        "test_directional_accuracy": round(
            directional_accuracy(test_actual, test_prediction, test_baseline), 4
        ),
        "interval_method": "conservative_rolling_conformal_90",
        "interval_half_width_90": round(interval_half_width, 4),
        "interval_calibration_quantile": INTERVAL_CALIBRATION_QUANTILE,
        "cv_interval_coverage_90": round(cv_coverage, 4),
        "test_interval_coverage_90": round(test_coverage, 4),
        "n_test": len(test),
        "test_start": test["date"].iloc[0].date().isoformat(),
        "test_end": test["date"].iloc[-1].date().isoformat(),
        "backtest_windows": backtest["fold_ranges"],
        "top_features": feature_importance(
            final_lightgbm, final_extra, final_ridge, features, weights
        ),
    }
    log(
        f"  {target:<20} CV={cv_rmse:.3f} ({cv_skill:+.1%})  "
        f"TEST={test_rmse:.3f} ({test_skill:+.1%})  "
        f"IC90={test_coverage:.1%}  {artifact['quality_gate']}"
    )
    return metrics


def main():
    global REPORT_LINES
    REPORT_LINES = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, contract = load_dataset()
    log("=" * 78)
    log("FORECAST V10 - modelo operativo D+1")
    log(f"Dataset: {len(df)} días | {df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"Contrato: corte {contract['cutoff_hour_local']:02d}:00 {contract['timezone']}")
    log("Métrica entre paréntesis: skill RMSE frente a persistencia")
    log("=" * 78)

    metrics = {}
    for target in TARGETS:
        metrics[f"target_{target}"] = train_target(df, target)

    passed = sum(value["quality_gate"] == "passed" for value in metrics.values())
    artifacts = {}
    for target in TARGETS:
        artifact_path = MODELS_DIR / f"forecast_v10_{target}.joblib"
        artifacts[target] = {
            "file": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        }
    manifest = {
        "model_version": "forecast_v10",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_start": df["date"].min().date().isoformat(),
        "data_end": df["date"].max().date().isoformat(),
        "cutoff_hour_local": contract["cutoff_hour_local"],
        "timezone": contract["timezone"],
        "feature_window": contract["feature_window"],
        "target_window": contract["target_window"],
        "targets_passed": passed,
        "targets_total": len(TARGETS),
        "ready_for_production": passed == len(TARGETS),
        "readiness_scope": "technical_model_gate",
        "recommended_deployment": "controlled_institutional_pilot",
        "external_validation_completed": False,
        "monitoring_required": True,
        "dataset_sha256": file_sha256(DATASET_PATH),
        "artifacts": artifacts,
        "acceptance_rules": {
            "minimum_cv_skill_vs_persistence": MIN_CV_SKILL,
            "minimum_test_skill_vs_persistence": MIN_TEST_SKILL,
            "minimum_test_interval_coverage_90": MIN_TEST_COVERAGE,
            "interval_calibration_quantile": INTERVAL_CALIBRATION_QUANTILE,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log("=" * 78)
    log(f"Control de calidad: {passed}/{len(TARGETS)} objetivos aprobados")
    if passed != len(TARGETS):
        log("Los objetivos no aprobados usarán persistencia de forma automática")
    REPORT_PATH.write_text("\n".join(REPORT_LINES), encoding="utf-8")


REPORT_LINES = []


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}")
        sys.exit(1)
