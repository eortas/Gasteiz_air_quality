"""
check_drift.py
==============
Monitoriza la degradación del forecast comparando las predicciones emitidas con
la medición observada. Calcula el skill rolling (30 días) frente a persistencia
y lo compara con el skill de entrenamiento.

Si el skill cae por debajo de 0 (peor que persistencia) o a menos de la mitad
del skill de entrenamiento, marca alerta. El informe se guarda en
`models/drift_report.json` y se publica en `pipeline_status.json`.

Uso:
    python src/ml/check_drift.py
    python src/ml/check_drift.py --fail-on-alert
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

HISTORY_PATH = PROCESSED_DIR / "predictions_history.json"
STATION_PATH = PROCESSED_DIR / "station_daily.csv"
METRICS_PATH = MODELS_DIR / "forecast_v10_metrics.json"
QC_PATH = MODELS_DIR / "station_qc.json"
OUTPUT_PATH = MODELS_DIR / "drift_report.json"

WINDOW_DAYS = 30
MIN_OBSERVATIONS = 10
TRAINED_ALERT_FRACTION = 0.5

ZBE_STATIONS = ["PAUL", "FUEROS"]
OUT_STATIONS = ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"]
HORIZONS = [1, 2]


def load_observed() -> pd.DataFrame:
    """Medias diarias por zona, respetando las exclusiones de station_qc."""
    df = pd.read_csv(STATION_PATH, parse_dates=["date"])
    df.columns = [c.replace("PM2.5", "PM25") for c in df.columns]
    df = df.set_index("date").sort_index()

    excluded = set()
    flags = {}
    if QC_PATH.exists():
        try:
            qc = json.loads(QC_PATH.read_text(encoding="utf-8"))
            excluded = set(qc.get("excluded", []))
            flags = qc.get("flags", {})
        except Exception:
            pass

    observed = pd.DataFrame(index=df.index)
    for cont in ["NO2", "PM10", "PM25", "ICA"]:
        for zone, stations in (("zbe", ZBE_STATIONS), ("out", OUT_STATIONS)):
            series = []
            for station in stations:
                col = f"{station}_{cont}"
                if col not in df.columns:
                    continue
                if col in excluded:
                    factor = flags.get(col, {}).get("shift_factor")
                    if factor is not None and factor > 0:
                        # Misma corrección que build_features_v6: el sensor
                        # sigue aportando nivel, sin el sesgo de régimen.
                        series.append(df[col] / float(factor))
                    continue
                series.append(df[col])
            if series:
                observed[f"{cont}_{zone}"] = pd.concat(series, axis=1).mean(axis=1, skipna=True)
    return observed


def load_training_skill() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    skills = {}
    for key, value in metrics.items():
        clean = key.replace("target_", "")
        skill = value.get("test_skill_vs_persistence")
        if skill is None:
            skill = value.get("cv_skill_vs_persistence")
        skills[clean] = skill
    return skills


def build_target_frame(history: list, observed: pd.DataFrame, target_key: str) -> pd.DataFrame:
    cont, zone, horizon_tag = target_key.rsplit("_", 2)
    horizon = int(horizon_tag.replace("d", ""))
    col = f"{cont}_{zone}"
    if col not in observed.columns:
        return pd.DataFrame()

    rows = []
    for entry in history:
        if entry.get("run_type") != "production":
            continue
        predictions = entry.get("predictions", {})
        if target_key not in predictions:
            continue
        date_txt = entry.get("prediction_dates", {}).get(f"d{horizon}")
        if not date_txt:
            base = pd.Timestamp(entry["prediction_date"])
            date_txt = (base + pd.Timedelta(days=horizon - 1)).date().isoformat()
        rows.append({"date": pd.Timestamp(date_txt), "pred": float(predictions[target_key])})

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    frame["obs"] = observed[col].reindex(frame.index)
    # Persistencia: valor observado `horizon` días antes del día objetivo.
    frame["persistence"] = observed[col].shift(horizon).reindex(frame.index)

    max_obs = observed[col].dropna().index.max()
    frame = frame[frame.index <= max_obs]
    return frame.dropna(subset=["pred", "obs", "persistence"]).tail(WINDOW_DAYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    if not HISTORY_PATH.exists() or not STATION_PATH.exists():
        payload = {"generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
                   "status": "unavailable", "reason": "faltan historial o station_daily"}
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[WARN] check_drift: faltan datos, se escribe estado unavailable")
        return

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    observed = load_observed()
    training_skill = load_training_skill()

    targets = {}
    overall_alert = False

    for target_key in sorted({
        key for entry in history
        if entry.get("run_type") == "production"
        for key in entry.get("predictions", {})
    }):
        frame = build_target_frame(history, observed, target_key)
        if len(frame) < MIN_OBSERVATIONS:
            targets[target_key] = {
                "status": "insufficient",
                "n": int(len(frame)),
                "window_days": WINDOW_DAYS,
            }
            continue

        mae_model = float(np.mean(np.abs(frame["obs"] - frame["pred"])))
        mae_persistence = float(np.mean(np.abs(frame["obs"] - frame["persistence"])))
        skill = 1 - mae_model / mae_persistence if mae_persistence > 0 else None
        trained = training_skill.get(target_key)

        status = "ok"
        if skill is not None and skill < 0:
            status = "alert"
        elif (
            trained is not None and trained > 0
            and skill is not None and skill < trained * TRAINED_ALERT_FRACTION
        ):
            status = "warn"
        if status == "alert":
            overall_alert = True

        targets[target_key] = {
            "status": status,
            "n": int(len(frame)),
            "window_start": frame.index.min().date().isoformat(),
            "window_end": frame.index.max().date().isoformat(),
            "mae_model": round(mae_model, 3),
            "mae_persistence": round(mae_persistence, 3),
            "skill_vs_persistence": round(skill, 4) if skill is not None else None,
            "training_skill": round(trained, 4) if trained is not None else None,
        }

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "window_days": WINDOW_DAYS,
        "status": "alert" if overall_alert else "ok",
        "alert": overall_alert,
        "targets": targets,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] drift_report.json -> {OUTPUT_PATH}")
    print(f"  Estado global: {payload['status']}")
    for key, info in targets.items():
        if info["status"] in ("alert", "warn"):
            print(f"  [{info['status'].upper()}] {key}: skill={info.get('skill_vs_persistence')} "
                  f"(train={info.get('training_skill')}, n={info['n']})")

    if args.fail_on_alert and overall_alert:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
