"""
station_qc.py
=============
Control de calidad de las estaciones antes del análisis causal.

Detecta cambios de régimen en la relación estación/referencia (mediana de las
estaciones exteriores) y posibles contaminaciones por spillover entre la ZBE y
los controles. El resultado se guarda en `models/station_qc.json` y lo consumen
`train_model_v9.py` (excluye pares) y `build_v10.py` (avisa en el dashboard).

Uso:
    python src/analysis/station_qc.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
STATION_PATH = ROOT_DIR / "data" / "processed" / "station_daily.csv"
OUTPUT_PATH = ROOT_DIR / "models" / "station_qc.json"

TREATED = ["PAUL", "FUEROS"]
CONTROLS = ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"]
CONTAMINANTS = ["NO2", "PM10", "PM25"]

# Umbral de cambio de régimen: 30 % respecto a la relación pre-ZBE.
SHIFT_THRESHOLD = 0.30
ROLLING_DAYS = 28
MIN_ROLLING_DAYS = 20

ZBE_DATE = pd.Timestamp("2025-09-01")

# Coordenadas para el chequeo de spillover (mismas que el dashboard).
COORDS = {
    "HUETOS": (42.853846, -2.699907),
    "LANDAZURI": (42.847626, -2.677065),
    "BEATO": (42.849319, -2.675857),
    "PAUL": (42.851130, -2.670824),
    "FUEROS": (42.846270, -2.669415),
    "ZUMABIDE": (42.835437, -2.673657),
}


def haversine_m(a, b):
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))


def load_station_daily() -> pd.DataFrame:
    df = pd.read_csv(STATION_PATH, parse_dates=["date"])
    df.columns = [c.replace("PM2.5", "PM25") for c in df.columns]
    return df.set_index("date").sort_index()


def detect_regime_shifts(df: pd.DataFrame) -> dict:
    flags = {}
    for cont in CONTAMINANTS:
        control_cols = [f"{s}_{cont}" for s in CONTROLS if f"{s}_{cont}" in df.columns]
        if len(control_cols) < 2:
            continue
        reference = df[control_cols].median(axis=1, skipna=True)

        for station in TREATED:
            col = f"{station}_{cont}"
            if col not in df.columns:
                continue
            ratio = (df[col] / reference).replace([np.inf, -np.inf], np.nan)
            baseline = ratio[ratio.index < ZBE_DATE].median()
            rolling = ratio.rolling(ROLLING_DAYS, min_periods=MIN_ROLLING_DAYS).median()
            current = rolling.dropna().iloc[-1] if rolling.notna().any() else np.nan

            if pd.isna(baseline) or pd.isna(current) or baseline <= 0:
                continue

            shift = current / baseline
            flagged = shift >= 1 + SHIFT_THRESHOLD or shift <= 1 - SHIFT_THRESHOLD
            if not flagged:
                continue

            first_shift = None
            monthly = rolling.resample("MS").median().dropna()
            if not monthly.empty:
                if shift >= 1 + SHIFT_THRESHOLD:
                    trigger = baseline * (1 + SHIFT_THRESHOLD)
                    sustain = baseline * (1 + SHIFT_THRESHOLD * 0.5)
                    candidates = monthly[monthly >= trigger]
                    for month in candidates.index:
                        if (monthly.loc[month:] >= sustain).all():
                            first_shift = month
                            break
                else:
                    trigger = baseline * (1 - SHIFT_THRESHOLD)
                    sustain = baseline * (1 - SHIFT_THRESHOLD * 0.5)
                    candidates = monthly[monthly <= trigger]
                    for month in candidates.index:
                        if (monthly.loc[month:] <= sustain).all():
                            first_shift = month
                            break

            flags[col] = {
                "station": station,
                "contaminant": cont,
                "status": "flagged",
                "action": "excluded_from_causal",
                "baseline_ratio": round(float(baseline), 3),
                "current_ratio": round(float(current), 3),
                "shift_factor": round(float(shift), 3),
                "since": first_shift.date().isoformat() if first_shift is not None else None,
                "method": f"rolling_{ROLLING_DAYS}d_median_vs_pre_zbe",
            }
    return flags


def detect_spillover(df: pd.DataFrame) -> dict:
    """Correlación pre-ZBE entre control y tratado más cercano.

    Una correlación muy alta en un radio < 1 km sugiere que el control puede
    estar recibiendo parte del efecto ZBE.
    """
    result = {}
    pre = df[df.index < ZBE_DATE]
    if pre.empty:
        return result

    for cont in CONTAMINANTS:
        for control in CONTROLS:
            ccol = f"{control}_{cont}"
            if ccol not in df.columns:
                continue
            nearest = min(
                TREATED,
                key=lambda s: haversine_m(COORDS[control], COORDS[s]),
            )
            tcol = f"{nearest}_{cont}"
            if tcol not in pre.columns:
                continue
            pair = pre[[ccol, tcol]].dropna()
            if len(pair) < 60:
                continue
            corr = pair[ccol].corr(pair[tcol])
            if pd.isna(corr):
                continue
            distance = haversine_m(COORDS[control], COORDS[nearest])
            result[f"{control}_{cont}"] = {
                "nearest_treated": nearest,
                "distance_m": round(float(distance), 0),
                "pre_correlation": round(float(corr), 3),
                "spillover_risk": bool(corr >= 0.85 and distance < 1000),
            }
    return result


def main():
    if not STATION_PATH.exists():
        raise FileNotFoundError(STATION_PATH)

    df = load_station_daily()
    flags = detect_regime_shifts(df)

    # El ICA de una estación se calcula con sus PM; si el PM está contaminado,
    # el ICA también lo está aunque su ratio no se dispare.
    derived = {}
    for key, info in list(flags.items()):
        if info["contaminant"] in ("PM10", "PM25"):
            ica_key = f"{info['station']}_ICA"
            if ica_key in df.columns and ica_key not in flags:
                derived[ica_key] = {
                    "station": info["station"],
                    "contaminant": "ICA",
                    "status": "flagged",
                    "action": "excluded_from_causal",
                    "since": info["since"],
                    "method": "derived_from_flagged_pm",
                    "derived_from": [key],
                }
    flags.update(derived)

    spillover = detect_spillover(df)

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "date_range": {
            "start": df.index.min().date().isoformat(),
            "end": df.index.max().date().isoformat(),
        },
        "shift_threshold": SHIFT_THRESHOLD,
        "flags": flags,
        "excluded": sorted(flags.keys()),
        "spillover": spillover,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] station_qc.json -> {OUTPUT_PATH}")
    if flags:
        for key, info in flags.items():
            factor = info.get("shift_factor")
            factor_txt = f"x{factor:.2f} vs pre-ZBE " if factor is not None else ""
            print(f"  [FLAG] {key}: {factor_txt}(desde {info.get('since')})")
    else:
        print("  Sin cambios de régimen detectados.")
    risky = [k for k, v in spillover.items() if v["spillover_risk"]]
    if risky:
        print(f"  [WARN] Posible spillover en: {', '.join(risky)}")


if __name__ == "__main__":
    main()
