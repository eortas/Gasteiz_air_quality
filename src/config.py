"""
config.py
=========
Constantes centralizadas del proyecto Vitoria Air Quality.

Todos los módulos deben importar de aquí en lugar de definir
sus propias copias de ZBE_DATE, estaciones, targets, etc.

Para importar desde cualquier módulo del proyecto:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))  # o la ruta adecuada a src/
    from config import ZBE_DATE, ZBE_STATIONS, ...
"""

import pandas as pd

# ─── FECHA ZBE ────────────────────────────────────────────────────────────────
# Fecha de implementación de la Zona de Bajas Emisiones en Vitoria-Gasteiz
ZBE_DATE_STR = "2025-09-01"
ZBE_DATE     = pd.Timestamp(ZBE_DATE_STR, tz="UTC")
ZBE_DATE_NAIVE = pd.Timestamp(ZBE_DATE_STR)  # Sin timezone, para comparar con fechas naive

# ─── ESTACIONES ───────────────────────────────────────────────────────────────
ZBE_STATIONS = ["PAUL", "FUEROS"]
OUT_STATIONS = ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"]

# ─── CONTAMINANTES ────────────────────────────────────────────────────────────
CONTAMINANTS       = ["NO2", "PM10", "PM2.5", "ICA"]
TARGETS_CAUSALES   = ["NO2", "PM10", "PM2.5"]  # Para análisis causal (sin ICA)

# ─── GEOGRAFÍA ────────────────────────────────────────────────────────────────
LATITUDE      = 42.8467
LONGITUDE     = -2.6716

# ─── CALEFACCIÓN ──────────────────────────────────────────────────────────────
HDD_BASE_TEMP = 15.0  # Temperatura de encendido estándar España

# ─── HDD FEATURES REQUERIDAS ─────────────────────────────────────────────────
# Lista única de features HDD que se fuerzan en el modelo
HDD_FEATURES_REQUIRED = [
    "HDD", "HDD_acum_7d", "HDD_acum_14d", "HDD_lag_1d", "HDD_lag_7d",
    "dia_muy_frio", "es_invierno_estricto", "domingo_invierno", "es_domingo",
]

# ─── ICA: SUBÍNDICES NORMALIZADOS (CAQI EUROPEO) ─────────────────────────────
# Umbrales de concentración para cada contaminante (µg/m³, medias diarias)
# y niveles de subíndice correspondientes (0-100 escala CAQI)
ICA_THRESHOLDS = {
    "NO2":   [0, 50, 100, 200, 400],
    "PM10":  [0, 25,  50, 90, 180],
    "PM2.5": [0, 15,  30, 55, 110],
}
ICA_LEVELS = [0, 25, 50, 75, 100]  # Muy Bueno, Bueno, Regular, Pobre, Muy Pobre


def compute_ica_subindex(contaminant: str, value: float) -> float:
    """
    Calcula el subíndice ICA (CAQI europeo) para un contaminante y valor dado.

    Interpola linealmente entre los umbrales de concentración para obtener
    un valor entre 0 (aire limpio) y 100+ (muy contaminado).

    Args:
        contaminant: "NO2", "PM10" o "PM2.5"
        value: concentración en µg/m³

    Returns:
        Subíndice ICA (0-100+)
    """
    if pd.isna(value) or value < 0:
        return 0.0

    # Normalizar el nombre del contaminante
    cont_key = contaminant.replace("PM25", "PM2.5")
    if cont_key not in ICA_THRESHOLDS:
        return float(value)  # Devolver el valor crudo si no hay umbrales

    thresholds = ICA_THRESHOLDS[cont_key]
    levels = ICA_LEVELS

    # Interpolar linealmente entre umbrales
    for i in range(len(thresholds) - 1):
        if value <= thresholds[i + 1]:
            ratio = (value - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
            return levels[i] + ratio * (levels[i + 1] - levels[i])

    # Si supera el último umbral, extrapolar
    ratio = (value - thresholds[-1]) / (thresholds[-1] - thresholds[-2])
    return levels[-1] + ratio * (levels[-1] - levels[-2])
