"""
build_features.py  — v6
==================
Construye el dataset de entrenamiento para el modelo ML multioutput.

CAMBIOS v6 (hipótesis calderas / ZBE):
  - Nueva variable HDD (Heating Degree Days, base 15°C): captura demanda de
    calefacción residencial, principal hipótesis para explicar el aumento de
    NO2 dentro de la ZBE pese a reducción de tráfico.
  - Variables derivadas: temp_range (amplitud térmica), dias_frio (HDD > 10),
    demanda_acumulada_7d (suma HDD últimos 7 días).
  - Features temporales ampliadas: es_invierno_estricto, es_verano,
    es_domingo (para Prueba del Domingo).
  - HDD entra explícitamente en las features del modelo (no se descarta
    como raw target).

Targets (predicción diaria, separados por grupo ZBE):
  - NO2_zbe, PM10_zbe, PM2.5_zbe, ICA_zbe  → estación PAUL (dentro ZBE)
  - NO2_out, PM10_out, PM2.5_out, ICA_out  → media resto estaciones (fuera ZBE)
  → targets: target_NO2_zbe_d1 ... target_ICA_out_d3

Estaciones:
  - PAUL, LANDAZURI  → grupo ZBE (dentro de la Zona de Bajas Emisiones)
  - BEATO, FUEROS, HUETOS, ZUMABIDE → grupo OUT (fuera ZBE)

Uso:
    python build_features.py
    python build_features.py --with-forecast
    python build_features.py --csv
    python build_features.py --days 180
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent.parent
AIR_DIR       = ROOT_DIR / "data" / "raw" / "air"
TRAFFIC_DIR   = ROOT_DIR / "data" / "raw" / "traffic"
WEATHER_DIR   = ROOT_DIR / "data" / "raw" / "weather"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ZBE_STATIONS  = ["PAUL", "BEATO", "FUEROS"]
OUT_STATIONS  = ["LANDAZURI", "HUETOS", "ZUMABIDE"]
CONTAMINANTS  = ["NO2", "PM10", "PM2.5", "ICA"]
TARGETS       = [f"{c}_{g}" for c in CONTAMINANTS for g in ["zbe", "out"]]
HORIZON_DAYS  = 3
LAG_DAYS      = [1, 2, 3, 7, 14, 30]
ROLLING_WINS  = [3, 7, 14, 30]
ZBE_DATE      = pd.Timestamp("2025-09-01", tz="UTC")
LATITUDE      = 42.8467
LONGITUDE     = -2.6716

# Base para HDD — temperatura de encendido estándar España
HDD_BASE_TEMP = 15.0

FORECAST_VARS = [
    "temperature_2m", "precipitation", "rain", "snowfall",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "cloud_cover", "relative_humidity_2m", "boundary_layer_height",
    "weather_code", "sunshine_duration",
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def log(msg=""):
    print(msg)

def section(title):
    log(); log("═" * 65); log(f"  {title}"); log("═" * 65)

def subsection(title):
    log(); log(f"── {title} " + "─" * max(0, 58 - len(title)))

def load_csvs(directory: Path, pattern: str, ts_col: str) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Sin archivos '{pattern}' en {directory}")
    frames = []
    for f in files:
        log(f"  Leyendo {f.name} ({f.stat().st_size/1024/1024:.1f} MB)...")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, format="mixed", errors="coerce")
    return df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)


# ─── 1. AIRE ──────────────────────────────────────────────────────────────────
def load_air_daily() -> pd.DataFrame:
    section("1. Calidad del aire → media diaria por grupo ZBE / OUT")

    df = load_csvs(AIR_DIR, "kunak_[0-9]*.csv", "timestamp")
    log(f"  Filas brutas    : {len(df):,}")
    log(f"  Estaciones      : {sorted(df['estacion'].unique())}")

    df["date"] = df["timestamp"].dt.floor("D")
    frames = []

    df_zbe = df[df["estacion"].isin(ZBE_STATIONS)]
    daily_zbe = (df_zbe.groupby(["date", "contaminante"])["valor"]
                       .mean().unstack("contaminante").reset_index())
    daily_zbe.columns = (["date"] +
                         [f"{c}_zbe" for c in daily_zbe.columns[1:]])
    frames.append(daily_zbe)

    df_out = df[df["estacion"].isin(OUT_STATIONS)]
    daily_out = (df_out.groupby(["date", "contaminante"])["valor"]
                       .mean().unstack("contaminante").reset_index())
    daily_out.columns = (["date"] +
                         [f"{c}_out" for c in daily_out.columns[1:]])
    frames.append(daily_out)

    daily = frames[0].merge(frames[1], on="date", how="outer")

    for t in TARGETS:
        if t not in daily.columns:
            log(f"  ⚠️  Target '{t}' no encontrado — rellenando con NaN")
            daily[t] = np.nan

    daily["date"] = pd.to_datetime(daily["date"], utc=True)

    # Verificación pre/post ZBE
    subsection("Verificación pre/post ZBE por grupo")
    pre  = daily[daily["date"] <  ZBE_DATE]
    post = daily[daily["date"] >= ZBE_DATE]
    log(f"  {'Contaminante':<15} {'pre':>10} {'post':>10} {'cambio':>8}")
    log(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8}")
    for cont in CONTAMINANTS:
        for grp in ["zbe", "out"]:
            col = f"{cont}_{grp}"
            if col in daily.columns:
                pre_m  = pre[col].mean()
                post_m = post[col].mean()
                chg    = (post_m - pre_m) / pre_m * 100 if pre_m > 0 else 0
                arrow  = "↓" if chg < 0 else "↑"
                log(f"  {col:<15} {pre_m:>10.2f} {post_m:>10.2f} {chg:>+7.1f}% {arrow}")

    return daily.sort_values("date").reset_index(drop=True)


# ─── 2. TRÁFICO ───────────────────────────────────────────────────────────────
def load_traffic_daily() -> pd.DataFrame:
    section("2. Tráfico → agregado diario + proxy futuro")

    df = load_csvs(TRAFFIC_DIR, "trafico_[0-9]*.csv", "start_date")
    df["date"] = df["start_date"].dt.floor("D")

    daily = df.groupby("date").agg(
        traffic_volume   =("volume",    "sum"),
        traffic_occupancy=("occupancy", "mean"),
        traffic_load     =("load",      "mean"),
        traffic_sensors  =("code",      "nunique"),
    ).reset_index()

    log(f"  Días con datos de tráfico: {len(daily):,}")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily["dow"]         = daily["date"].dt.dayofweek
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["month"]       = daily["date"].dt.month

    subsection("Calculando proxy de tráfico futuro (DOW ± 15d + mes anterior)")
    traffic_lookup = {}
    for dow in range(7):
        for doy in range(1, 366):
            mask = (
                (daily["dow"] == dow) &
                (((daily["day_of_year"] - doy).abs() <= 15) |
                 ((daily["day_of_year"] - doy + 365).abs() <= 15) |
                 ((daily["day_of_year"] - doy - 365).abs() <= 15))
            )
            subset = daily[mask]
            if len(subset) > 0:
                traffic_lookup[(dow, doy)] = {
                    "volume":    subset["traffic_volume"].mean(),
                    "occupancy": subset["traffic_occupancy"].mean(),
                }

    monthly_base = daily.groupby("month").agg(
        monthly_volume   =("traffic_volume",    "mean"),
        monthly_occupancy=("traffic_occupancy", "mean"),
    ).reset_index()

    for h in range(1, HORIZON_DAYS + 1):
        daily[f"exp_traffic_volume_d{h}"]    = np.nan
        daily[f"exp_traffic_occupancy_d{h}"] = np.nan

    for idx, row in daily.iterrows():
        for h in range(1, HORIZON_DAYS + 1):
            future_date = row["date"] + pd.Timedelta(days=h)
            fdow = future_date.dayofweek
            fdoy = future_date.dayofyear
            fmon = future_date.month
            lookup_val  = traffic_lookup.get((fdow, fdoy), {})
            monthly_val = monthly_base[monthly_base["month"] == fmon]
            vol = lookup_val.get("volume", np.nan)
            occ = lookup_val.get("occupancy", np.nan)
            if not monthly_val.empty:
                m_vol = monthly_val["monthly_volume"].iloc[0]
                m_occ = monthly_val["monthly_occupancy"].iloc[0]
                if not np.isnan(vol):
                    vol = 0.7 * vol + 0.3 * m_vol
                    occ = 0.7 * occ + 0.3 * m_occ
                else:
                    vol, occ = m_vol, m_occ
            daily.at[idx, f"exp_traffic_volume_d{h}"]    = vol
            daily.at[idx, f"exp_traffic_occupancy_d{h}"] = occ

    import json
    lookup_serial = {f"{k[0]}_{k[1]}": v for k, v in traffic_lookup.items()}
    (PROCESSED_DIR / "traffic_lookup.json").write_text(
        json.dumps(lookup_serial), encoding="utf-8"
    )

    daily = daily.drop(columns=["dow", "day_of_year", "month"])
    return daily.sort_values("date").reset_index(drop=True)


# ─── 3. METEOROLOGÍA ─────────────────────────────────────────────────────────
def load_weather_daily() -> pd.DataFrame:
    section("3. Meteorología → agregado diario + variables de demanda térmica")

    df = load_csvs(WEATHER_DIR, "weather_[0-9]*.csv", "timestamp")
    df["date"] = df["timestamp"].dt.floor("D")

    agg_dict = {}
    for col in FORECAST_VARS:
        if col in df.columns:
            if col in ["precipitation", "rain", "snowfall", "sunshine_duration"]:
                agg_dict[col] = "sum"
            elif col == "wind_direction_10m":
                agg_dict[col] = lambda x: (
                    np.degrees(np.arctan2(
                        np.sin(np.radians(x)).mean(),
                        np.cos(np.radians(x)).mean()
                    )) % 360
                )
            else:
                agg_dict[col] = "mean"

    # También necesitamos temp max y min para amplitud térmica
    for extra in ["temperature_2m_max", "temperature_2m_min"]:
        if extra in df.columns:
            agg_dict[extra] = "max" if "max" in extra else "min"

    weather = df.groupby("date").agg(agg_dict).reset_index()

    # ── VARIABLES DE DEMANDA TÉRMICA (hipótesis calderas) ─────────────────
    # Si no vienen precalculadas las columnas daily_*, las calculamos nosotros
    if "daily_temperature_2m_max" in df.columns:
        # Los CSV ya traen columnas daily_ precalculadas — usarlas directamente
        daily_agg = (df.groupby("date")[
            ["daily_temperature_2m_mean",
             "daily_temperature_2m_max",
             "daily_temperature_2m_min"]]
            .first()
            .reset_index()
            .rename(columns={
                "daily_temperature_2m_mean": "temp_media_dia",
                "daily_temperature_2m_max":  "temp_max_dia",
                "daily_temperature_2m_min":  "temp_min_dia",
            })
        )
        weather = weather.merge(daily_agg, on="date", how="left")
        temp_col = "temp_media_dia"
    else:
        # Calcular desde temperatura horaria
        temp_agg = df.groupby("date")["temperature_2m"].agg(
            temp_media_dia="mean",
            temp_max_dia="max",
            temp_min_dia="min"
        ).reset_index()
        weather = weather.merge(temp_agg, on="date", how="left")
        temp_col = "temp_media_dia"

    # HDD: Heating Degree Days base 15°C
    # Cuanto más alto, más frío el día → más calderas encendidas
    weather["HDD"] = (HDD_BASE_TEMP - weather[temp_col]).clip(lower=0)

    # Amplitud térmica diaria (alta amplitud → irradiación nocturna fuerte → más frío)
    weather["temp_range"] = weather["temp_max_dia"] - weather["temp_min_dia"]

    # Flag día muy frío (HDD > 10 equivale a temp media < 5°C)
    weather["dia_muy_frio"] = (weather["HDD"] > 10).astype(int)

    # Demanda acumulada 7 días (representa carga térmica reciente del edificio)
    # Se calculará como rolling en add_lags_and_rolling, aquí dejamos HDD base

    if "boundary_layer_height" in weather.columns:
        weather["boundary_layer_height"] = (
            weather["boundary_layer_height"]
            .interpolate(method="linear").bfill().ffill()
        )

    if "wind_speed_10m" in weather.columns:
        rad = np.deg2rad(weather["wind_direction_10m"])
        weather["wind_u"] = -weather["wind_speed_10m"] * np.sin(rad)
        weather["wind_v"] = -weather["wind_speed_10m"] * np.cos(rad)

    weather["date"] = pd.to_datetime(weather["date"], utc=True)

    log(f"  Días únicos: {len(weather):,}")
    log(f"  HDD medio global   : {weather['HDD'].mean():.2f}")
    log(f"  HDD medio invierno : {weather[weather['date'].dt.month.isin([11,12,1,2])]['HDD'].mean():.2f}")
    log(f"  Días muy fríos (HDD>10): {weather['dia_muy_frio'].sum()}")

    # ── DIAGNÓSTICO HDD PRE vs POST ZBE ───────────────────────────────────
    subsection("Diagnóstico HDD: ¿fue el invierno post-ZBE más frío?")
    pre_hdd  = weather[weather["date"] <  ZBE_DATE]["HDD"]
    post_hdd = weather[weather["date"] >= ZBE_DATE]["HDD"]
    log(f"  HDD medio PRE-ZBE  : {pre_hdd.mean():.2f}")
    log(f"  HDD medio POST-ZBE : {post_hdd.mean():.2f}")
    delta = post_hdd.mean() - pre_hdd.mean()
    if delta > 0.5:
        log(f"  ⚠️  Invierno post-ZBE MÁS FRÍO (+{delta:.2f} HDD) — puede enmascarar mejora NO2")
    elif delta < -0.5:
        log(f"  ✅ Invierno post-ZBE MÁS CÁLIDO ({delta:.2f} HDD) — si NO2 sube, es efecto estructural")
    else:
        log(f"  ≈  Inviernos comparables (Δ={delta:+.2f} HDD) — comparación directa válida")

    return weather.sort_values("date").reset_index(drop=True)


# ─── 4. PRONÓSTICO OPEN-METEO ────────────────────────────────────────────────
def fetch_forecast() -> pd.DataFrame:
    section("4. Descargando pronóstico Open-Meteo (7 días)")
    try:
        import requests
    except ImportError:
        log("  ⚠️  pip install requests — saltando pronóstico")
        return pd.DataFrame()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly":        ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "rain", "snowfall", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", "cloud_cover", "boundary_layer_height",
            "sunshine_duration", "weather_code",
        ]),
        "forecast_days": 8,
        "timezone":      "UTC",
        "wind_speed_unit": "ms",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  ⚠️  Error en Open-Meteo: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data.get("hourly", {}))
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df["date"]      = df["timestamp"].dt.floor("D")

    agg_dict = {}
    for col in FORECAST_VARS:
        if col in df.columns:
            if col in ["precipitation", "rain", "snowfall", "sunshine_duration"]:
                agg_dict[col] = "sum"
            elif col == "wind_direction_10m":
                agg_dict[col] = lambda x: (
                    np.degrees(np.arctan2(
                        np.sin(np.radians(x)).mean(),
                        np.cos(np.radians(x)).mean()
                    )) % 360
                )
            else:
                agg_dict[col] = "mean"

    daily_fc = df.groupby("date").agg(agg_dict).reset_index()
    daily_fc["date"] = pd.to_datetime(daily_fc["date"], utc=True)

    # HDD del pronóstico
    daily_fc["HDD"] = (HDD_BASE_TEMP - daily_fc["temperature_2m"]).clip(lower=0)

    if "wind_speed_10m" in daily_fc.columns:
        rad = np.deg2rad(daily_fc["wind_direction_10m"])
        daily_fc["wind_u"] = -daily_fc["wind_speed_10m"] * np.sin(rad)
        daily_fc["wind_v"] = -daily_fc["wind_speed_10m"] * np.cos(rad)

    today = pd.to_datetime(pd.Timestamp.now(tz="Europe/Madrid").date()).tz_localize("UTC")
    fc_flat = {}
    days_found = 0
    for h in range(1, HORIZON_DAYS + 1):
        target_date = today + pd.Timedelta(days=h)
        row_fc = daily_fc[daily_fc["date"] == target_date]
        if row_fc.empty:
            continue
        for col in row_fc.columns:
            if col != "date":
                fc_flat[f"fc_{col}_d{h}"] = row_fc[col].iloc[0]
        days_found += 1

    log(f"  Pronóstico descargado: {days_found} días — {len(fc_flat)} features")
    return pd.DataFrame([fc_flat]) if fc_flat else pd.DataFrame()


# ─── 5. MERGE ─────────────────────────────────────────────────────────────────
def merge_daily(air, traffic, weather) -> pd.DataFrame:
    section("5. Merge diario de las 3 fuentes")

    for df_input in [air, traffic, weather]:
        df_input["date"] = pd.to_datetime(df_input["date"], utc=True)

    # ─── 4. MERGE FINAL ───────────────────────────────────────────────────────────
    section("4. Unificando datasets (outer join)")
    df = (air
          .merge(traffic, on="date", how="outer")
          .merge(weather, on="date", how="outer"))
          
    # Colapsar filas duplicadas para la misma fecha (ej. diferencias microscópicas de merge)
    df = df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)

    # === REPARTO FINAL: PREVENCIÓN DE CORTES POR LAGS DE KUNAK ===
    # Añadimos fila de hoy explícita si falta (outer join puede no traerla si nadie la tiene)
    today_dt = pd.to_datetime(pd.Timestamp.now(tz="Europe/Madrid").date()).tz_localize("UTC")
    if not (df["date"] == today_dt).any():
        log(f"  Añadiendo fila de hoy {today_dt.strftime('%Y-%m-%d')} para predicciones...")
        today_df = pd.DataFrame({"date": [today_dt]})
        df = pd.concat([df, today_df], ignore_index=True)

    log(f"  Días tras merge: {len(df):,}")
    log(f"  Rango: {df['date'].min().date()} → {df['date'].max().date()}")
    return df.sort_values("date").reset_index(drop=True)


# ─── 6. FEATURES TEMPORALES ───────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    section("6. Features temporales")

    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_workday"]   = (df["day_of_week"] < 5).astype(int)
    df["season"]       = df["month"].map({
        12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4
    })

    # Codificación cíclica
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ZBE flags
    df["is_post_zbe"]    = (df["date"] >= ZBE_DATE).astype(int)
    df["days_since_zbe"] = (df["date"] - ZBE_DATE).dt.days.clip(lower=0)

    # ── NUEVAS: variables para análisis de calderas ────────────────────────
    # Invierno estricto (nov-feb): máxima demanda de calefacción residencial
    df["es_invierno_estricto"] = df["month"].isin([11, 12, 1, 2]).astype(int)

    # Verano (jun-sep): calderas apagadas → aísla efecto tráfico puro
    # (útil cuando haya datos post-ZBE de verano, a partir de jun-2026)
    df["es_verano"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Domingo: tráfico mínimo pero calderas encendidas igual
    # Prueba del Domingo: si NO2 no baja los domingos post-ZBE en invierno
    # → el problema es estructural (calderas), no de tráfico puntual
    df["es_domingo"] = (df["day_of_week"] == 6).astype(int)

    # Interacción clave: domingo de invierno (tráfico mínimo + calderas máximo)
    df["domingo_invierno"] = (df["es_domingo"] & df["es_invierno_estricto"]).astype(int)

    log(f"  Features temporales base: dow, month, season, is_weekend, is_workday")
    log(f"  Features ZBE: is_post_zbe, days_since_zbe")
    log(f"  Features calderas: es_invierno_estricto, es_verano, es_domingo, domingo_invierno")
    log(f"  Codificación cíclica: sin/cos para dow, month, doy")
    return df


# ─── 7. LAGS Y ROLLING ────────────────────────────────────────────────────────
def add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    section("7. Lags y rolling stats diarios")

    df = df.set_index("date").sort_index()

    for target in TARGETS:
        if target not in df.columns:
            continue

        for lag in LAG_DAYS:
            df[f"{target}_lag_{lag}d"] = df[target].shift(lag)

        for win in ROLLING_WINS:
            df[f"{target}_roll_mean_{win}d"] = (df[target].shift(1)
                                                 .rolling(win, min_periods=2).mean())
            df[f"{target}_roll_std_{win}d"]  = (df[target].shift(1)
                                                 .rolling(win, min_periods=2).std()
                                                 .fillna(0))
            df[f"{target}_roll_max_{win}d"]  = (df[target].shift(1)
                                                 .rolling(win, min_periods=2).max())

        df[f"{target}_diff_7d"]  = df[target].shift(1) - df[target].shift(8)
        df[f"{target}_diff_30d"] = df[target].shift(1) - df[target].shift(31)

        log(f"  {target}: {len(LAG_DAYS)} lags + {len(ROLLING_WINS)*3} rolling + 2 diffs")

    # Lags de tráfico
    for lag in [1, 2, 7, 14]:
        df[f"traffic_volume_lag_{lag}d"]    = df["traffic_volume"].shift(lag)
        df[f"traffic_occupancy_lag_{lag}d"] = df["traffic_occupancy"].shift(lag)

    # ── ROLLING DE HDD (demanda acumulada de calefacción) ─────────────────
    # Suma HDD últimos 7 y 14 días → representa la carga térmica acumulada
    # del edificio: si llevas 7 días fríos, las calderas llevan 7 días a tope
    if "HDD" in df.columns:
        df["HDD_acum_7d"]  = df["HDD"].shift(1).rolling(7,  min_periods=2).sum()
        df["HDD_acum_14d"] = df["HDD"].shift(1).rolling(14, min_periods=2).sum()
        df["HDD_lag_1d"]   = df["HDD"].shift(1)
        df["HDD_lag_7d"]   = df["HDD"].shift(7)
        log(f"  HDD: rolling acumulado 7d y 14d + lags 1d y 7d")

    return df.reset_index()


# ─── 8. TARGETS FUTUROS ───────────────────────────────────────────────────────
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    section("8. Construyendo targets futuros")

    df = df.set_index("date").sort_index()

    for target in TARGETS:
        if target not in df.columns:
            continue
        for h in range(1, HORIZON_DAYS + 1):
            df[f"target_{target}_d{h}"] = df[target].shift(-h)

    target_cols = [c for c in df.columns if c.startswith("target_")]
    log(f"  Targets: {len(target_cols)} columnas")
    log(f"  ({len(TARGETS)} series × {HORIZON_DAYS} días = {len(TARGETS)*HORIZON_DAYS})")

    return df.reset_index()


# ─── 9. PRONÓSTICO COMO FEATURES ─────────────────────────────────────────────
def add_forecast_features(df: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    if fc.empty:
        log("  Sin pronóstico — usando proxy histórico en training")
        weather_base_cols = [c for c in df.columns if any(x in c for x in
            ["temperature", "precipitation", "wind", "cloud",
             "humidity", "boundary", "sunshine", "HDD"])
            and "_lag_" not in c and "_roll_" not in c
            and "_diff_" not in c and "_acum_" not in c
            and not c.startswith("fc_")]
        df = df.set_index("date").sort_index()
        for h in range(1, HORIZON_DAYS + 1):
            for col in weather_base_cols:
                df[f"fc_{col}_d{h}"] = df[col].shift(-h)
        return df.reset_index()

    section("9. Añadiendo features del pronóstico Open-Meteo")
    weather_base_cols = [c for c in df.columns if any(x in c for x in
        ["temperature", "precipitation", "wind", "cloud",
         "humidity", "boundary", "sunshine", "HDD"])
        and "_lag_" not in c and "_roll_" not in c
        and "_diff_" not in c and "_acum_" not in c
        and not c.startswith("fc_")]

    df = df.set_index("date").sort_index()
    for h in range(1, HORIZON_DAYS + 1):
        for col in weather_base_cols:
            df[f"fc_{col}_d{h}"] = df[col].shift(-h)

    for col in fc.columns:
        if col in df.columns:
            df.loc[df.index[-1], col] = fc[col].iloc[0]

    log(f"  Features de pronóstico añadidas (incluyendo HDD pronosticado)")
    return df.reset_index()


# ─── 10. ANÁLISIS PRUEBA DEL DOMINGO ─────────────────────────────────────────
def prueba_del_domingo(df: pd.DataFrame) -> None:
    """
    Diagnóstico inline: compara NO2_zbe los domingos de invierno
    pre vs post ZBE. Si NO2 no baja, el problema son las calderas.
    """
    section("10b. DIAGNÓSTICO — Prueba del Domingo (calderas vs tráfico)")

    if "NO2_zbe" not in df.columns:
        log("  ⚠️  NO2_zbe no disponible")
        return

    df["date_ts"] = pd.to_datetime(df["date"], utc=True)
    inv = df[df["date_ts"].dt.month.isin([11, 12, 1, 2])].copy()
    inv_dom = inv[inv["date_ts"].dt.dayofweek == 6]

    pre_dom  = inv_dom[inv_dom["date_ts"] <  ZBE_DATE]["NO2_zbe"]
    post_dom = inv_dom[inv_dom["date_ts"] >= ZBE_DATE]["NO2_zbe"]

    if pre_dom.empty or post_dom.empty:
        log("  ⚠️  Datos insuficientes para la Prueba del Domingo")
        return

    delta = post_dom.mean() - pre_dom.mean()
    pct   = delta / pre_dom.mean() * 100

    log(f"  Domingos de invierno PRE-ZBE  (n={len(pre_dom)}): NO2_zbe = {pre_dom.mean():.2f} µg/m³")
    log(f"  Domingos de invierno POST-ZBE (n={len(post_dom)}): NO2_zbe = {post_dom.mean():.2f} µg/m³")
    log(f"  Δ = {delta:+.2f} µg/m³  ({pct:+.1f}%)")

    if pct > 5:
        log(f"  🏠 RESULTADO: NO2 SUBE los domingos de invierno post-ZBE.")
        log(f"     → Evidencia de que las CALDERAS son la fuente dominante.")
        log(f"     → La ZBE puede estar funcionando contra el tráfico, pero")
        log(f"       el efecto queda enmascarado por la combustión residencial.")
    elif pct < -5:
        log(f"  🚗 RESULTADO: NO2 BAJA los domingos de invierno post-ZBE.")
        log(f"     → La ZBE tiene efecto incluso en días de tráfico mínimo.")
        log(f"     → Posible mejora en calderas o cambio de combustible.")
    else:
        log(f"  ≈  RESULTADO: Sin cambio significativo — datos insuficientes")
        log(f"     o efectos contrapuestos que se anulan.")

    # También: días muy fríos (HDD > 10)
    if "HDD" in df.columns:
        log(f"")
        frio = df[df["HDD"] > 10].copy()
        frio["date_ts"] = pd.to_datetime(frio["date"], utc=True)
        pre_frio  = frio[frio["date_ts"] <  ZBE_DATE]["NO2_zbe"]
        post_frio = frio[frio["date_ts"] >= ZBE_DATE]["NO2_zbe"]
        if not pre_frio.empty and not post_frio.empty:
            delta_f = post_frio.mean() - pre_frio.mean()
            pct_f   = delta_f / pre_frio.mean() * 100
            log(f"  Días muy fríos (HDD>10) PRE-ZBE  (n={len(pre_frio)}): {pre_frio.mean():.2f} µg/m³")
            log(f"  Días muy fríos (HDD>10) POST-ZBE (n={len(post_frio)}): {post_frio.mean():.2f} µg/m³")
            log(f"  Δ días fríos = {delta_f:+.2f} µg/m³  ({pct_f:+.1f}%)")


# ─── 11. LIMPIAR Y GUARDAR ───────────────────────────────────────────────────
def clean_and_save(df: pd.DataFrame, save_csv: bool = False) -> pd.DataFrame:
    section("11. Limpieza final y guardado")

    # Columnas raw que NO deben ir como features
    # Incluye los targets brutos, pero NO HDD (que sí es feature)
    raw_cols = TARGETS

    target_cols  = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns
                    if not c.startswith("target_")
                    and c != "date"
                    and c not in raw_cols]

    before = len(df)
    df = df.dropna(subset=target_cols, how="all")
    log(f"  Filas eliminadas (sin targets): {before - len(df):,}")

    lag_cols = [c for c in df.columns if "_lag_" in c]
    before = len(df)
    if lag_cols:
        df = df.dropna(subset=lag_cols[:4], how="any")
    log(f"  Filas eliminadas (sin lags)   : {before - len(df):,}")

    log(f"\n  ✅ Días finales     : {len(df):,}")
    log(f"  Features entrada   : {len(feature_cols)}")
    log(f"  Targets salida     : {len(target_cols)}")
    log(f"  Rango              : {df['date'].min().date()} → {df['date'].max().date()}")

    # Verificar que HDD está en features
    hdd_feats = [c for c in feature_cols if "HDD" in c or "hdd" in c.lower()]
    log(f"  Features HDD/calderas: {hdd_feats}")

    out = PROCESSED_DIR / "features_daily.parquet"
    df.to_parquet(out, index=False)
    log(f"\n  ✅ {out}  ({out.stat().st_size/1024/1024:.1f} MB)")

    if save_csv:
        out_csv = PROCESSED_DIR / "features_daily.csv"
        df.to_csv(out_csv, index=False)
        log(f"  ✅ {out_csv}")

    # Split temporal
    n = len(df)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    log(f"\n── SPLIT TEMPORAL SUGERIDO ──────────────────────────────")
    log(f"  Train (70%): {df['date'].iloc[0].date()} → {df['date'].iloc[t_end].date()}  ({t_end} días)")
    log(f"  Val   (15%): {df['date'].iloc[t_end].date()} → {df['date'].iloc[v_end].date()}  ({v_end-t_end} días)")
    log(f"  Test  (15%): {df['date'].iloc[v_end].date()} → {df['date'].iloc[-1].date()}  ({n-v_end} días)")

    # Stats targets d1
    log(f"\n── ESTADÍSTICAS TARGETS (d1) ─────────────────────────────")
    for t in TARGETS:
        col = f"target_{t}_d1"
        if col in df.columns:
            s = df[col].describe()
            log(f"  {t:<12}: mean={s['mean']:.2f}  std={s['std']:.2f}  "
                f"min={s['min']:.2f}  max={s['max']:.2f}")

    log(f"\n── SIGUIENTE PASO ────────────────────────────────────────")
    log(f"  python train_model.py")
    log(f"  HDD y variables de calderas incluidas — espera ver HDD en Top 10")
    log(f"  Feature Importance para NO2_zbe_d1")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    save_csv      = "--csv"           in sys.argv
    with_forecast = "--with-forecast" in sys.argv
    days_arg      = next((a for a in sys.argv if "--days" in a), None)

    log("=" * 65)
    log("  BUILD FEATURES v6 — Vitoria Air Quality (ZBE + Calderas)")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  Grupo ZBE : {ZBE_STATIONS}")
    log(f"  Grupo OUT : {OUT_STATIONS}")
    log(f"  Targets   : {len(TARGETS)} series × {HORIZON_DAYS} días = {len(TARGETS)*HORIZON_DAYS} targets")
    log(f"  NUEVO v6  : HDD (Heating Degree Days) + variables de calderas")
    log("=" * 65)

    air     = load_air_daily()
    traffic = load_traffic_daily()
    weather = load_weather_daily()

    if days_arg:
        n_days = int(''.join(filter(str.isdigit, days_arg.split("days")[-1])) or
                     sys.argv[sys.argv.index(days_arg)+1])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=n_days)
        air     = air[air["date"]         >= cutoff]
        traffic = traffic[traffic["date"] >= cutoff]
        weather = weather[weather["date"] >= cutoff]
        log(f"\n  ⚡ Modo rápido: últimos {n_days} días")

    df = merge_daily(air, traffic, weather)
    df = add_temporal_features(df)
    df = add_lags_and_rolling(df)
    df = add_targets(df)

    fc = fetch_forecast() if with_forecast else pd.DataFrame()
    df = add_forecast_features(df, fc)

    # Diagnóstico inline antes de guardar
    prueba_del_domingo(df)

    # ── Guardar dataframe completo para predicción y dashboard (ANTES del target-drop) ──────
    # Contiene todas las filas hasta HOY completas, incluso si sus targets de mañana/pasado
    # son NaN (y por tanto serán borradas del set de entrenamiento).
    pred_feature_cols = [c for c in df.columns if not c.startswith("target_")]
    df_latest = df[pred_feature_cols].copy()
    pred_full_path = PROCESSED_DIR / "features_latest.parquet"
    df_latest.to_parquet(pred_full_path, index=False)
    log(f"  → features_latest.parquet: {len(df_latest)} filas (para predecir mañana y backtest)")

    df = clean_and_save(df, save_csv=save_csv)

    log()
    log("=" * 65)
    log("  ✅ DATASET v6 LISTO — siguiente paso: python train_model.py")
    log("  📊 Comprueba Feature Importance de HDD en NO2_zbe_d1")
    log("  🏠 Si HDD entra en Top 5 → hipótesis calderas confirmada")
    log("=" * 65)


if __name__ == "__main__":
    main()