import sys
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import shutil

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"

sys.path.insert(0, str(ROOT_DIR))
# Una única definición de ICA (CAQI europeo) en todo el proyecto.
from src.config import compute_ica_subindex, RISK_THRESHOLDS  # noqa: E402

# --- 1. CONFIGURACIÓN DE ARCHIVOS Y RUTAS ---
csv_file = MODELS_DIR / "counterfactual_gap_v8.csv"

_out_idx = sys.argv.index("--output") if "--output" in sys.argv else None
output_html = sys.argv[_out_idx + 1] if _out_idx is not None else str(ROOT_DIR / "index.html")

# Forzamos ruta local (misma carpeta que el HTML) y copiamos los archivos
img_base_path = ""
plots_source = ROOT_DIR / "reports" / "plots"
plots_dest   = Path(output_html).parent

if plots_source.exists():
    for f in plots_source.glob("*.png"):
        shutil.copy(f, plots_dest / f.name)
    print(f"  OK Copiados {len(list(plots_source.glob('*.png')))} plots a {plots_dest}")

traffic_path = "traffic_map.html"

# ==============================================================================
# 2. DATOS PESTAÑAS 1 Y 2 (Causal v8)
# ==============================================================================
print(f"Leyendo {csv_file}...")
cf_data = {}
df_cv = pd.DataFrame(columns=['date', 'contaminant', 'zone', 'version', 'observed', 'counterfactual', 'gap', 'gap_pct'])
try:
    if csv_file.exists():
        df_cv = pd.read_csv(csv_file)
        # Si ICA no está en df_cv, lo calculamos de forma determinista combinando NO2, PM10 y PM2.5 si están disponibles
        if 'ICA' not in df_cv['contaminant'].unique():
            print("  ICA no encontrado en counterfactual_gap_v8.csv. Generándolo dinámicamente a partir de NO2, PM10 y PM2.5...")
            
            ica_rows = []
            for zone in df_cv['zone'].unique():
                for version in df_cv['version'].unique():
                    # Obtener las series de cada contaminante
                    m_no2 = (df_cv['contaminant'] == 'NO2') & (df_cv['zone'] == zone) & (df_cv['version'] == version)
                    m_pm10 = (df_cv['contaminant'] == 'PM10') & (df_cv['zone'] == zone) & (df_cv['version'] == version)
                    m_pm25 = (df_cv['contaminant'] == 'PM2.5') & (df_cv['zone'] == zone) & (df_cv['version'] == version)
                    
                    df_no2 = df_cv[m_no2].set_index('date')
                    df_pm10 = df_cv[m_pm10].set_index('date')
                    df_pm25 = df_cv[m_pm25].set_index('date')
                    
                    # Unir para alinear fechas
                    df_merged = pd.concat([df_no2[['observed', 'counterfactual']], 
                                           df_pm10[['observed', 'counterfactual']], 
                                           df_pm25[['observed', 'counterfactual']]], 
                                          axis=1, keys=['NO2', 'PM10', 'PM25']).dropna()
                    
                    if not df_merged.empty:
                        # Fix auditoría #8: Calcular ICA con subíndices CAQI europeos normalizados
                        
                        obs_ica = df_merged.apply(lambda r: max(
                            compute_ica_subindex("NO2",   r[('NO2', 'observed')]),
                            compute_ica_subindex("PM10",  r[('PM10', 'observed')]),
                            compute_ica_subindex("PM2.5", r[('PM25', 'observed')])
                        ), axis=1)
                        cf_ica = df_merged.apply(lambda r: max(
                            compute_ica_subindex("NO2",   r[('NO2', 'counterfactual')]),
                            compute_ica_subindex("PM10",  r[('PM10', 'counterfactual')]),
                            compute_ica_subindex("PM2.5", r[('PM25', 'counterfactual')])
                        ), axis=1)
                        
                        df_ica = pd.DataFrame(index=df_merged.index)
                        df_ica['contaminant'] = 'ICA'
                        df_ica['zone'] = zone
                        df_ica['version'] = version
                        df_ica['observed'] = obs_ica
                        df_ica['counterfactual'] = cf_ica
                        df_ica['gap'] = df_ica['observed'] - df_ica['counterfactual']
                        df_ica['gap_pct'] = (df_ica['gap'] / df_ica['counterfactual'] * 100).fillna(0)
                        
                        ica_rows.append(df_ica.reset_index())
            if ica_rows:
                df_cv = pd.concat([df_cv, *ica_rows], ignore_index=True)

        for cont in df_cv['contaminant'].unique():
            for zone in df_cv['zone'].unique():
                key = f"{cont}_{zone}"
                cf_data[key] = {}
                for version in df_cv['version'].unique():
                    mask = (df_cv['contaminant'] == cont) & (df_cv['zone'] == zone) & (df_cv['version'] == version)
                    sub = df_cv[mask].sort_values('date')
                    if not sub.empty:
                        cf_data[key][version] = {
                            "dates": sub['date'].tolist(),
                            "observed": sub['observed'].tolist(),
                            "counterfactual": sub['counterfactual'].tolist(),
                            "gap": sub['gap'].tolist(),
                            "gap_pct": sub['gap_pct'].tolist() 
                        }
        print("  OK Datos causales cargados.")
    else:
        print(f"  WARN El archivo {csv_file} no existe. Se omiten datos causales.")
except Exception as e:
    print(f"  WARN Error procesando {csv_file}: {e}")

cf_json_str = json.dumps(cf_data)

# ==============================================================================
# 3. DATOS PESTAÑAS 2 (Causal v8 Summary & DiD)
# ==============================================================================
summary_stats = {}

def _ratio_gap_pct(frame):
    """Gap como ratio de medias (criterio del informe), no media de ratios."""
    if frame.empty:
        return None
    observed = frame['observed'].mean()
    counterfactual = frame['counterfactual'].mean()
    if counterfactual <= 0:
        return None
    return round((observed - counterfactual) / counterfactual * 100, 1)

try:
    for cont in df_cv['contaminant'].unique():
        for zone in df_cv['zone'].unique():
            key = f"{cont}_{zone}"
            
            # Meteo-Puro
            mask_pure = (df_cv['contaminant'] == cont) & (df_cv['zone'] == zone) & (df_cv['version'] == 'METEO-PURO')
            sub_pure = df_cv[mask_pure]
            
            # Con-Lags
            mask_lags = (df_cv['contaminant'] == cont) & (df_cv['zone'] == zone) & (df_cv['version'] == 'CON-LAGS')
            sub_lags = df_cv[mask_lags]
            
            if not sub_pure.empty:
                summary_stats[key] = {
                    "obs": round(sub_pure['observed'].mean(), 1),
                    "cf_pure": round(sub_pure['counterfactual'].mean(), 1),
                    "pure": _ratio_gap_pct(sub_pure),
                    "pure_daily": round(sub_pure['gap_pct'].mean(), 1),
                    "unit": "µg/m³" if cont != "ICA" else "",
                    "lags": _ratio_gap_pct(sub_lags) if not sub_lags.empty else None,
                    "lags_daily": round(sub_lags['gap_pct'].mean(), 1) if not sub_lags.empty else None,
                }
    print("  OK Summary Stats calculados.")
except Exception as e:
    print(f"  WARN Error calculando summary stats: {e}")

sum_json_str = json.dumps(summary_stats)

try:
    with open(MODELS_DIR / "did_results_v8.json", "r", encoding="utf-8") as f:
        did_data = json.load(f)
    did_json_str = json.dumps(did_data)
except Exception:
    did_json_str = "{}"


def _load_json_file(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


station_qc = _load_json_file(MODELS_DIR / "station_qc.json", {})
causal_validation = _load_json_file(MODELS_DIR / "causal_validation_v8.json", {})
drift_report = _load_json_file(MODELS_DIR / "drift_report.json", {})
pipeline_status_data = _load_json_file(PROCESSED_DIR / "pipeline_status.json", {})

# ==============================================================================
# 4. PREDICCIONES DE MAÑANA
# ==============================================================================
print("Leyendo predicciones de mañana desde predictions_latest.json...")
pred_json_path = PROCESSED_DIR / "predictions_latest.json"
manana_data = {'zbe': {}, 'out': {}}
prediction_date_str = "N/D"

try:
    with open(pred_json_path, encoding="utf-8") as f:
        pred_json = json.load(f)

    pred_date = pred_json.get("prediction_date", "")
    if pred_date:
        prediction_date_str = datetime.strptime(pred_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    targets = pred_json.get("targets", {})
    for cont in ["NO2", "PM10", "PM2.5", "ICA"]:
        for zone in ["zbe", "out"]:
            key = f"{cont}_{zone}_d1"
            # None (no 0) para que la UI muestre "sin datos" en vez de aire limpio.
            val = targets.get(key, {}).get("prediction")
            manana_data[zone][cont] = val

    targets_json_str = json.dumps(targets)

    print(f"  OK Predicciones para {prediction_date_str} cargadas (usando refinamiento v2 si disponible).")

except Exception as e:
    print(f"  WARN No se pudo leer predictions_latest.json: {e}")
    targets_json_str = "{}"
    prediction_date_str = "N/D"
    for z in ['zbe', 'out']:
        for c in ['NO2', 'PM10', 'PM2.5', 'ICA']:
            manana_data[z][c] = None


# ==============================================================================
# 4. BACKTESTING (usando historial de predicciones reales del pipeline)
# ==============================================================================
print("Cargando backtesting desde historial de predicciones...")
perf_data = {'zbe': {}, 'out': {}}

try:
    # Cargar el parquet con datos reales (observaciones)
    df_feat = pd.read_parquet(DATASET_PATH)
    df_feat["date"] = pd.to_datetime(df_feat["date"], utc=True)

    pred_full_path = PROCESSED_DIR / "features_latest.parquet"
    if pred_full_path.exists():
        df_latest = pd.read_parquet(pred_full_path)
        df_latest["date"] = pd.to_datetime(df_latest["date"], utc=True)
        if not df_feat.empty and not df_latest.empty:
            if df_latest["date"].iloc[-1] >= df_feat["date"].iloc[-1]:
                df_feat = df_latest

    # station_daily se publica diariamente y puede ser más reciente que el
    # parquet local. Lo usamos como fuente prioritaria de observaciones reales.
    station_daily_path = PROCESSED_DIR / "station_daily.csv"
    if station_daily_path.exists():
        station_daily = pd.read_csv(station_daily_path)
        station_daily["date"] = pd.to_datetime(station_daily["date"], utc=True)
        observed_daily = pd.DataFrame({"date": station_daily["date"]})
        station_groups = {
            "zbe": ["PAUL", "FUEROS"],
            "out": ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"],
        }
        station_suffixes = {
            "NO2": "NO2", "PM10": "PM10", "PM2.5": "PM25", "ICA": "ICA"
        }
        for zone, stations in station_groups.items():
            for contaminant, suffix in station_suffixes.items():
                columns = [
                    f"{station}_{suffix}"
                    for station in stations
                    if f"{station}_{suffix}" in station_daily.columns
                ]
                if columns:
                    observed_daily[f"{contaminant}_{zone}"] = station_daily[columns].mean(
                        axis=1, skipna=True
                    )

        df_feat = df_feat.set_index("date")
        observed_daily = observed_daily.set_index("date")
        df_feat = df_feat.reindex(df_feat.index.union(observed_daily.index))
        for column in observed_daily.columns:
            df_feat.loc[observed_daily.index, column] = observed_daily[column]
        df_feat = df_feat.reset_index()

    if not df_feat.empty:
        df_feat = df_feat.sort_values("date").reset_index(drop=True)

        # Excluimos hoy y futuros (datos incompletos)
        today_date = datetime.now().date()
        df_past = df_feat[df_feat['date'].dt.date < today_date].copy()

        # Encontrar las últimas 7 fechas con datos reales en cualquier target
        # (antes solo se miraba PM10_out, que podía faltar y desplazar la ventana).
        target_cols_bt = [
            f"{cont}_{zone}"
            for cont in ["NO2", "PM10", "PM2.5", "ICA"]
            for zone in ["zbe", "out"]
            if f"{cont}_{zone}" in df_past.columns
        ]
        if target_cols_bt:
            valid_mask = df_past[target_cols_bt].notna().any(axis=1)
            valid_indices = df_past.index[valid_mask].tolist()
        else:
            valid_indices = []
        if not valid_indices:
            last_rows = df_past.tail(12).copy()
            targets_backtest = last_rows.iloc[-8:-1] if len(last_rows) >= 8 else last_rows
        else:
            last_real_idx = valid_indices[-1]
            last_real_idx_pos = df_past.index.get_loc(last_real_idx)
            targets_backtest = df_past.iloc[max(0, last_real_idx_pos-6) : last_real_idx_pos+1]

        # Fechas para las etiquetas del gráfico
        fechas = targets_backtest['date'].dt.strftime('%d/%m').tolist()

        # --- Cargar historial de predicciones del pipeline ---
        history_path = PROCESSED_DIR / "predictions_history.json"
        pred_history = {}  # Mapa: "YYYY-MM-DD" -> {target: valor}
        if history_path.exists():
            try:
                history_list = json.loads(history_path.read_text(encoding="utf-8"))
                valid_entries = {}
                for entry in history_list:
                    if entry.get("run_type") not in {"production", "backfill_no_leakage"}:
                        continue
                    prediction_date = entry.get("prediction_date")
                    source_date = entry.get("source_date")
                    generated_at = entry.get("generated_at")
                    try:
                        available_at = entry.get("available_at", generated_at)
                        emitted_on_time = pd.Timestamp(available_at) < pd.Timestamp(prediction_date, tz="UTC")
                        expected_source = (
                            pd.Timestamp(prediction_date) - pd.Timedelta(days=1)
                        ).date().isoformat()
                    except Exception:
                        emitted_on_time = False
                        expected_source = None
                    if not emitted_on_time or source_date != expected_source:
                        continue
                    # Si hubo reejecuciones válidas, conservamos la más reciente.
                    previous = valid_entries.get(prediction_date)
                    if previous is None or generated_at > previous.get("generated_at", ""):
                        valid_entries[prediction_date] = entry
                pred_history = {
                    key: entry.get("predictions", {})
                    for key, entry in valid_entries.items()
                }
                print(f"  OK Historial válido cargado: {len(pred_history)} días de predicciones")
            except Exception as e:
                print(f"  WARN Error leyendo historial: {e}")


        for zone in ['zbe', 'out']:
            perf_data[zone]["labels"] = fechas
            for cont in ['NO2', 'PM10', 'PM2.5']:
                target_name = f"{cont}_{zone}_d1"
                contam_col = f"{cont}_{zone}"

                # Obtener valores reales (mediciones) del parquet
                if contam_col in targets_backtest.columns:
                    real_vals = [None if pd.isna(v) else round(v, 1)
                                 for v in targets_backtest[contam_col]]
                else:
                    real_vals = [None] * len(targets_backtest)

                # Obtener predicciones del historial (preferido) o recalcular (fallback)
                pred_vals = []
                for idx_i in range(len(targets_backtest)):
                    target_date = targets_backtest.iloc[idx_i]["date"].date().isoformat()

                    # Opción 1: Buscar en el historial de predicciones
                    if target_date in pred_history and target_name in pred_history[target_date]:
                        pred_val = pred_history[target_date][target_name]
                        pred_vals.append(round(pred_val, 1))
                    else:
                        # No recalcular con el modelo actual: sería información futura.
                        pred_vals.append(None)

                # Asegurar longitud 7
                while len(real_vals) < 7: real_vals.insert(0, None)
                while len(pred_vals) < 7: pred_vals.insert(0, None)

                perf_data[zone][cont] = {
                    "real": real_vals[-7:],
                    "pred": pred_vals[-7:]
                }

            # ICA determinista: misma fórmula CAQI que el resto del dashboard.
            # Si el parquet trae el ICA medido de la zona, ese valor manda para
            # la serie real; el CAQI solo se usa como respaldo.
            measured_ica = [None] * 7
            ica_col = f"ICA_{zone}"
            if ica_col in targets_backtest.columns:
                offset = max(0, len(targets_backtest) - 7)
                for i in range(len(targets_backtest)):
                    arr_idx = i - offset
                    if 0 <= arr_idx < 7:
                        value = targets_backtest.iloc[i].get(ica_col)
                        if pd.notna(value):
                            measured_ica[arr_idx] = round(float(value), 1)

            ica_real = []
            ica_pred = []
            for i in range(7):
                r_vals = [
                    perf_data[zone]['NO2']['real'][i],
                    perf_data[zone]['PM10']['real'][i],
                    perf_data[zone]['PM2.5']['real'][i],
                ]
                if any(v is None for v in r_vals):
                    ica_real.append(None)
                else:
                    ica_real.append(round(max(
                        compute_ica_subindex("NO2", r_vals[0]),
                        compute_ica_subindex("PM10", r_vals[1]),
                        compute_ica_subindex("PM2.5", r_vals[2]),
                    ), 1))

                p_vals = [
                    perf_data[zone]['NO2']['pred'][i],
                    perf_data[zone]['PM10']['pred'][i],
                    perf_data[zone]['PM2.5']['pred'][i],
                ]
                if any(v is None for v in p_vals):
                    ica_pred.append(None)
                else:
                    ica_pred.append(round(max(
                        compute_ica_subindex("NO2", p_vals[0]),
                        compute_ica_subindex("PM10", p_vals[1]),
                        compute_ica_subindex("PM2.5", p_vals[2]),
                    ), 1))

            # Las predicciones históricas de ICA (deterministas) mandan sobre
            # la reconstrucción a partir de los contaminantes.
            for i in range(len(targets_backtest)):
                target_date = targets_backtest.iloc[i]["date"].date().isoformat()
                ica_target = f"ICA_{zone}_d1"
                if target_date in pred_history and ica_target in pred_history[target_date]:
                    arr_idx = i - max(0, len(targets_backtest) - 7)
                    if 0 <= arr_idx < 7:
                        ica_pred[arr_idx] = round(pred_history[target_date][ica_target], 1)

            # Prioridad: ICA medido > reconstrucción CAQI desde contaminantes.
            ica_real = [
                measured if measured is not None else computed
                for measured, computed in zip(measured_ica, ica_real)
            ]

            perf_data[zone]['ICA'] = {"real": ica_real, "pred": ica_pred,
                                      "real_source": "measured_ICA_when_available"}

        n_from_history = sum(1 for d in [t.date().isoformat() for _, t in targets_backtest['date'].items()] if d in pred_history)
        print(f"  OK Backtest listo: {n_from_history}/{len(targets_backtest)} días con predicciones históricas reales.")
    else:
        raise ValueError("El DataFrame de features está vacío.")

except Exception as e:
    import traceback
    print(f"  WARN Error en backtesting: {e}")
    traceback.print_exc()
    for z in ['zbe', 'out']:
        perf_data[z] = {"labels": ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Ayer"]}
        for c in ['NO2', 'PM10', 'PM2.5', 'ICA']:
            perf_data[z][c] = {"real": [None]*7, "pred": [None]*7}

perf_json_str = json.dumps(perf_data)
manana_json_str = json.dumps(manana_data)

try:
    metrics_path = MODELS_DIR / "forecast_v10_metrics.json"
    if not metrics_path.exists():
        metrics_path = MODELS_DIR / "metrics_v8.json"
    metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception:
    metrics_raw = {}
metrics_json_str = json.dumps(metrics_raw)

try:
    with open(MODELS_DIR / "synthetic_control_v9.json", "r", encoding="utf-8") as f:
        sc_data = json.load(f)
        v9_stats = {}
        for cont, stations in sc_data.items():
            cont_key = cont.replace("PM2.5", "PM25")
            for st, st_data in stations.items():
                v9_stats[f"{cont_key}_{st}"] = {
                    "preR2": st_data.get("preR2", "N/A"),
                    "gap": st_data.get("gap", "N/A"),
                    "desc": st_data.get("desc", "Sin datos")
                }
        v9_json_str = json.dumps(v9_stats)
except Exception:
    v9_json_str = "{}"

try:
    with open(MODELS_DIR / "meta_metrics.json", "r", encoding="utf-8") as f:
        meta_metrics = json.load(f)
    meta_json_str = json.dumps(meta_metrics)
except Exception:
    meta_json_str = "{}"

# Pre-tendencia real del event study: un pre-coeficiente significativo implica
# que el supuesto de tendencias paralelas no se sostiene.
try:
    with open(MODELS_DIR / "event_study_v9.json", "r", encoding="utf-8") as f:
        event_study_raw = json.load(f)
    event_study_trend = {}
    for cont, payload in event_study_raw.items():
        coefs = (payload or {}).get("coefficients", {}) or {}
        pre_coefs = {
            int(month): values
            for month, values in coefs.items()
            if int(month) < 0 and int(month) != -1
        }
        n_sig = sum(
            1 for values in pre_coefs.values()
            if values.get("ci_low", 0) > 0 or values.get("ci_high", 0) < 0
        )
        event_study_trend[cont] = {
            "pre_trend_ok": bool(pre_coefs) and n_sig == 0,
            "n_sig": n_sig,
            "n_pre": len(pre_coefs),
        }
    event_study_trend_json_str = json.dumps(event_study_trend)
except Exception as e:
    print(f"  WARN No se pudo evaluar la pre-tendencia del event study: {e}")
    event_study_trend_json_str = "{}"

# ==============================================================================
# 5. DATOS MAPA DE ESTACIONES
# ==============================================================================
STATION_COORDS = {
    "HUETOS":    {"lat": 42.853846, "lon": -2.699907, "zone": "OUT", "label": "Huetos"},
    "LANDAZURI": {"lat": 42.847626, "lon": -2.677065, "zone": "OUT", "label": "Landazuri"},
    "BEATO":     {"lat": 42.849319, "lon": -2.675857, "zone": "OUT", "label": "Beato (Control)"},
    "PAUL":      {"lat": 42.851130, "lon": -2.670824, "zone": "ZBE", "label": "Vicente de Paul (ZBE)"},
    "FUEROS":    {"lat": 42.846270, "lon": -2.669415, "zone": "ZBE", "label": "Fueros (ZBE)"},
    "ZUMABIDE":  {"lat": 42.835437, "lon": -2.673657, "zone": "OUT", "label": "Zumabide"},
}

stations_data = {}

def _get_stn_val(row, stn_code, col_name):
    v = row.get(f"{stn_code}_{col_name}")
    return round(float(v), 1) if v is not None and str(v) != 'nan' else None

try:
    df_stn = pd.read_csv(PROCESSED_DIR / "station_daily.csv", index_col=0, parse_dates=True)
    if not df_stn.empty:
        latest = df_stn.iloc[-1]
        pred_zbe = manana_data.get("zbe", {})
        pred_out = manana_data.get("out", {})

        def _round_or_none(value):
            return round(float(value), 1) if value is not None else None
        
        for stn, meta in STATION_COORDS.items():
            pred_src = pred_zbe if meta["zone"] == "ZBE" else pred_out
            stations_data[stn] = {
                **meta,
                "NO2":  _get_stn_val(latest, stn, "NO2"),
                "PM10": _get_stn_val(latest, stn, "PM10"),
                "PM25": _get_stn_val(latest, stn, "PM25"),
                "ICA":  _get_stn_val(latest, stn, "ICA"),
                "pred_NO2":  _round_or_none(pred_src.get("NO2")),
                "pred_PM10": _round_or_none(pred_src.get("PM10")),
                "pred_PM25": _round_or_none(pred_src.get("PM2.5")),
            }
    else:
        raise ValueError("station_daily.csv está vacío.")
        
except Exception as e:
    print(f"  WARN Error cargando station_daily para el mapa: {e}")
    for stn, meta in STATION_COORDS.items():
        stations_data[stn] = {**meta, "NO2": None, "PM10": None, "PM25": None, "ICA": None,
                              "pred_NO2": None, "pred_PM10": None, "pred_PM25": None}

stations_json_str = json.dumps(stations_data)

# ==============================================================================
# 5b. DATOS TRÁFICO YOY
# ==============================================================================
try:
    with open(PROCESSED_DIR / "peripheral_traffic_yoy.json", "r", encoding="utf-8") as f:
        traffic_yoy_data = json.load(f)
    traffic_yoy_json_str = json.dumps(traffic_yoy_data)
    print("  OK Datos de tráfico YoY cargados.")
except Exception as e:
    print(f"  WARN No se pudo cargar peripheral_traffic_yoy.json: {e}")
    traffic_yoy_json_str = "[]"

# ==============================================================================
# 6. PLANTILLA HTML
# ==============================================================================
html_template = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZBE Vitoria-Gasteiz — Dashboard Integral</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #f4f6f8;
    --surface:   #ffffff;
    --surface2:  #f0f2f5;
    --border:    #dcdfe4;
    --text:      #111827;
    --muted:     #6b7280;
    --observed:  #0288d1;
    --cf-pure:   #d84315;
    --cf-lags:   #ef6c00;
    --accent:    #4f46e5;
    --green:     #059669;
    --turquoise: #0d9488;
    --red:       #dc2626;
    --yellow:    #d97706;
  }

  [data-theme="dark"] {
    --bg:        #0d0f14;
    --surface:   #151820;
    --surface2:  #1c2030;
    --border:    #2a2f3f;
    --text:      #e2e6f0;
    --muted:     #6b7494;
    --observed:  #4fc3f7;
    --cf-pure:   #ff7043;
    --cf-lags:   #ffb74d;
    --accent:    #7c6af7;
    --green:     #4caf82;
    --turquoise: #2dd4bf;
    --red:       #ef5350;
    --yellow:    #ffd54f;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body { background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; min-height: 100vh; overflow-x: hidden; }

  /* Contenedor fijo para header y pestañas */
  .header-wrapper { position: sticky; top: 0; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
  
  /* Barra superior (Azul) + Bandas (Blanca y Roja francesa) */
  .main-header { position: relative; display: flex; justify-content: space-between; align-items: center; padding: 16px 48px; background: #0055A4; color: #fff; border: none; }
  .main-header::after { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 6px; background: linear-gradient(to bottom, #ffffff 50%, #EF4135 50%); }

  .main-header-title { font-size: 18px; font-weight: 700; }  .header-right { display: flex; align-items: center; gap: 24px; }
  .main-header-author { font-size: 13px; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9; }
  .pipeline-banner { display: none; padding: 9px 48px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; border-bottom: 1px solid var(--border); background: rgba(217, 119, 6, 0.12); color: #8a4b00; }
  .pipeline-banner.warn { display: block; }
  .pipeline-banner.alert { display: block; background: rgba(220, 38, 38, 0.12); color: #8b1a1a; }
  [data-theme="dark"] .pipeline-banner { color: #f5c26b; }
  [data-theme="dark"] .pipeline-banner.alert { color: #ff9b9b; }
  
  /* Botón del selector de tema / idioma */
  .theme-toggle { 
    background: rgba(255,255,255,0.15); 
    border: 1px solid rgba(255,255,255,0.3); 
    color: #fff; 
    padding: 6px 14px; 
    border-radius: 4px; 
    cursor: pointer; 
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 12px; 
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 4px;
    height: 32px;
  }
  .theme-toggle:hover { background: rgba(255,255,255,0.25); }

  .header { padding: 40px 48px 28px; border-bottom: 1px solid var(--border); display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
  .header-left { max-width: 620px; }
  .label-tag { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); border: 1px solid var(--accent); padding: 3px 8px; border-radius: 2px; margin-bottom: 12px; }
  h1 { font-size: 26px; font-weight: 700; line-height: 1.25; letter-spacing: -0.02em; color: var(--text); }
  h1 span { color: var(--accent); }
  .subtitle { margin-top: 8px; color: var(--muted); font-size: 13px; line-height: 1.6; }
  
  .legend-global { display: flex; flex-direction: column; gap: 8px; min-width: 200px; background: var(--surface); border: 1px solid var(--border); padding: 16px 18px; border-radius: 6px; }
  .legend-item { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); }
  .legend-line { width: 28px; height: 2px; border-radius: 1px; flex-shrink: 0; }
  .legend-line.dashed { background: repeating-linear-gradient(90deg, currentColor 0, currentColor 5px, transparent 5px, transparent 9px); }

  .controls { padding: 20px 48px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .controls-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-right: 4px; }
  .tab-group { display: flex; gap: 4px; }
  .tab { padding: 6px 14px; border-radius: 3px; border: 1px solid var(--border); background: transparent; color: var(--muted); font-size: 12px; font-family: 'IBM Plex Mono', monospace; cursor: pointer; transition: all 0.15s; }
  .tab:hover { border-color: var(--accent); color: var(--text); }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .sep { width: 1px; height: 24px; background: var(--border); margin: 0 8px; }

  .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
  .summary-card { background: var(--surface); padding: 20px 24px; display: flex; flex-direction: column; gap: 6px; }
  .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
  .card-value { font-size: 24px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }
  .card-value.neg { color: var(--green); } .card-value.pos { color: var(--red); } .card-value.neutral { color: var(--yellow); }
  .card-sub { font-size: 11px; color: var(--muted); }
  .card-range { font-size: 11px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }

  .charts-section { padding: 32px 48px; display: flex; flex-direction: column; gap: 40px; }
  .fig-block { display: flex; flex-direction: column; gap: 16px; }
  .fig-header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
  .fig-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
  .fig-title strong { color: var(--text); font-size: 13px; }
  .fig-note { font-size: 11px; color: var(--muted); font-style: italic; }

  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; position: relative; height: 320px; }
  canvas { display: block; }
  
  .img-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; display: flex; justify-content: center; align-items: center; }
  .img-wrap img { max-width: 100%; height: auto; border-radius: 4px; }
  .error-msg { color: var(--red); font-family: 'IBM Plex Mono', monospace; font-size: 12px; display: none; text-align: center; background: rgba(239, 83, 80, 0.1); padding: 20px; border: 1px dashed var(--red); border-radius: 6px; }

  .did-section { padding: 0 48px 40px; }
  .did-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 16px; }
  .did-title strong { color: var(--text); font-size: 13px; }
  .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 480px; }
  th { text-align: left; padding: 10px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); border-bottom: 1px solid var(--border); background: var(--surface2); }
  td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }
  .sig-yes { color: var(--green); font-weight: 600; }
  .sig-no  { color: var(--muted); }
  .beta-neg { color: var(--green); } .beta-pos { color: var(--red); } .beta-neu { color: var(--yellow); }

  .footer { padding: 20px 48px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; color: var(--muted); font-size: 11px; font-family: 'IBM Plex Mono', monospace; }

  .top-nav { display: flex; background: var(--surface2); border-bottom: 1px solid var(--border); padding: 0 48px; }
  .top-nav-btn { padding: 16px 24px; background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; cursor: pointer; transition: all 0.2s; }
  .top-nav-btn:hover { color: var(--text); }
  .top-nav-btn.active { border-bottom-color: var(--accent); color: var(--text); }
  .view-container { display: none; }
  .view-container.active { display: block; }
  
  .v10-risk-container { text-align: center; padding: 50px 40px; background: var(--surface2); border-bottom: 1px solid var(--border); }
  .v10-planning-container { text-align: center; padding: 28px 40px; background: var(--surface); border-bottom: 1px solid var(--border); }
  .v10-planning-note { max-width: 760px; margin: 0 auto 18px; color: var(--muted); font-size: 12px; }
  .v10-planning-container .v10-risk-grid { gap: 48px; }
  .v10-planning-container .v10-risk-item .val { font-size: 22px; }
  .v10-interval { font-size: 10px; color: var(--muted); margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
  .v10-risk-badge { display: inline-block; font-size: 32px; font-weight: 800; padding: 12px 30px; border-radius: 8px; margin-bottom: 30px; font-family: 'IBM Plex Mono', monospace; border: 2px solid; }
  .v10-risk-grid { display: flex; justify-content: center; gap: 70px; }
  .v10-risk-item .val { font-size: 28px; font-weight: 700; color: var(--text); font-family: 'IBM Plex Mono', monospace;}
  .v10-risk-item .lab { font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 8px; font-family: 'IBM Plex Mono', monospace;}
  
  .perf-chart-wrap, .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; height: 350px; position: relative; }
  .perf-chart-wrap canvas, .chart-wrap canvas { display: block; width: 100% !important; height: 100% !important; }

  /* ── Foresight grid (desktop) ── */
  .foresight-grid { display: grid; grid-template-columns: minmax(280px, 1.2fr) minmax(280px, 1.8fr); gap: 40px; }

  /* ── RESPONSIVE: tablet (≤900px) ── */
  @media (max-width: 900px) {
    .main-header { padding: 12px 16px; flex-direction: column; align-items: flex-start; gap: 10px; }
    .main-header-title { font-size: 15px; }
    .header-right { width: 100%; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
    .main-header-author { display: none; }
    .top-nav { padding: 0 8px; overflow-x: auto; -webkit-overflow-scrolling: touch; flex-shrink: 0; white-space: nowrap; }
    .top-nav-btn { padding: 12px 14px; font-size: 11px; letter-spacing: 0; white-space: nowrap; }
    .header { padding: 24px 16px 20px; flex-direction: column; gap: 16px; }
    .header-left { max-width: 100%; }
    .legend-global { min-width: 0; width: 100%; flex-direction: row; flex-wrap: wrap; gap: 12px; }
    .controls { padding: 14px 16px; gap: 8px; }
    .controls-label { font-size: 10px; }
    .tab { padding: 5px 10px; font-size: 11px; }
    .sep { display: none; }
    .summary-grid { grid-template-columns: 1fr 1fr; }
    .card-value { font-size: 20px; }
    h1 { font-size: 20px; }
    .v10-risk-container { padding: 30px 16px; }
    .v10-risk-grid { flex-wrap: wrap; gap: 24px 40px; justify-content: center; }
    .v10-risk-item .val { font-size: 22px; }
    .v10-risk-badge { font-size: 26px; padding: 10px 22px; }
    .charts-section { padding: 20px 16px; gap: 28px; }
    .chart-wrap, .perf-chart-wrap { height: 280px; padding: 16px; }
    .did-section { padding: 0 16px 28px; }
    .did-section > div[style*="padding: 0 48px"] { padding: 0 !important; }
    .map-header { padding: 16px 16px 12px; flex-direction: column; gap: 12px; }
    .map-legend { min-width: 0; width: 100%; flex-direction: row; flex-wrap: wrap; gap: 10px; }
    .map-wrap { padding: 16px 16px 24px; }
    #stationMap { height: 380px; }
    .foresight-grid { grid-template-columns: 1fr; }
    .footer { padding: 16px; flex-direction: column; gap: 6px; text-align: center; }
  }

  /* ── RESPONSIVE: móvil (≤600px) ── */
  @media (max-width: 600px) {
    .main-header-title { font-size: 13px; }
    .theme-toggle { font-size: 10px; padding: 5px 10px; height: 28px; }
    .top-nav-btn { padding: 10px 12px; font-size: 10px; }
    .summary-grid { grid-template-columns: 1fr; }
    .card-value { font-size: 22px; }
    h1 { font-size: 18px; }
    .v10-risk-container { padding: 20px 12px; }
    .v10-risk-grid { gap: 16px 24px; }
    .v10-risk-item .val { font-size: 20px; }
    .v10-risk-badge { font-size: 22px; padding: 8px 18px; margin-bottom: 20px; }
    .controls { padding: 10px 12px; }
    .tab-group { flex-wrap: wrap; }
    .charts-section { padding: 14px 12px; gap: 20px; }
    .chart-wrap, .perf-chart-wrap { height: 240px; padding: 12px; }
    .img-wrap { padding: 12px; }
    .did-section { padding: 0 12px 20px; }
    .map-wrap { padding: 12px 12px 20px; }
    #stationMap { height: 320px; }
    .footer { padding: 12px; font-size: 10px; }
  }

  #stationMap { height: 540px; border-radius: 4px; border: 1px solid var(--border); }
  .map-header { padding: 32px 48px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
  .map-legend { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; min-width: 170px; }
  .map-legend-item { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); }
  .map-legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .map-wrap { padding: 24px 48px 40px; }
  
  /* Filtro modo oscuro para tiles de OpenStreetMap libres sin API key */
  [data-theme="dark"] .leaflet-tile-pane {
    filter: brightness(0.6) invert(1) contrast(3) hue-rotate(200deg) saturate(0.3) brightness(0.75);
  }
  
  .leaflet-popup-content-wrapper { background: var(--surface2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important; }
  .leaflet-popup-tip { background: var(--surface2) !important; }
  .leaflet-popup-content { margin: 14px 18px !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 13px !important; line-height: 1.6 !important; }
  .popup-name { font-weight: 700; font-size: 14px; margin-bottom: 8px; color: var(--text); }
  .popup-row { display: flex; justify-content: space-between; gap: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 2px 0; }
  .popup-label { color: var(--muted); }
  .popup-val { font-weight: 600; }
  .popup-section { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
  .popup-section-title { color: var(--accent); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; }
  
  .leaflet-control-attribution { background: var(--surface) !important; color: var(--muted) !important; }
  .leaflet-control-attribution a { color: var(--accent) !important; }
  .leaflet-control-zoom a { background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; }
  
  /* Estilos para el iframe de tráfico */
  .traffic-iframe-container { width: 100%; height: calc(100vh - 120px); border: none; }
  #trafficIframe { width: 100%; height: 100%; border: none; }
</style>
</head>
<body>

<div class="header-wrapper">
  <div class="main-header">
    <div class="main-header-title" data-i18n="mainTitle">Vitoria-Gasteiz — Análisis Calidad del Aire y ZBE</div>
    <div class="header-right">
      <div class="main-header-author">Eduardo Ortas Armentia</div>
      <div style="display: flex; gap: 8px;">
        <button class="theme-toggle" onclick="toggleLang()" id="langBtn">EU</button>
        <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">Modo Oscuro</button>
        <a href="https://www.linkedin.com/in/eortas/" target="_blank" class="theme-toggle" id="contactBtn" data-i18n="contactBtn" style="text-decoration:none; display: flex; align-items: center; gap: 6px;">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
            <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
          </svg>
          Contactar con el creador
        </a>
      </div>
    </div>
  </div>

  <div class="top-nav">
    <button class="top-nav-btn active" onclick="switchMainView('v10', this)" data-i18n="nav1">1. Predicción Operativa</button>
    <button class="top-nav-btn" onclick="switchMainView('v8', this)" data-i18n="nav2">2. Monitor Interactivo Diario</button>
    <button class="top-nav-btn" onclick="switchMainView('v9', this)" data-i18n="nav3">3. Validación Causal</button>
    <button class="top-nav-btn" onclick="switchMainView('map', this)" data-i18n="nav4">4. Mapa de Estaciones</button>
    <button class="top-nav-btn" onclick="switchMainView('traffic', this)" data-i18n="nav5">5. Mapa de Tráfico</button>
    <button class="top-nav-btn" onclick="switchMainView('foresight', this)" data-i18n="nav6">6. Foresight AI</button>
  </div>
</div>

<div id="pipelineBanner" class="pipeline-banner"></div>

<div id="view-v8" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag" data-i18n="v8Tag">Análisis Causal — ZBE Vitoria-Gasteiz</div>
      <h1 data-i18n="v8Title">Counterfactual Meteorológico<br><span>ZBE Sep·2025 → Actual</span></h1>
      <p class="subtitle" data-i18n="v8Subtitle">Comparación entre contaminación observada y el escenario si no se hubiera implementado la ZBE.</p>
    </div>
    <div class="legend-global">
      <div class="card-label" style="margin-bottom:4px" data-i18n="legendTitle">Leyenda</div>
      <div class="legend-item"><div class="legend-line" style="background:var(--observed);height:2px"></div><span data-i18n="legendObs">Observado (real)</span></div>
      <div class="legend-item"><div class="legend-line dashed" style="color:var(--cf-pure);height:0;border-top:2px dashed var(--cf-pure)"></div><span data-i18n="legendCFPure">CF Meteo-Puro</span></div>
      <div class="legend-item"><div class="legend-line dashed" style="color:var(--cf-lags);height:0;border-top:2px dashed var(--cf-lags)"></div><span data-i18n="legendCFLags">CF Con-Lags</span></div>
      <div class="legend-item"><div style="width:28px;height:12px;background:rgba(79,195,247,0.2);border-radius:2px;flex-shrink:0"></div><span data-i18n="legendBand">Banda de efecto</span></div>
    </div>
  </div>

  <div class="controls">
    <span class="controls-label" data-i18n="contaminant">Contaminante</span>
    <div class="tab-group" id="contTabs">
      <button class="tab active" onclick="selectCont('NO2',this)">NO₂</button>
      <button class="tab" onclick="selectCont('PM10',this)">PM10</button>
      <button class="tab" onclick="selectCont('PM2.5',this)">PM2.5</button>
      <button class="tab" onclick="selectCont('ICA',this)">ICA</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label" data-i18n="zone">Zona</span>
    <div class="tab-group" id="zoneTabs">
      <button class="tab active" onclick="selectZone('zbe',this)" data-i18n="zoneIn">ZBE (dentro)</button>
      <button class="tab" onclick="selectZone('out',this)" data-i18n="zoneOut">FUERA DE ZBE (control)</button>
    </div>
  </div>

  <div class="summary-grid" id="summaryCards"></div>

  <div class="charts-section">
    <div class="chart-container">
      <div class="card-label" style="margin-bottom:12px" data-i18n="v8Fig1">Figura 1 — Observado vs Counterfactual</div>
      <div class="chart-wrap"><canvas id="fig1"></canvas></div>
    </div>

    <div class="chart-container">
      <div class="card-label" style="margin-bottom:12px" data-i18n="v8Fig2">Figura 2 — Banda de incertidumbre del efecto ZBE</div>
      <div class="chart-wrap"><canvas id="fig2"></canvas></div>
    </div>

    <div class="chart-container">
      <div class="card-label" style="margin-bottom:12px" data-i18n="v8Fig3">Figura 3 — Gap medio por contaminante y zona</div>
      <div class="chart-wrap"><canvas id="fig3"></canvas></div>
    </div>
  </div>

  <div class="did-section">
    <div class="did-title" data-i18n="didTitle"><strong>Tabla 1</strong> — Difference-in-Differences v8</div>
    <div class="table-scroll"><table><thead><tr><th data-i18n="didCont">Contaminante</th><th>β₃</th><th data-i18n="didPVal">p-valor</th><th>IC 95%</th><th data-i18n="didRel">Efecto relativo</th><th>R²</th><th>n obs</th><th data-i18n="didSig">Significativo</th></tr></thead><tbody id="didTable"></tbody></table></div>
  </div>
</div>

<div id="view-v9" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag" data-i18n="v9Tag">Métodos Econométricos — Evaluación de Impacto</div>
      <h1 data-i18n="v9Title">Evaluación Causal de la ZBE<br><span>Event Study & Synthetic Control</span></h1>
    </div>
  </div>
  <div class="controls">
    <span class="controls-label" data-i18n="contaminant">Contaminante</span>
    <div class="tab-group" id="contTabsV9">
      <button class="tab active" onclick="selectContV9('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContV9('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContV9('PM2.5', this)">PM2.5</button>
      <button class="tab" onclick="selectContV9('ICA', this)">ICA</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label" data-i18n="v9Station">Estación IntraZBE</span>
    <div class="tab-group" id="stationTabsV9">
      <button class="tab active" onclick="selectStationV9('PAUL', this)">PAUL</button>
      <button class="tab" onclick="selectStationV9('FUEROS', this)">FUEROS</button>
    </div>
  </div>
  <div class="summary-grid" id="summaryCardsV9"></div>
  
  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title" data-i18n="v9Fig1"><strong>Figura 1</strong> — Control Sintético (Serie Suavizada)</div></div>
      <div class="img-wrap">
        <img id="img-sc" src="" alt="Gráfico Control Sintético" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div id="err-sc" class="error-msg" data-i18n="imgNotFound">Imagen no encontrada. Verifica el nombre del archivo.</div>
      </div>
    </div>
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title" data-i18n="v9Fig2"><strong>Figura 2</strong> — Event Study DiD</div></div>
      <div class="img-wrap">
        <img id="img-es" src="" alt="Gráfico Event Study" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div id="err-es" class="error-msg" data-i18n="imgNotFound">Imagen no encontrada. Verifica el nombre del archivo.</div>
      </div>
    </div>
  </div>
</div>

<div id="view-v10" class="view-container active">
  <div class="v10-risk-container">
    <div class="label-tag" data-i18n="v10Tag">EEA Standards — Air Quality Risk</div>
    <h2 style="margin-bottom: 10px; color: var(--muted); font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.1em;"><span data-i18n="v10Title">Calidad del aire prevista para</span> <span style="color:var(--text);font-weight:700;" id="riskDateSpan"></span></h2>
    <div id="riskBadge" class="v10-risk-badge" data-i18n="v10Calculating">CALCULANDO...</div>
    <div class="v10-risk-grid">
      <div class="v10-risk-item"><div class="val" id="val-no2">--</div><div class="lab">NO₂ µg/m³</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm25">--</div><div class="lab">PM2.5 µg/m³</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm10">--</div><div class="lab">PM10 µg/m³</div></div>
      <div class="v10-risk-item"><div class="val" id="val-ica">--</div><div class="lab">ICA</div></div>
    </div>
  </div>

  <div class="v10-planning-container" id="d2PlanningBlock">
    <div class="label-tag" data-i18n="v10PlanningTag">Planificación D+2 · menor confianza</div>
    <h2 style="margin-bottom: 8px; color: var(--muted); font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.1em;">
      <span data-i18n="v10PlanningTitle">Perspectiva de calidad del aire para</span>
      <span style="color:var(--text);font-weight:700;" id="riskDateD2Span"></span>
    </h2>
    <p class="v10-planning-note" data-i18n="v10PlanningNote">Horizonte orientado a planificación. Los intervalos son más amplios que en D+1.</p>
    <div id="riskBadgeD2" class="v10-risk-badge" data-i18n="v10Calculating">CALCULANDO...</div>
    <div class="v10-risk-grid">
      <div class="v10-risk-item"><div class="val" id="val-no2-d2">--</div><div class="lab">NO₂ µg/m³</div><div class="v10-interval" id="range-no2-d2">IC90 --</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm25-d2">--</div><div class="lab">PM2.5 µg/m³</div><div class="v10-interval" id="range-pm25-d2">IC90 --</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm10-d2">--</div><div class="lab">PM10 µg/m³</div><div class="v10-interval" id="range-pm10-d2">IC90 --</div></div>
      <div class="v10-risk-item"><div class="val" id="val-ica-d2">--</div><div class="lab">ICA</div><div class="v10-interval" id="range-ica-d2">IC90 --</div></div>
    </div>
  </div>

  <div class="controls">
    <span class="controls-label" data-i18n="contaminant">Contaminante</span>
    <div class="tab-group" id="contTabsV10">
      <button class="tab active" onclick="selectContV10('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContV10('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContV10('PM2.5', this)">PM2.5</button>
      <button class="tab" onclick="selectContV10('ICA', this)">ICA</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label" data-i18n="zone">ZONA </span>
    <div class="tab-group" id="zoneTabsV10">
      <button class="tab active" onclick="selectZoneV10('zbe', this)" data-i18n="zoneIn">ZBE</button>
      <button class="tab" onclick="selectZoneV10('out', this)" data-i18n="zoneOut">FUERA DE ZBE</button>
    </div>
  </div>

  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header">
        <div class="fig-title" data-i18n="auditTitle"><strong>Evolución y previsión:</strong> histórico validado + D+1 + D+2</div>
      </div>
      <div class="perf-chart-wrap">
        <canvas id="perfChart"></canvas>
      </div>
    </div>
    
    <div class="did-section" style="padding: 0;">
      <div class="did-title" data-i18n="backtestTitle"><strong>Último día validado</strong> — control de precisión</div>
      <div class="table-scroll"><table>
        <thead><tr><th data-i18n="colParam">Parámetro</th><th data-i18n="colPred">Predicción emitida</th><th data-i18n="colReal">Medición real</th><th data-i18n="colStatus">Estado</th></tr></thead>
        <tbody id="backtestTable"></tbody>
      </table></div>
    </div>
  </div>

  <div class="did-section" style="padding: 0 0 40px;">
      <div class="did-title" style="padding: 0 0 16px;" data-i18n="perfTitle"><strong>Rendimiento D+1 · forecast v10</strong> — validación temporal</div>
    <div class="table-scroll">
      <table id="metricsTable">
        <thead><tr><th data-i18n="colContZone">Contaminante + Zona</th><th data-i18n="colRMSE">RMSE (µg/m³)</th><th data-i18n="colMAE">MAE (µg/m³)</th><th data-i18n="colR2">R²</th><th data-i18n="colMAPE">MAPE %</th><th data-i18n="colNFeatures">N. Features</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
    <p style="padding: 12px 0 0; font-size:11px; color:var(--muted); font-family:'IBM Plex Mono',monospace;">
      <span data-i18n="mapeNote">MAPE calculado solo sobre filas con valor observado > 5 µg/m³.</span>&nbsp;&nbsp;<span data-i18n="mapeThreshold">Criterio operativo: mejora frente a persistencia y cobertura suficiente del intervalo.</span>
    </p>
  </div>

  <div class="did-section" style="padding: 0 0 40px;">
    <div class="did-title" style="padding: 0 0 16px;" data-i18n="perfTitleD2"><strong>Rendimiento D+2 · forecast v10</strong> — validación temporal</div>
    <div class="table-scroll">
      <table id="metricsTableD2">
        <thead><tr><th data-i18n="colContZone">Contaminante + Zona</th><th data-i18n="colRMSE">RMSE (µg/m³)</th><th data-i18n="colMAE">MAE (µg/m³)</th><th data-i18n="colR2">R²</th><th data-i18n="colMAPE">MAPE %</th><th data-i18n="colNFeatures">N. Features</th></tr></thead>
        <tbody id="metricsBodyD2"></tbody>
      </table>
    </div>
  </div>
</div>

<div id="view-map" class="view-container">
  <div class="map-header">
    <div class="header-left">
      <div class="label-tag" data-i18n="mapTag">Vitoria-Gasteiz — Red Sensores Municipales</div>
      <h1 data-i18n="mapTitle">Mapa de <span>Estaciones</span></h1>
      <p class="subtitle" data-i18n="mapSubtitle">Medias parciales por estación y predicción vigente para la fecha indicada. Haz click sobre un punto para ver el detalle.</p>
    </div>
    <div class="map-legend">
      <div class="card-label" style="margin-bottom:4px" data-i18n="mapLegendTitle">Semáforo ICA</div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--green)"></div><span data-i18n="icaGood">Buena (≤25)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--yellow)"></div><span data-i18n="icaMod">Moderada (25-50)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:#ff7043"></div><span data-i18n="icaBad">Mala (50-75)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--red)"></div><span data-i18n="icaVeryBad">Muy mala (>75)</span></div>
    </div>
  </div>
  <div class="map-wrap">
    <div id="stationMap"></div>
  </div>
</div>

<div id="view-traffic" class="view-container">
  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header">
        <div class="fig-title" data-i18n="trafficYoYTitle"><strong>Estudio de Impacto:</strong> Evolución del Tráfico Perimetral (YoY)</div>
      </div>
      <div class="subtitle" style="margin-bottom: 20px;" data-i18n="trafficYoYSubtitle">
        Comparativa entre el tráfico actual y el mismo periodo del año anterior (alineado por día de la semana).
      </div>
      <div class="chart-wrap" style="height: 400px;">
        <canvas id="trafficYoYChart"></canvas>
      </div>
    </div>
  </div>
  <div class="traffic-iframe-container" style="height: 600px; border-top: 1px solid var(--border); margin-top: 40px;">
    <iframe id="trafficIframe" src="traffic_map.html" title="Mapa de Tráfico"></iframe>
  </div>
</div>

<div id="view-foresight" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag" data-i18n="fsTag">Explicación de la predicción</div>
      <h1 data-i18n="fsTitle">Foresight <span> con IA</span></h1>
      <p class="subtitle" data-i18n="fsSubtitle">Factores meteorológicos y de contexto asociados a la predicción vigente del modelo.</p>
    </div>
  </div>
  
  <div class="controls">
    <span class="controls-label" data-i18n="contaminant">Contaminante</span>
    <div class="tab-group" id="contTabsForesight">
      <button class="tab active" onclick="selectContFs('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContFs('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContFs('PM2.5', this)">PM2.5</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label" data-i18n="zone">ZONA</span>
    <div class="tab-group" id="zoneTabsForesight">
      <button class="tab active" onclick="selectZoneFs('zbe', this)" data-i18n="zoneIn">ZBE</button>
      <button class="tab" onclick="selectZoneFs('out', this)" data-i18n="zoneOut">FUERA DE ZBE</button>
    </div>
  </div>

  <div class="charts-section foresight-grid">
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 30px;">
      <h3 style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--accent); letter-spacing:1px; margin-bottom:16px; text-transform:uppercase;" data-i18n="fsNarrativeTitle">Explicación de la predicción</h3>
      <div id="fs-narrative" style="font-size:15px; line-height:1.7; color:var(--text); white-space: pre-line;">Cargando narrativa...</div>
    </div>
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 30px; display: flex; flex-direction: column;">
      <h3 style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); letter-spacing:1px; margin-bottom:12px; text-transform:uppercase;" data-i18n="fsShapTitle">Top Factores de Impacto (SHAP)</h3>
      <div style="flex: 1; min-height: 300px; position: relative;">
        <canvas id="fsChart"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
// ===========================================================================
// INYECCIÓN DE DATOS Y VARIABLES GLOBALES
// ===========================================================================
let cfData = __CF_DATA_PLACEHOLDER__;
let perfStats = __PERF_DATA_PLACEHOLDER__;
let manana = __MANANA_DATA_PLACEHOLDER__;
let metricsData = __METRICS_DATA_PLACEHOLDER__;
let stationsData = __STATIONS_DATA_PLACEHOLDER__;
let targetsData = __TARGETS_DATA_PLACEHOLDER__;
const v9Stats = __V9_DATA_PLACEHOLDER__;
const metaData = __META_DATA_PLACEHOLDER__;
const trafficYoyData = __TRAFFIC_YOY_DATA_PLACEHOLDER__;

let currentTheme = 'light';
let mapInstance = null;
let mapTileLayer = null;
let _mapInitialized = false;

let currentLang = localStorage.getItem('vitoria_lang') || 'eu';
if (currentLang !== 'es' && currentLang !== 'eu') currentLang = 'eu';
const predDate = "__PRED_DATE_PLACEHOLDER__";

const getCssVar = (name) => {
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (val) return val;
  const fallbacks = {
    '--bg':        '#f4f6f8',
    '--surface':   '#ffffff',
    '--surface2':  '#f0f2f5',
    '--border':    '#dcdfe4',
    '--text':      '#111827',
    '--muted':     '#6b7280',
    '--observed':  '#0288d1',
    '--cf-pure':   '#d84315',
    '--cf-lags':   '#ef6c00',
    '--accent':    '#4f46e5',
    '--green':     '#059669',
    '--turquoise': '#0d9488',
    '--red':       '#dc2626',
    '--yellow':    '#d97706'
  };
  if (document.documentElement.getAttribute('data-theme') === 'dark') {
    const darkFallbacks = {
      '--bg':        '#0d0f14',
      '--surface':   '#151820',
      '--surface2':  '#1c2030',
      '--border':    '#2a2f3f',
      '--text':      '#e2e6f0',
      '--muted':     '#6b7494',
      '--observed':  '#4fc3f7',
      '--cf-pure':   '#ff7043',
      '--cf-lags':   '#ffb74d',
      '--accent':    '#7c6af7',
      '--green':     '#4caf82',
      '--turquoise': '#2dd4bf',
      '--red':       '#ef5350',
      '--yellow':    '#ffd54f'
    };
    return darkFallbacks[name] || fallbacks[name] || '';
  }
  return fallbacks[name] || '';
};

const SUMMARY_STATS = __SUMMARY_DATA_PLACEHOLDER__;
const DID_RESULTS = __DID_DATA_PLACEHOLDER__;
const EVENT_STUDY_TREND = __EVENT_STUDY_TREND_PLACEHOLDER__;
const RISK_THRESHOLDS = __RISK_THRESHOLDS_PLACEHOLDER__;
const STATION_QC = __STATION_QC_PLACEHOLDER__;
const CAUSAL_VALIDATION = __CAUSAL_VALIDATION_PLACEHOLDER__;
const DRIFT_REPORT = __DRIFT_REPORT_PLACEHOLDER__;
const PIPELINE_STATUS = __PIPELINE_STATUS_DATA_PLACEHOLDER__;

function movingAvg(arr, windowSize) {
  // Ventana centrada exacta también para tamaños pares (14 → 6 antes, 7 después).
  const halfLow = Math.floor((windowSize - 1) / 2);
  const halfHigh = Math.floor(windowSize / 2);
  return arr.map((_, i, a) => {
    let start = Math.max(0, i - halfLow);
    let end = Math.min(a.length, i + halfHigh + 1);
    let sub = a.slice(start, end).filter(v => v !== null && !isNaN(v));
    return sub.length ? sub.reduce((s, v) => s + v, 0) / sub.length : null;
  });
}

function fmt(v) { return (v === null || v === undefined) ? 'N/D' : v.toFixed(1); }
function num(v, digits) {
  return (v === null || v === undefined || Number.isNaN(Number(v))) ? 'N/D' : Number(v).toFixed(digits);
}

const translations = {
  es: {
    mainTitle: "Vitoria-Gasteiz — Análisis Calidad del Aire y ZBE",
    nav1: "1. Predicción Operativa",
    nav2: "2. Monitor Interactivo Diario",
    nav3: "3. Validación Causal",
    nav4: "4. Mapa de Estaciones",
    nav5: "5. Mapa de Tráfico",
    nav6: "6. Foresight AI",
    themeDark: "Modo Oscuro",
    themeLight: "Modo Claro",
    v8Tag: "Análisis Causal — ZBE Vitoria-Gasteiz",
    v8Title: "Counterfactual Meteorológico<br><span>ZBE Sep 2025 → Actualidad</span>",
    v8Subtitle: "Comparativa entre la contaminación observada y el escenario contrafactual (qué habría pasado sin ZBE).",
    legendTitle: "Leyenda",
    legendObs: "Observado (real)",
    legendCFPure: "CF Meteo-Puro",
    legendCFLags: "CF Con-Lags",
    legendBand: "Banda de incertidumbre",
    contaminant: "Contaminante",
    zone: "Zona",
    zoneIn: "ZBE (interior)",
    zoneOut: "FUERA DE ZBE (control)",
    v9Tag: "Métodos Econométricos — Evaluación de Impacto",
    v9Title: "Evaluación Causal de la ZBE<br><span>Event Study & Synthetic Control</span>",
    v9Station: "Estación interior ZBE",
    imgNotFound: "Imagen no encontrada. Verifique la generación del plot.",
    v10Tag: "EEA Standards — Air Quality Risk",
    v10Title: "Calidad del aire prevista para",
    v10Calculating: "CALCULANDO...",
    v10Today: "hoy",
    v10Date: "fecha",
    auditTitle: "<strong>Evolución y previsión:</strong> histórico validado + D+1 + D+2",
    backtestTitle: "<strong>Último día validado</strong> — control de precisión",
    colParam: "Parametro",
    colPred: "Predicción emitida",
    trafficYoYTitle: "Estudio de Impacto: Evolución del Tráfico Perimetral (YoY)",
    trafficYoYSubtitle: "Comparativa entre el tráfico actual y el mismo periodo del año anterior (alineado por día de la semana).",
    trafficCurrent: "Tráfico Actual (ZBE)",
    trafficPrevious: "Mismo periodo del año anterior",
    trafficPreviousPre: "Año anterior (pre-ZBE)",
    trafficPreviousYear1: "Año anterior (1er año ZBE)",
    trafficPreviousMixed: "Año anterior (pre-ZBE y 1er año ZBE)",
    trafficVolume: "Vehículos/Día (Media Perímetro)",
    colReal: "Medición real",
    colDev: "Desviación",
    colStatus: "Estado",
    perfTitle: "<strong>Rendimiento D+1 · forecast v10</strong> — validación temporal",
    perfTitleD2: "<strong>Rendimiento D+2 · forecast v10</strong> — validación temporal",
    colContZone: "Contaminante + Zona",
    colR2: "R²",
    colNFeatures: "N. Features",
    mapeNote: "MAPE calculado solo sobre filas con valor observado > 5 µg/m³.",
    mapeThreshold: "Criterio operativo: mejora frente a persistencia y cobertura suficiente del intervalo.",
    mapTag: "Vitoria-Gasteiz — Red Sensores Municipales",
    mapTitle: "Mapa de <span>Estaciones</span>",
    mapSubtitle: "Medias parciales por estación y predicción vigente para la fecha indicada. Haga clic en un punto para ver detalles.",
    mapLegendTitle: "Semáforo ICA",
    icaGood: "Buena (≤25)",
    icaMod: "Moderada (25-50)",
    icaBad: "Mala (50-75)",
    icaVeryBad: "Muy Mala (>75)",
    trafficTitle: "Mapa de Tráfico",
    sumObs: "Observado (media)",
    sumCF: "Contrafactual (meteo-puro)",
    sumEffect: "Efecto ZBE (meteo-puro)",
    sumRange: "Sensibilidad (exploratoria)",
    sumPeriod: "Sep 2025 → Mar 2026",
    sumBound: "Límite conservador",
    sumMeteoLag: "Con-Lags no es un bound causal",
    sumRobRed: "✓ Reducción robusta",
    sumUncertain: "⚠ Incierto",
    sumIncrease: "✕ Incremento neto",
    v8Fig1: "<strong>Figura 1</strong> — Observado vs Counterfactual",
    v8Fig2: "<strong>Figura 2</strong> — Banda de incertidumbre del efecto ZBE",
    v8Fig3: "<strong>Figura 3</strong> — Gap medio por contaminante y zona",
    v9Fig1: "<strong>Figura 1</strong> — Control Sintético (Serie Suavizada)",
    v9Fig2: "<strong>Figura 2</strong> — Event Study DiD",
    v10Tomorrow: "mañana",
    v10PlanningTag: "Planificación D+2 · menor confianza",
    v10PlanningTitle: "Perspectiva de calidad del aire para",
    v10PlanningNote: "Horizonte orientado a planificación. Los intervalos son más amplios que en D+1.",
    mapPrediction: "Predicción",
    sumSinZBE: "sin ZBE",
    sumAbsoluto: "absoluto",
    fig1Obs: "Observado",
    fig1CFPure: "CF Meteo-Puro",
    fig1CFLags: "CF Con-Lags",
    fig2NoEffect: "Sin efecto (0%)",
    fig2GapPure: "Gap Meteo-Puro",
    fig2GapLags: "Gap Con-Lags",
    fig3Label: "Gap % (METEO-PURO)",
    v9Adj: "Métrica de Ajuste (Pre-ZBE)",
    v9Impact: "Impacto Post-ZBE Estimado",
    v9Diag: "Diagnóstico Causal",
    v9Trend: "Pre-Trend Test",
    v9ParallelOK: "✓ OK Parallel Trends",
    v9ParallelBad: "⚠ Pre-tendencia significativa",
    v9TrendNoData: "Sin datos",
    causalBlocked: "Efecto causal no publicado",
    causalQc: "El control de calidad ha detectado un cambio de régimen en una estación tratada (PAUL PM10/PM2.5). Se excluye para no atribuir a la ZBE un artefacto de sensor.",
    causalNoSkill: "El contrafactual no supera el gate de skill (R² rolling ≥ 0,4 y mejor que persistencia). No se publica como efecto causal.",
    causalMissing: "No hay validación causal disponible todavía. Se oculta el efecto hasta que el pipeline la genere.",
    v10Good: "🟢 BUENA",
    v10Mod: "🟡 MODERADA",
    v10Bad: "🔴 MALA",
    v10NoData: "⚪ SIN DATOS",
    backReal: "Medición Real",
    backPred: "Predicción D+1 validada",
    forecastD1: "Continuidad prevista hasta D+1",
    forecastD2: "Trayectoria prevista D+1 → D+2",
    backCurrentPred: "Predicción de mañana",
    backWait: "⏳ Esperando sensores municipales",
    backExcel: "✓ Precisión Excelente",
    backGood: "✓ Precisión Buena",
    backAccept: "✓ Precisión Aceptable",
    backReview: "🔴 Revisar Modelo",
    mapDailyAyer: "Medias parciales de hoy",
    mapPredManana: "▶ Predicción mañana",
    mapZone: "Zona",
    mapNote: "⚠ Los valores son medias diarias, no lecturas horarias",
    colRMSE: "RMSE (µg/m³)",
    colMAE: "MAE (µg/m³)",
    colMAPE: "MAPE %",
    metaImprovement: "mejora meta-model",
    fsTag: "Explicación de la predicción",
    fsTitle: "Foresight <span> con IA</span>",
    fsSubtitle: "Factores meteorológicos y de contexto asociados a la predicción vigente del modelo.",
    fsNarrativeTitle: "Explicación de la predicción",
    fsShapTitle: "Top Factores de Impacto (SHAP)",
    fsLoading: "Cargando narrativa...",
    fsNoData: "No hay análisis disponible para esta selección.",
    fsNoNarrative: "Este modelo no ha generado narrativa LLM (posiblemente porque no es el target principal o falla la API).",
    fsImpactLabel: "Impacto en µg/m³",
    contactBtn: "Contactar con el creador",
  },
  eu: {
    mainTitle: "Vitoria-Gasteiz — Airearen Kalitatearen eta EGEaren Analisia",
    nav1: "1. Iragarpen Operatiboa",
    nav2: "2. Eguneroko Monitore Interaktiboa",
    nav3: "3. Baliozkotze Kausala",
    nav4: "4. Estazioen Mapa",
    nav5: "5. Trafikoaren Mapa",
    nav6: "6. Foresight AI",
    themeDark: "Modu Iluna",
    themeLight: "Modu Argia",
    v8Tag: "Analisi Kausala — Gasteizko EGE",
    v8Title: "Kontrafaktual Meteorologikoa<br><span>EGE 2025eko Iraila → Gaur egun</span>",
    v8Subtitle: "Behatutako kutsaduraren eta agertoki kontrafaktualaren arteko alderaketa.",
    legendTitle: "Legenda",
    legendObs: "Behatua (reala)",
    legendCFPure: "KM Meteo-Purua",
    legendCFLags: "KM Lag-ekin",
    legendBand: "Ziurgabetasun-banda",
    contaminant: "Kutsatzailea",
    zone: "Eremua",
    zoneIn: "EGE (barruan)",
    zoneOut: "EGEtik KANPO (kontrola)",
    v9Tag: "Metodo ekonometrikoak — Eraginaren ebaluazioa",
    v9Title: "EGEren Ebaluazio Kausala<br><span>Event Study & Synthetic Control</span>",
    v9Station: "EGE barneko estazioa",
    imgNotFound: "Irudia ez da aurkitu. Egiaztatu fitxategiaren izena.",
    v10Tag: "EEA Arauak — Airearen Kalitatearen Arriskua",
    v10Title: "Airearen kalitatearen aurreikuspena:",
    v10Calculating: "KALKULATZEN...",
    v10Today: "gaur",
    v10Date: "data",
    auditTitle: "<strong>Bilakaera eta iragarpena:</strong> baliozkotutako historikoa + D+1 + D+2",
    backtestTitle: "<strong>Baliozkotutako azken eguna</strong> — doitasun-kontrola",
    trafficYoYTitle: "Eraginaren Azterketa: Perimetroko Trafikoaren Bilakaera (YoY)",
    trafficYoYSubtitle: "Gaur egungo trafikoaren eta aurreko urteko epe beraren arteko alderaketa (asteko egunaren arabera lerrokatuta).",
    trafficCurrent: "Gaur egungo Trafikoa (ZBE)",
    trafficPrevious: "Aurreko urteko aldi bera",
    trafficPreviousPre: "Aurreko urtea (EGE aurretik)",
    trafficPreviousYear1: "Aurreko urtea (EGEko 1. urtea)",
    trafficPreviousMixed: "Aurreko urtea (EGE aurretik eta 1. urtea)",
    trafficVolume: "Ibilgailuak/Eguna (Perimetroko batez bestekoa)",
    colParam: "Parametroa",
    colPred: "Jaulkitako iragarpena",
    colReal: "Benetako neurketa",
    colDev: "Desbideratzea",
    colStatus: "Egoera",
    perfTitle: "<strong>D+1 errendimendua · forecast v10</strong> — denborazko baliozkotzea",
    perfTitleD2: "<strong>D+2 errendimendua · forecast v10</strong> — denborazko baliozkotzea",
    colContZone: "Kutsatzailea + Eremua",
    colR2: "R²",
    colNFeatures: "Ezaugarri Kopurua",
    mapeNote: "MAPE behatutako balioa > 5 µg/m³ duten errenkadetan soilik kalkulatua.",
    mapeThreshold: "Irizpide operatiboa: persistentziarekiko hobekuntza eta tartearen estaldura nahikoa.",
    mapTag: "Vitoria-Gasteiz — Udaltzaingoaren sentsoreen sarea",
    mapTitle: "Estazioen <span>Mapa</span>",
    mapSubtitle: "Estazioetako batez besteko partzialak eta adierazitako datarako indarrean dagoen iragarpena. Egin klik puntu batean xehetasunak ikusteko.",
    mapLegendTitle: "ICA Semaforoa",
    icaGood: "Ona (≤25)",
    icaMod: "Ertaina (25-50)",
    icaBad: "Txarra (50-75)",
    icaVeryBad: "Oso txarra (>75)",
    trafficTitle: "Trafikoaren Mapa",
    sumObs: "Behatua (batez bestekoa)",
    sumCF: "Kontrafaktuala (meteo-purua)",
    sumEffect: "EGEaren efektua (meteo-purua)",
    sumRange: "Sentikortasun (esploraziozkoa)",
    sumPeriod: "2025 Ira → 2026 Mar",
    sumBound: "Muga kontserbadorea",
    sumMeteoLag: "Con-Lags ez da bound kausala",
    sumRobRed: "✓ Murrizketa sendoa",
    sumUncertain: "⚠ Ziurgabea",
    sumIncrease: "✕ Gehikuntza garbia",
    v9Adj: "Egokitze Metrika (EGE aurretik)",
    v9Impact: "EGE osteko eragin estimatua",
    v9Diag: "Diagnostiko Kausala",
    v9Trend: "Pre-Trend Test-a",
    v9ParallelOK: "✓ OK Parallel Trends",
    v9ParallelBad: "⚠ Aurre-tendentzia esanguratsua",
    v9TrendNoData: "Daturik gabe",
    causalBlocked: "Eragin kausala ez da argitaratu",
    causalQc: "Kalitate-kontrolak tratatutako estazio batean erregimen-aldaketa detektatu du (PAUL PM10/PM2.5). Kanpoan uzten da ZBEari sentsore-artefaktua egotzi ez dadin.",
    causalNoSkill: "Kontrafaktualak ez du skill gate-a gainditzen (R² rolling ≥ 0,4 eta persistencia baino hobea). Ez da eragin kausal gisa argitaratzen.",
    causalMissing: "Oraindik ez dago baliozkotze kausalik. Eragina ezkutuan mantentzen da pipelineak sortu arte.",
    v10Good: "🟢 ONA",
    v10Mod: "🟡 ERTAINA",
    v10Bad: "🔴 TXARRA",
    v10NoData: "⚪ DATURIK GABE",
    backReal: "Neurketa Erreala",
    backPred: "Baliozkotutako D+1 iragarpena",
    forecastD1: "D+1era arteko aurreikuspen-jarraipena",
    forecastD2: "D+1 → D+2 aurreikuspen-ibilbidea",
    backCurrentPred: "Biharko iragarpena",
    backWait: "⏳ Sentsoreen zain",
    backExcel: "✓ Doitasun bikaina",
    backGood: "✓ Doitasun ona",
    backAccept: "✓ Doitasun onargarria",
    backReview: "🔴 Eredua berrikusi",
    mapDailyAyer: "Gaurko batez besteko partzialak",
    mapPredManana: "▶ Biharko aurreikuspena",
    mapZone: "Eremua",
    mapNote: "⚠ Balioak eguneroko batez bestekoak dira, ez orduko irakurketak",
    colRMSE: "RMSE (µg/m³)",
    colMAE: "MAE (µg/m³)",
    colMAPE: "MAPE %",
    metaImprovement: "hobekuntza meta-eredua",
    v8Fig1: "<strong>1. Irudia</strong> — Behatua vs Kontrafaktuala",
    v8Fig2: "<strong>2. Irudia</strong> — EGE efektuaren ziurgabetasun-banda",
    v8Fig3: "<strong>3. Irudia</strong> — Gap-a batez beste, kutsatzaile eta eremuka",
    v9Fig1: "<strong>1. Irudia</strong> — Kontrol Sintetikoa (Serie Leundua)",
    v9Fig2: "<strong>2. Irudia</strong> — Event Study DiD",
    v10Tomorrow: "bihar",
    v10PlanningTag: "D+2 plangintza · konfiantza txikiagoa",
    v10PlanningTitle: "Airearen kalitatearen ikuspegia:",
    v10PlanningNote: "Plangintzarako horizontea. Tarteak D+1ekoak baino zabalagoak dira.",
    mapPrediction: "Iragarpena",
    sumSinZBE: "EGE barik",
    sumAbsoluto: "absolutua",
    fig1Obs: "Behatua",
    fig1CFPure: "KM Meteo-Purua",
    fig1CFLags: "KM Lag-ekin",
    fig2NoEffect: "Eraginik gabe (%0)",
    fig2GapPure: "Meteo-Puro Gap-a",
    fig2GapLags: "Lag-ekin Gap-a",
    fig3Label: "Gap-a % (METEO-PURUA)",
    didTitle: "1. Taula — Difference-in-Differences v8",
    didCont: "Kutsatzailea",
    didPVal: "P-Balioa",
    didRel: "Eragin Erlatiboa",
    didSig: "Signifikatiboa",
    didYes: "Bai (p<0.05)",
    didNo: "Ez",
    fsTag: "Aurreikuspenaren azalpena",
    fsTitle: "Foresight <span> AIarekin</span>",
    fsSubtitle: "Indarrean dagoen ereduaren iragarpenari lotutako faktore meteorologikoak eta testuingurukoak.",
    fsNarrativeTitle: "Aurreikuspenaren azalpena",
    fsShapTitle: "Eragin-faktore nagusiak (SHAP)",
    fsLoading: "Narratiba kargatzen...",
    fsNoData: "Ez dago analisirik erabilgarri hautapen honetarako.",
    fsNoNarrative: "Eredu honek ez du LLM narratibarik sortu (ziurrenik ez delako helburu nagusia edo APIak huts egin duelako).",
    fsImpactLabel: "Eragina µg/m³-tan",
    contactBtn: "Sortzailearekin harremanetan jarri",
  }
};

function updateI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const t = translations[currentLang];
    if (t && t[key]) {
      // Usamos innerHTML si la traducción tiene HTML (como <strong>) o si el elemento lo permite
      if (t[key].includes('<') || el.classList.contains('card-label') || el.classList.contains('did-title') || el.classList.contains('fig-title')) {
          el.innerHTML = t[key];
      } else {
          el.innerText = t[key];
      }
    }
  });
  
  // Actualizar el botón de idioma
  const langBtn = document.getElementById('langBtn');
  if (langBtn) {
    if (currentLang === 'es') {
      const ikurrina = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12" viewBox="0 0 28 20" style="vertical-align: middle; margin-right: 6px; border-radius: 1px;"><rect width="28" height="20" fill="#DA121A"/><path d="M0 0 L28 20 M28 0 L0 20" stroke="#009543" stroke-width="3"/><path d="M14 0 V20 M0 10 H28" stroke="#FFFFFF" stroke-width="3"/></svg>';
      langBtn.innerHTML = ikurrina + 'EUSKARA';
    } else {
      langBtn.innerHTML = 'CASTELLANO';
    }
  }
  
  // Actualizar el texto del botón del tema
  const themeBtn = document.getElementById('themeBtn');
  if (themeBtn) {
    themeBtn.innerText = currentTheme === 'dark' 
      ? translations[currentLang].themeLight 
      : translations[currentLang].themeDark;
  }
  
  // Actualizar el iframe de tráfico
  const iframe = document.getElementById('trafficIframe');
  if (iframe) {
    const baseUrl = iframe.src.split('?')[0];
    iframe.src = `traffic_map.html?lang=${currentLang}`;
  }
}

function predictionDateRelation() {
    const parts = predDate.split('/').map(Number);
    if (parts.length !== 3 || parts.some(Number.isNaN)) return null;
    const [day, month, year] = parts;
    const now = new Date();
    const targetUtc = Date.UTC(year, month - 1, day);
    const todayUtc = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
    return Math.round((targetUtc - todayUtc) / 86400000);
}

function predictionDayLabel() {
    const relation = predictionDateRelation();
    const t = translations[currentLang];
    if (relation === 0) return t.v10Today;
    if (relation === 1) return t.v10Tomorrow;
    return t.v10Date;
}

function predictionDateOffsetLabel(offsetDays) {
    const parts = predDate.split('/').map(Number);
    if (parts.length !== 3 || parts.some(Number.isNaN)) return predDate;
    const [day, month, year] = parts;
    const date = new Date(Date.UTC(year, month - 1, day + offsetDays));
    return date.toLocaleDateString(
        currentLang === 'es' ? 'es-ES' : 'eu-ES',
        { day: 'numeric', month: 'short', timeZone: 'UTC' }
    );
}

function predictionDateOffsetFull(offsetDays) {
    const parts = predDate.split('/').map(Number);
    if (parts.length !== 3 || parts.some(Number.isNaN)) return predDate;
    const [day, month, year] = parts;
    const date = new Date(Date.UTC(year, month - 1, day + offsetDays));
    const resultDay = String(date.getUTCDate()).padStart(2, '0');
    const resultMonth = String(date.getUTCMonth() + 1).padStart(2, '0');
    return `${resultDay}/${resultMonth}/${date.getUTCFullYear()}`;
}

function adaptNarrativeDate(text) {
    if (!text || typeof text !== 'string') return text;
    const relation = predictionDateRelation();
    if (relation === 0 && currentLang === 'es') return text.replaceAll('mañana', 'hoy');
    if (relation === 0 && currentLang === 'eu') return text.replaceAll('Biharko', 'Gaurko');
    if (relation !== 1 && currentLang === 'es') return text.replaceAll('mañana', `el ${predDate}`);
    if (relation !== 1 && currentLang === 'eu') return text.replaceAll('Biharko', `${predDate}ko`);
    return text;
}

function updateRiskDate() {
    const el = document.getElementById('riskDateSpan');
    if (el) {
        el.innerHTML = `<span style="text-transform: uppercase;">${predictionDayLabel()}</span> ${predDate}`;
    }
    const d2El = document.getElementById('riskDateD2Span');
    if (d2El) d2El.textContent = predictionDateOffsetFull(1);
}

function toggleLang() {
  currentLang = currentLang === 'es' ? 'eu' : 'es';
  localStorage.setItem('vitoria_lang', currentLang);
  updateI18n();
  updateRiskDate();
  renderPipelineBanner();
  
  // Re-renderizar componentes que dependen de traducciones
  renderSummaryCards();
  renderFig1();
  renderFig2();
  renderFig3();
  renderDidTable();
  renderV9Cards();
  renderMetricsTable();
  renderTrafficYoY();
  renderForesight();
  if (document.getElementById('view-v10').classList.contains('active')) renderDashboard3();
}
// ── NAVEGACIÓN Y THEME ──
function switchMainView(view, btn) {
  document.querySelectorAll('.view-container').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.top-nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  btn.classList.add('active');
  if (view === 'v8') { 
    setTimeout(() => {
        renderSummaryCards(); 
        renderFig1(); 
        renderFig2(); 
        renderFig3(); 
        renderDidTable(); 
        window.dispatchEvent(new Event('resize'));
    }, 400);
  }
  if (view === 'v10') {
    setTimeout(() => {
        renderDashboard3();
    }, 100);
  }
  if (view === 'map') { setTimeout(() => initMap(), 100); }
  if (view === 'traffic') { 
    setTimeout(() => {
        renderTrafficYoY();
        window.dispatchEvent(new Event('resize'));
    }, 400);
  }
}

function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme');
    btn.innerText = ' Modo Oscuro';
    currentTheme = 'light';
  } else {
    html.setAttribute('data-theme', 'dark');
    btn.innerText = ' Modo Claro';
    currentTheme = 'dark';
  }
  
  renderFig1();
  renderFig2();
  renderFig3();
  if(document.getElementById('view-v10').classList.contains('active')) renderDashboard3();
  
  if (_mapInitialized && mapTileLayer) {
      if (mapInstance) mapInstance.closePopup();
  }
}

// ===========================================================================
// JAVASCRIPT ORIGINAL DE D1 (ADAPTADO A VARIABLES CSS)
// ===========================================================================
let currentCont = 'NO2'; let currentZone = 'zbe'; let fig1Chart = null, fig2Chart = null, fig3Chart = null;

const monthsEU = ['urt.', 'ots.', 'mar.', 'api.', 'mai.', 'eka.', 'uzt.', 'abu.', 'ira.', 'urr.', 'aza.', 'abe.'];
// Las fechas del pipeline son días locales sin hora: interpretarlas como UTC
// evita que el navegador las desplace un día según su zona horaria.
function parseDateUTC(d) {
  const s = String(d);
  return new Date(s.includes('T') ? s : `${s}T00:00:00Z`);
}
function formatDate(d) {
  const date = parseDateUTC(d);
  if (currentLang === 'eu') {
    return `${date.getUTCDate()} ${monthsEU[date.getUTCMonth()]}`;
  }
  return date.toLocaleDateString('es-ES', { day:'numeric', month:'short', timeZone:'UTC' });
}
function getClass(val) { if (val < -5)  return 'neg'; if (val > 5)   return 'pos'; return 'neutral'; }

// Gate causal: sin validación temporal y QC favorable no se publican efectos.
function causalGate(cont, zone) {
  const v = CAUSAL_VALIDATION[`${cont}_${zone}_METEO-PURO`] || CAUSAL_VALIDATION[`${cont}_${zone}`];
  if (!v) return { publishable: false, missing: true, qc_flag: false };
  return v;
}

function setCausalChartsVisible(visible) {
  ['fig1', 'fig2', 'fig3'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? 'block' : 'none';
  });
}

function renderPipelineBanner() {
  const el = document.getElementById('pipelineBanner');
  if (!el) return;
  const issues = [];
  let alert = false;
  const st = PIPELINE_STATUS || {};

  if (st.prediction_date) {
    const pred = parseDateUTC(st.prediction_date);
    const now = new Date();
    const todayUTC = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
    const predUTC = Date.UTC(pred.getUTCFullYear(), pred.getUTCMonth(), pred.getUTCDate());
    const days = Math.round((todayUTC - predUTC) / 86400000);
    if (days > 1) {
      issues.push(currentLang === 'es'
        ? `Predicción de hace ${days} días (${st.prediction_date})`
        : `${days} eguneko iragarpena (${st.prediction_date})`);
      alert = days > 2;
    }
  } else {
    issues.push(currentLang === 'es' ? 'Sin fecha de predicción publicada' : 'Iragarpen-datarik ez');
    alert = true;
  }

  // Los avisos internos de drift, QC, spillover y meteorología quedan en
  // pipeline_status.json para operación, pero no se muestran al visitante.

  if (!issues.length) {
    el.className = 'pipeline-banner';
    el.textContent = '';
    return;
  }
  el.className = `pipeline-banner ${alert ? 'alert' : 'warn'}`;
  el.textContent = issues.join(' · ');
}

function renderSummaryCards() {
  const s = SUMMARY_STATS[`${currentCont}_${currentZone}`];
  const t = translations[currentLang];

  const gate = causalGate(currentCont, currentZone);
  if (!gate.publishable) {
    setCausalChartsVisible(false);
    const reason = gate.missing ? t.causalMissing : (gate.qc_flag ? t.causalQc : t.causalNoSkill);
    document.getElementById('summaryCards').innerHTML =
      `<div class="summary-card" style="grid-column:1/-1"><div class="card-label">${t.causalBlocked}</div><div class="card-sub">${reason}</div></div>`;
    return;
  }
  setCausalChartsVisible(true);
  if (!s || s.pure === null) return;
  const lagsVal = (s.lags === null || s.lags === undefined) ? s.pure : s.lags;
  const low = Math.min(s.pure, lagsVal), high = Math.max(s.pure, lagsVal), gap_abs = ((s.obs - s.cf_pure)).toFixed(2);
  // El rango se deriva de los datos servidos, no de una fecha fija en el HTML.
  const cfSeries = cfData[`${currentCont}_${currentZone}`]?.['METEO-PURO'];
  const periodRange = cfSeries && cfSeries.dates.length
    ? `${formatDate(cfSeries.dates[0])} → ${formatDate(cfSeries.dates[cfSeries.dates.length - 1])}`
    : t.v9TrendNoData;
  const cards = [ 
      { label: t.sumObs, value: s.obs.toFixed(2), sub: s.unit, cls: 'neutral', range: periodRange },
      { label: t.sumCF, value: s.cf_pure.toFixed(2), sub: `${s.unit} ${t.sumSinZBE}`, cls: 'neutral', range: t.sumBound }, 
      { label: t.sumEffect, value: `${s.pure > 0 ? '+' : ''}${s.pure.toFixed(1)}%`, sub: `${gap_abs} ${s.unit} ${t.sumAbsoluto}`, cls: getClass(s.pure), range: t.sumBound }, 
      { label: t.sumRange, value: `[${low.toFixed(1)}%, ${high.toFixed(1)}%]`, sub: t.sumMeteoLag, cls: low < 0 && high < 0 ? 'neg' : (low > 0 && high > 0 ? 'pos' : 'neutral'), range: low < 0 && high < 0 ? t.sumRobRed : (low < 0 ? t.sumUncertain : t.sumIncrease) } 
  ];
  document.getElementById('summaryCards').innerHTML = cards.map(c => `<div class="summary-card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div><div class="card-sub">${c.sub}</div><div class="card-range">${c.range}</div></div>`).join('');
}

function renderFig1() {
  const key = `${currentCont}_${currentZone}`; 
  const gate = causalGate(currentCont, currentZone);
  if (!gate.publishable) { if(fig1Chart) fig1Chart.destroy(); return; }
  if (!cfData || !cfData[key] || !cfData[key]['METEO-PURO']) {
    console.warn("No data for Fig1:", key);
    if(fig1Chart) fig1Chart.destroy();
    return;
  }
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'] || null, dates = pure.dates.map(formatDate); const obsSmooth = movingAvg(pure.observed, 7);
  const lagsSeries = lags ? movingAvg(lags.counterfactual, 7) : dates.map(() => null);
  const t = translations[currentLang];
  const ctx = document.getElementById('fig1').getContext('2d');
  if (fig1Chart) fig1Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');
  
  fig1Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_fillUpper', data: pure.counterfactual, borderColor: 'transparent', backgroundColor: getCssVar('--observed') + '20', fill: '+1', pointRadius: 0, tension: 0.3 }, { label: t.fig1Obs, data: obsSmooth, borderColor: getCssVar('--observed'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false, order: 1 }, { label: t.fig1CFPure, data: movingAvg(pure.counterfactual, 7), borderColor: getCssVar('--cf-pure'), backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, tension: 0.3, fill: false, order: 2 }, { label: t.fig1CFLags, data: lagsSeries, borderColor: getCssVar('--cf-lags'), backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [3,3], pointRadius: 0, tension: 0.3, fill: false, order: 3 } ] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: text, font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; const unit = currentCont === 'ICA' ? '' : ' µg/m³'; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2)}${unit}`; } } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: grid } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(1) + (currentCont === 'ICA' ? '' : ' µg/m³') }, grid: { color: grid }, title: { display: true, text: currentCont === 'ICA' ? 'Índice' : 'µg/m³', color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig2() {
  const key = `${currentCont}_${currentZone}`; 
  const gate = causalGate(currentCont, currentZone);
  if (!gate.publishable) { if(fig2Chart) fig2Chart.destroy(); return; }
  if (!cfData || !cfData[key] || !cfData[key]['METEO-PURO']) {
    console.warn("No data for Fig2:", key);
    if(fig2Chart) fig2Chart.destroy();
    return;
  }
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'] || null, dates = pure.dates.map(formatDate);
  const gapPureSmooth = movingAvg(pure.gap_pct, 14);
  const gapLagsSmooth = lags ? movingAvg(lags.gap_pct, 14) : dates.map(() => null);
  const zero = dates.map(() => 0);
  // Con un solo bound no hay banda: se usan nulls, no ceros.
  const bandUpper = gapPureSmooth.map((p, i) => (p === null || gapLagsSmooth[i] === null) ? null : Math.max(p, gapLagsSmooth[i]));
  const bandLower = gapPureSmooth.map((p, i) => (p === null || gapLagsSmooth[i] === null) ? null : Math.min(p, gapLagsSmooth[i]));
  const t = translations[currentLang];
  const ctx = document.getElementById('fig2').getContext('2d');
  if (fig2Chart) fig2Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');

  fig2Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_bandUpper', data: bandUpper, borderColor: 'transparent', backgroundColor: getCssVar('--accent') + '26', fill: '+1', pointRadius: 0, tension: 0.4 }, { label: '_bandLower', data: bandLower, borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0, tension: 0.4 }, { label: t.fig2NoEffect, data: zero, borderColor: text, borderWidth: 1, borderDash: [2,4], pointRadius: 0, fill: false, order: 10 }, { label: t.fig2GapPure, data: gapPureSmooth, borderColor: getCssVar('--cf-pure'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false, order: 1 }, { label: t.fig2GapLags, data: gapLagsSmooth, borderColor: getCssVar('--cf-lags'), backgroundColor: 'transparent', borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.4, fill: false, order: 2 } ] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: text, font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}%`; } } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: grid } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(0) + '%' }, grid: { color: grid }, title: { display: true, text: 'Gap %', color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig3() {
  const ctxEl = document.getElementById('fig3');
  if (!ctxEl) return;
  const keys = Object.keys(SUMMARY_STATS).filter(k => {
    const [cont, zone] = k.split('_');
    return SUMMARY_STATS[k].pure !== null && causalGate(cont, zone).publishable;
  });
  if (!keys.length) {
    if (fig3Chart) fig3Chart.destroy();
    ctxEl.style.display = 'none';
    return;
  }
  ctxEl.style.display = 'block';
  const vals = keys.map(k => SUMMARY_STATS[k].pure);
  const labels = keys.map(k => k.replace('PM2.5','PM₂.₅').replace('_',' '));
  const colors = vals.map(v => v < -5 ? getCssVar('--green')+'CC' : v > 5 ? getCssVar('--red')+'CC' : getCssVar('--yellow')+'CC');
  const borders = vals.map(v => v < -5 ? getCssVar('--green') : v > 5 ? getCssVar('--red') : getCssVar('--yellow'));
  const t = translations[currentLang];
  const ctx = ctxEl.getContext('2d');
  if (fig3Chart) fig3Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');

  fig3Chart = new Chart(ctx, { type: 'bar', data: { labels: labels, datasets: [{ label: t.fig3Label, data: vals, backgroundColor: colors, borderColor: borders, borderWidth: 1, borderRadius: 3 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => ` Efecto: ${ctx.parsed.y.toFixed(1)}%` } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 11 } }, grid: { display: false } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v + '%' }, grid: { color: grid }, title: { display: true, text: 'Gap %', color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderDidTable() {
  const t = translations[currentLang];
  const rows = Object.entries(DID_RESULTS).map(([cont, d]) => {
    if (d.publishable === false) {
      return `<tr>
        <td><strong>${cont}</strong></td>
        <td colspan="6">${t.causalBlocked} — ${d.qc_flag ? t.causalQc : t.causalNoSkill}</td>
      </tr>`;
    }
    const hasBeta = d.beta3 !== null && d.beta3 !== undefined;
    const sig = Boolean(d.significant);
    const bCls = hasBeta ? (d.beta3 < -0.1 ? 'beta-neg' : d.beta3 > 0.1 ? 'beta-pos' : 'beta-neu') : 'beta-neu';
    const sign = hasBeta && d.beta3 > 0 ? '+' : '';
    const pFmt = d.pvalue === null || d.pvalue === undefined
      ? 'N/D'
      : (d.pvalue < 0.0001 ? '<0.0001' : d.pvalue.toFixed(4));
    const sigText = sig ? t.didYes : t.didNo;
    const betaTxt = hasBeta ? `${sign}${d.beta3.toFixed(3)} µg/m³` : 'N/D';
    const ciTxt = d.ci_low !== null && d.ci_low !== undefined && d.ci_high !== null && d.ci_high !== undefined
      ? `[${d.ci_low.toFixed(3)}, ${d.ci_high.toFixed(3)}]`
      : 'N/D';
    const r2Txt = d.r2 !== null && d.r2 !== undefined ? d.r2.toFixed(3) : 'N/D';
    
    return `<tr>
      <td><strong>${cont}</strong></td>
      <td class="${bCls}">${betaTxt}</td>
      <td class="${sig ? 'sig-yes' : 'sig-no'}">${pFmt} ${sig ? '✓' : ''}</td>
      <td>${ciTxt}</td>
      <td>${r2Txt}</td>
      <td>${(d.n_obs || d.n || 0).toLocaleString()}</td>
      <td class="${sig ? 'sig-yes' : 'sig-no'}">${sigText}</td>
    </tr>`;
  });
  const html = rows.join('');
  const el1 = document.getElementById('didTable'); if(el1) el1.innerHTML = html;
}

function selectCont(cont, btn) { 
  currentCont = cont; 
  document.querySelectorAll('#contTabs .tab').forEach(t => t.classList.remove('active')); 
  btn.classList.add('active'); 
  renderSummaryCards(); 
  renderFig1(); 
  renderFig2(); 
  window.dispatchEvent(new Event('resize'));
}
function selectZone(zone, btn) { 
  currentZone = zone; 
  document.querySelectorAll('#zoneTabs .tab').forEach(t => t.classList.remove('active')); 
  btn.classList.add('active'); 
  renderSummaryCards(); 
  renderFig1(); 
  renderFig2(); 
  window.dispatchEvent(new Event('resize'));
}

// ===========================================================================
// LÓGICA V9
// ===========================================================================
let currentContV9 = 'NO2', currentStationV9 = 'PAUL';

function renderV9Cards() {
  const contKey = currentContV9 === 'PM2.5' ? 'PM25' : currentContV9;
  const key = `${contKey}_${currentStationV9}`; const data = v9Stats[key] || { preR2: 'N/A', gap: 'N/A', desc: 'Sin datos' };
  const t = translations[currentLang];
  const colorClass = String(data.gap).includes('-') ? 'neg' : (String(data.gap).includes('+') ? 'pos' : 'neutral');
  // La tarjeta de tendencias refleja el event study real, no un texto fijo.
  const trend = EVENT_STUDY_TREND[currentContV9];
  let trendText = t.v9TrendNoData, trendClass = 'neutral';
  if (trend) {
    if (trend.pre_trend_ok) {
      trendText = t.v9ParallelOK; trendClass = 'pos';
    } else {
      trendText = `${t.v9ParallelBad} (${trend.n_sig}/${trend.n_pre})`; trendClass = 'neg';
    }
  }
  document.getElementById('summaryCardsV9').innerHTML = `<div class="summary-card"><div class="card-label">${t.v9Adj}</div><div class="card-value neutral">R² = ${data.preR2}</div></div><div class="summary-card"><div class="card-label">${t.v9Impact}</div><div class="card-value ${colorClass}">${data.gap}</div></div><div class="summary-card"><div class="card-label">${t.v9Diag}</div><div class="card-value neutral" style="font-size: 16px; margin-top:8px;">${data.desc}</div></div><div class="summary-card"><div class="card-label">${t.v9Trend}</div><div class="card-value ${trendClass}" style="font-size: 16px; margin-top:8px;">${trendText}</div></div>`;
}

function updateV9Images() {
  const contName = currentContV9 === 'PM2.5' ? 'PM25' : currentContV9;
  const imgSc = document.getElementById('img-sc'), imgEs = document.getElementById('img-es');
  if(!imgSc || !imgEs) return;
  
  imgSc.nextElementSibling.style.display = 'none';
  imgEs.nextElementSibling.style.display = 'none';
  imgSc.style.display = 'block'; imgEs.style.display = 'block'; 
  
  const basePath = "__IMG_BASE_PATH__";
  console.log("Loading V9 Images from:", basePath);
  
  imgSc.src = `${basePath}synthetic_control_${contName}_${currentStationV9}.png`; 
  imgEs.src = `${basePath}event_study_${contName}.png`;
}

function selectContV9(cont, btn) { currentContV9 = cont; document.querySelectorAll('#contTabsV9 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderV9Cards(); updateV9Images(); }
function selectStationV9(station, btn) { currentStationV9 = station; document.querySelectorAll('#stationTabsV9 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderV9Cards(); updateV9Images(); }

// ===========================================================================
// LÓGICA DASHBOARD 3 (OPERATIVA Y PREDICCIÓN)
// ===========================================================================
let currentContV10 = 'NO2', currentZoneV10 = 'zbe', perfChart = null;

function renderDashboard3() {
  const zoneData = manana[currentZoneV10] || {};
  const pick = (k) => (zoneData[k] === undefined ? null : zoneData[k]);
  const no2 = pick('NO2');
  const pm25 = pick('PM2.5');
  const pm10 = pick('PM10');
  const ica = pick('ICA');
  const t = translations[currentLang];

  const fmtVal = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? 'N/D' : Number(v).toFixed(1);
  const hasData = [no2, pm10, pm25].some(v => v !== null && v !== undefined);

  // Semáforo EEA único, con umbrales inyectados desde src/config.py.
  const riskLevelFor = (n, p10, p25, ica) => {
    const safeN = n ?? 0, safe10 = p10 ?? 0, safe25 = p25 ?? 0, safeIca = ica ?? 0;
    const th = RISK_THRESHOLDS;
    if (safeN >= th.NO2.bad || safe10 >= th.PM10.bad || safe25 >= th['PM2.5'].bad || safeIca >= th.ICA.bad) return 2;
    if (safeN >= th.NO2.moderate || safe10 >= th.PM10.moderate || safe25 >= th['PM2.5'].moderate || safeIca >= th.ICA.moderate) return 1;
    return 0;
  };
  const paintBadge = (badge, level, hasValues) => {
    if (!hasValues) {
      badge.innerText = t.v10NoData;
      badge.style.color = "var(--muted)";
      badge.style.borderColor = "var(--border)";
      badge.style.backgroundColor = "transparent";
      return;
    }
    if (level === 0) { badge.innerText = t.v10Good; badge.style.color = "var(--green)"; badge.style.borderColor = "var(--green)"; badge.style.backgroundColor = getCssVar('--green')+"1A"; }
    else if (level === 1) { badge.innerText = t.v10Mod; badge.style.color = "var(--yellow)"; badge.style.borderColor = "var(--yellow)"; badge.style.backgroundColor = getCssVar('--yellow')+"1A"; }
    else { badge.innerText = t.v10Bad; badge.style.color = "var(--red)"; badge.style.borderColor = "var(--red)"; badge.style.backgroundColor = getCssVar('--red')+"1A"; }
  };

  let badge = document.getElementById('riskBadge');
  paintBadge(badge, riskLevelFor(no2, pm10, pm25, ica), hasData);

  document.getElementById('val-no2').innerHTML = fmtVal(no2);
  document.getElementById('val-pm25').innerHTML = fmtVal(pm25);
  document.getElementById('val-pm10').innerHTML = fmtVal(pm10);
  document.getElementById('val-ica').innerHTML = fmtVal(ica);

  const d2Results = {};
  for (const contaminant of ['NO2', 'PM10', 'PM2.5', 'ICA']) {
    d2Results[contaminant] = targetsData[`${contaminant}_${currentZoneV10}_d2`] || null;
  }
  const hasD2 = Object.values(d2Results).every(Boolean);
  document.getElementById('d2PlanningBlock').style.display = hasD2 ? 'block' : 'none';
  const d2No2 = d2Results.NO2?.prediction ?? null;
  const d2Pm10 = d2Results.PM10?.prediction ?? null;
  const d2Pm25 = d2Results['PM2.5']?.prediction ?? null;
  const d2Ica = d2Results.ICA?.prediction ?? null;
  const d2Badge = document.getElementById('riskBadgeD2');
  const hasD2Values = hasD2 && [d2No2, d2Pm10, d2Pm25].some(v => v !== null && v !== undefined);
  paintBadge(d2Badge, riskLevelFor(d2No2, d2Pm10, d2Pm25, d2Ica), hasD2Values);

  const setD2Value = (id, rangeId, result) => {
    document.getElementById(id).textContent = result ? Number(result.prediction).toFixed(1) : '--';
    const hasInterval = result
      && result.lower !== null && result.lower !== undefined
      && result.upper !== null && result.upper !== undefined;
    document.getElementById(rangeId).textContent = hasInterval
      ? `IC90 ${Number(result.lower).toFixed(1)}–${Number(result.upper).toFixed(1)}`
      : 'IC90 N/D';
  };
  setD2Value('val-no2-d2', 'range-no2-d2', d2Results.NO2);
  setD2Value('val-pm25-d2', 'range-pm25-d2', d2Results['PM2.5']);
  setD2Value('val-pm10-d2', 'range-pm10-d2', d2Results.PM10);
  setD2Value('val-ica-d2', 'range-ica-d2', d2Results.ICA);

  const d = perfStats[currentZoneV10][currentContV10] || {labels:[], real:[], pred:[]};
  const ctx = document.getElementById('perfChart').getContext('2d');
  if(perfChart) perfChart.destroy();
  
  const historicalLabels = (perfStats[currentZoneV10].labels || []).map(
    l => l === 'Ayer' ? predictionDateOffsetLabel(-2) : l
  );
  const d1DateLabel = predDate.split('/').slice(0, 2).join('/');
  const d2DateLabel = predictionDateOffsetFull(1).split('/').slice(0, 2).join('/');
  const labels = [...historicalLabels, `${d1DateLabel} · D+1`, `${d2DateLabel} · D+2`];
  const d1Current = targetsData[`${currentContV10}_${currentZoneV10}_d1`]?.prediction ?? null;
  const d2Current = targetsData[`${currentContV10}_${currentZoneV10}_d2`]?.prediction ?? null;
  const historicalReal = [...d.real, null, null];
  const historicalPred = [...d.pred, null, null];
  const d1Forecast = Array(labels.length).fill(null);
  const d2Forecast = Array(labels.length).fill(null);
  if (d1Current !== null) {
    const lastHistoricalPrediction = d.pred[d.pred.length - 1];
    if (lastHistoricalPrediction !== null && lastHistoricalPrediction !== undefined) {
      d1Forecast[historicalLabels.length - 1] = lastHistoricalPrediction;
    }
    d1Forecast[historicalLabels.length] = d1Current;
  }
  if (d1Current !== null && d2Current !== null) {
    d2Forecast[historicalLabels.length] = d1Current;
    d2Forecast[historicalLabels.length + 1] = d2Current;
  }
  const text = getCssVar('--muted'); const grid = getCssVar('--border');

  perfChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: labels,
          datasets: [
              { label: t.backReal, data: historicalReal, borderColor: getCssVar('--observed'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.4 },
              { label: t.backPred, data: historicalPred, borderColor: getCssVar('--accent'), backgroundColor: 'transparent', borderDash: [5,5], borderWidth: 2, pointRadius: 4, tension: 0.4 },
              { label: t.forecastD1, data: d1Forecast, borderColor: getCssVar('--green'), backgroundColor: getCssVar('--green'), borderDash: [7,5], borderWidth: 3, pointRadius: ctx => ctx.dataIndex === historicalLabels.length - 1 ? 0 : 7, pointStyle: 'circle', tension: 0.2 },
              { label: t.forecastD2, data: d2Forecast, borderColor: getCssVar('--turquoise'), backgroundColor: getCssVar('--turquoise'), borderWidth: 3, pointRadius: ctx => ctx.dataIndex === historicalLabels.length ? 0 : 8, pointStyle: 'triangle', tension: 0.2 }
          ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { color: text, font: { family: 'IBM Plex Mono'} } }, }, scales: { x: { ticks: { color: text }, grid: { color: grid } }, y: { ticks: { color: text }, grid: { color: grid } } } }
  });

  const lastRealDraw = d.real[d.real.length - 1];
  const lastPredDraw = d.pred[d.pred.length - 1];
  const lastPred = (lastPredDraw === null || lastPredDraw === undefined) ? null : Number(lastPredDraw);
  const unit = currentContV10 === 'ICA' ? '' : ' µg/m³';
  const pendingText = currentLang === 'es' ? "N/D (Pendiente)" : "E/D (Zain)";
  let realText = pendingText;
  let predText = 'N/D';
  let interpret = "⏳";

  if (lastRealDraw !== null && lastRealDraw !== undefined) {
      const lastReal = Number(lastRealDraw);
      realText = `${lastReal.toFixed(1)}${unit}`;
      if (lastPred !== null && lastPred > 0) {
          predText = `${lastPred.toFixed(1)}${unit}`;
          const error = ((lastReal - lastPred) / lastPred) * 100;
          const absError = Math.abs(error);
          const absValDiff = Math.abs(lastReal - lastPred);

          let isLowAbsError = false;
          if (currentContV10 === 'ICA' && absValDiff <= 5.0) { isLowAbsError = true; }
          else if (currentContV10 === 'NO2' && absValDiff <= 4.0) { isLowAbsError = true; }
          else if (currentContV10 === 'PM10' && absValDiff <= 6.0) { isLowAbsError = true; }
          else if (currentContV10 === 'PM2.5' && absValDiff <= 4.0) { isLowAbsError = true; }

          if (absError <= 35) {
              interpret = "🟢"; // Precisión aceptable/buena/excelente
          } else if (isLowAbsError) {
              interpret = "🟡"; // Tolerancia extra en concentraciones bajas
          } else {
              interpret = "🔴"; // Desviación alta
          }
      }
  }

  // Se escribe siempre: antes, cuando faltaba la medición real, la tabla
  // quedaba con el contenido de la selección anterior.
  document.getElementById('backtestTable').innerHTML = `
    <tr>
      <td><strong>${currentContV10} (${currentZoneV10.toUpperCase()})</strong></td>
      <td>${predText}</td>
      <td>${realText}</td>
      <td style="font-size: 1.25rem; text-align: center; line-height: 1;">${interpret}</td>
    </tr>
  `;
}

function selectContV10(cont, btn) { currentContV10 = cont; document.querySelectorAll('#contTabsV10 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderDashboard3(); }
function selectZoneV10(zone, btn) { currentZoneV10 = zone; document.querySelectorAll('#zoneTabsV10 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderDashboard3(); }

function renderMetricsTable() {
  const order = [ ['NO2','zbe'], ['NO2','out'], ['PM10','zbe'], ['PM10','out'], ['PM2.5','zbe'], ['PM2.5','out'], ['ICA','zbe'], ['ICA','out'], ];
  const body = document.getElementById('metricsBody');
  if (!body) return;
  const t = translations[currentLang];
  // Actualizar cabecera de tabla de métricas si es necesario
  const thead = body.closest('table').querySelector('thead');
  if (thead) {
      const rows = thead.querySelectorAll('th');
      if (rows.length >= 5) {
          rows[1].innerText = t.colRMSE || 'RMSE';
          rows[2].innerText = t.colMAE || 'MAE';
          rows[4].innerText = t.colMAPE || 'MAPE %';
      }
  }

  body.innerHTML = order.map(([cont, zone]) => {
    const key = `target_${cont}_${zone}_d1`;
    const m = metricsData[key];
    if (!m) return '';
    const mapeColor = m.cv_mape <= 25 ? 'var(--green)' : 'var(--yellow)';
    const r2Color   = m.cv_r2   >= 0.35 ? 'var(--green)' : 'var(--yellow)';
    
    // Buscar mejora del meta-modelo
    const metaKey = `${cont}_${zone}_d1`;
    const meta = metaData[metaKey];
    let metaStr = '';
    if (meta && Number.isFinite(Number(meta.improvement_pct))) {
      const imp = Number(meta.improvement_pct);
      const sign = imp > 0 ? '+' : '';
      metaStr = `<div style="font-size:9px; color:var(--accent); margin-top:2px">${sign}${imp.toFixed(2)}% ${t.metaImprovement}</div>`;
    }

    return `<tr>
      <td><strong>${cont}</strong> <span style="color:var(--muted)">${zone.toUpperCase()}</span>${metaStr}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${num(m.cv_rmse, 2)}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${num(m.cv_mae, 2)}</td>
      <td style="color:${r2Color};font-family:'IBM Plex Mono',monospace">${num(m.cv_r2, 3)}</td>
      <td style="color:${mapeColor};font-family:'IBM Plex Mono',monospace;font-weight:bold">${num(m.cv_mape, 1)}%</td>
      <td style="color:var(--muted);font-family:'IBM Plex Mono',monospace">${num(m.n_features, 0)}</td>
    </tr>`;
  }).join('');
  renderMetricsTableD2();
}

function renderMetricsTableD2() {
  const body = document.getElementById('metricsBodyD2');
  if (!body) return;
  const order = [ ['NO2','zbe'], ['NO2','out'], ['PM10','zbe'], ['PM10','out'], ['PM2.5','zbe'], ['PM2.5','out'] ];
  const t = translations[currentLang];
  const headings = body.closest('table').querySelectorAll('thead th');
  if (headings.length >= 5) {
    headings[1].innerText = t.colRMSE || 'RMSE';
    headings[2].innerText = t.colMAE || 'MAE';
    headings[4].innerText = t.colMAPE || 'MAPE %';
  }
  body.innerHTML = order.map(([cont, zone]) => {
    const m = metricsData[`target_${cont}_${zone}_d2`];
    if (!m) return '';
    const mapeColor = m.cv_mape <= 25 ? 'var(--green)' : 'var(--yellow)';
    const r2Color = m.cv_r2 >= 0.35 ? 'var(--green)' : 'var(--yellow)';
    return `<tr>
      <td><strong>${cont}</strong> <span style="color:var(--muted)">${zone.toUpperCase()}</span></td>
      <td style="font-family:'IBM Plex Mono',monospace">${num(m.cv_rmse, 2)}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${num(m.cv_mae, 2)}</td>
      <td style="color:${r2Color};font-family:'IBM Plex Mono',monospace">${num(m.cv_r2, 3)}</td>
      <td style="color:${mapeColor};font-family:'IBM Plex Mono',monospace;font-weight:bold">${num(m.cv_mape, 1)}%</td>
      <td style="color:var(--muted);font-family:'IBM Plex Mono',monospace">${num(m.n_features, 0)}</td>
    </tr>`;
  }).join('');
}

// ── MAPA DE ESTACIONES (Leaflet) ──────────────────────────────────────────────
function icaColor(ica) {
  if (ica === null || ica === undefined) return getCssVar('--muted');
  if (ica <= 25)  return getCssVar('--green');
  if (ica <= 50)  return getCssVar('--yellow');
  if (ica <= 75)  return '#ff7043';
  return getCssVar('--red');
}
function icaEmoji(ica) {
  if (ica === null || ica === undefined) return '❓';
  if (ica <= 25)  return '🟢';
  if (ica <= 50)  return '🟡';
  if (ica <= 75)  return '🟠';
  return '🔴';
}

function initMap() {
  if (_mapInitialized) { return; }
  _mapInitialized = true;
  mapInstance = L.map('stationMap', { zoomControl: true });
  
  mapTileLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }).addTo(mapInstance);

  const ZBE_COORDS = [ [42.845707, -2.6764994], [42.846058, -2.678793], [42.846058, -2.678777], [42.84426, -2.6781747], [42.84412, -2.6780825], [42.84361, -2.6756046], [42.843502, -2.6743975], [42.843246, -2.6726305], [42.84232, -2.6727068], [42.8423, -2.6724603], [42.84322, -2.6724162], [42.84342, -2.6700573], [42.8442, -2.6686096], [42.845608, -2.6682484], [42.846146, -2.6681008], [42.846935, -2.667851], [42.84823, -2.668305], [42.849224, -2.6686778], [42.850346, -2.6691453], [42.852093, -2.670227], [42.85308, -2.6731813], [42.85263, -2.6730998], [42.852352, -2.6731632], [42.852013, -2.6732967], [42.851536, -2.6737828], [42.849266, -2.6755311], [42.847466, -2.6762118], [42.84709, -2.676142], [42.845707, -2.6764994] ];
  
  L.polygon(ZBE_COORDS, {
    color: getCssVar('--red'), weight: 2.5, opacity: 0.9, fillColor: getCssVar('--red'), fillOpacity: 0.07, interactive: false, dashArray: '6 4',
  }).addTo(mapInstance);

  Object.entries(stationsData).forEach(([name, s]) => {
    const color = icaColor(s.ICA);
    const isZbe = s.zone === 'ZBE';
    const markerColor = isZbe ? getCssVar('--accent') : getCssVar('--observed');

    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 11, color: markerColor, weight: 2, fillColor: color, fillOpacity: 0.95,
    }).addTo(mapInstance);

    const t = translations[currentLang];
    const popupHtml = `
      <div class="popup-name">${icaEmoji(s.ICA)} ${s.label}</div>
      <div style="font-size:10px;color:var(--accent);font-family:'IBM Plex Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em">${t.mapDailyAyer}</div>
      <div class="popup-row"><span class="popup-label">NO₂</span><span class="popup-val">${fmt(s.NO2)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">PM10</span><span class="popup-val">${fmt(s.PM10)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">PM2.5</span><span class="popup-val">${fmt(s.PM25)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">ICA</span><span class="popup-val">${fmt(s.ICA)}</span></div>
      <div class="popup-section">
        <div class="popup-section-title">${t.mapPrediction} · ${predictionDayLabel()} ${predDate}</div>
        <div class="popup-row"><span class="popup-label">NO₂</span><span class="popup-val">${fmt(s.pred_NO2)} µg/m³</span></div>
        <div class="popup-row"><span class="popup-label">PM10</span><span class="popup-val">${fmt(s.pred_PM10)} µg/m³</span></div>
        <div class="popup-row"><span class="popup-label">PM2.5</span><span class="popup-val">${fmt(s.pred_PM25)} µg/m³</span></div>
        <div style="font-size:10px;color:var(--muted);margin-top:6px">${t.mapZone} ${s.zone} </div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">${t.mapNote}</div>
      </div>`;
    marker.bindPopup(popupHtml, { maxWidth: 260 });

    const icon = L.divIcon({
      className: '',
      html: `<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text);background:var(--surface);padding:2px 5px;border-radius:3px;white-space:nowrap;border:1px solid var(--border)">${name}</div>`,
      iconAnchor: [-14, 6]
    });
    L.marker([s.lat, s.lon], { icon, interactive: false }).addTo(mapInstance);
  });

  const allLatLngs = Object.values(stationsData).map(s => [s.lat, s.lon]);
  if (allLatLngs.length > 0) mapInstance.fitBounds(L.latLngBounds(allLatLngs), { padding: [50, 50] });

  setTimeout(() => mapInstance.invalidateSize(), 100);
}

// ── FORESIGHT AI (SHAP & LLM) ──────────────────────────────────────────────
const featureNames = {
  es: {
    'traffic_occupancy': 'Ocupación Tráfico',
    'traffic_volume': 'Volumen Tráfico',
    'fc_temperature_2m': 'Temperatura',
    'fc_relative_humidity_2m': 'Humedad Relativa',
    'fc_precipitation': 'Precipitación',
    'fc_wind_speed_10m': 'Velocidad Viento',
    'fc_wind_gusts_10m': 'Ráfagas Viento',
    'fc_cloud_cover': 'Nubosidad',
    'fc_surface_pressure': 'Presión Atmosférica',
    'fc_sunshine_duration': 'Horas de Sol',
    'fc_boundary_layer_height': 'Altura Capa Límite',
    'lag_': ' (hace ',
    'd': ' días)',
    'roll_mean_': 'Media ',
    'roll_std_': 'Volatilidad ',
    'NO2': 'Dióxido Nitrógeno',
    'PM10': 'Partículas PM10',
    'PM2.5': 'Partículas PM2.5',
    'ICA': 'Índice Calidad Aire',
    'zbe': ' (ZBE)',
    'out': ' (Exterior)',
    'temp_min_dia': 'Temp. Mínima',
    'temp_max_dia': 'Temp. Máxima',
    'exp_traffic_volume': 'Tráfico esperado'
  },
  eu: {
    'traffic_occupancy': 'Trafiko Okupazioa',
    'traffic_volume': 'Trafiko Bolumena',
    'fc_temperature_2m': 'Tenperatura',
    'fc_relative_humidity_2m': 'Hezetasun Erlatiboa',
    'fc_precipitation': 'Prezipitazioa',
    'fc_wind_speed_10m': 'Haizearen Abiadura',
    'fc_wind_gusts_10m': 'Haize Boladak',
    'fc_cloud_cover': 'Hodeitza',
    'fc_surface_pressure': 'Presio Atmosferikoa',
    'fc_sunshine_duration': 'Eguzki Orduak',
    'fc_boundary_layer_height': 'Muga Geruzaren Altuera',
    'lag_': ' (duela ',
    'd': ' egun)',
    'roll_mean_': 'Batez besteko ',
    'roll_std_': 'Hegakorritasun ',
    'NO2': 'Nitrogeno Dioxidoa',
    'PM10': 'PM10 Partikulak',
    'PM2.5': 'PM2.5 Partikulak',
    'ICA': 'Aire Kalitatearen Indizea',
    'zbe': ' (EGE)',
    'out': ' (Kanpokoa)',
    'temp_min_dia': 'Gutxieneko Tenperatura',
    'temp_max_dia': 'Gehieneko Tenperatura',
    'exp_traffic_volume': 'Aurreikusitako trafikoa'
  }
};

function getFeatureLabel(feat) {
  const lang = currentLang || 'es';
  const dict = featureNames[lang];
  let label = feat;

  // 1. Limpieza inicial de prefijos comunes
  label = label.replace(/^fc_/, '').replace(/^exp_/, '');

  // 2. Traducción de términos base
  for (const [key, val] of Object.entries(dict)) {
    // Evitar pisar términos compuestos (ej: wind_speed vs wind)
    if (key.length > 3 && label.includes(key)) {
      label = label.replace(key, val);
    }
  }

  // 3. Formateo de Lags (hace X días)
  const lagMatch = label.match(/lag (\d+)d/i) || label.match(/_lag_(\d+)d/i);
  if (lagMatch) {
    const days = lagMatch[1];
    const suffix = (days === '1' && lang === 'es') ? ' (ayer)' : 
                   (days === '1' && lang === 'eu') ? ' (atzo)' :
                   dict['lag_'] + days + dict['d'];
    label = label.replace(lagMatch[0], suffix);
  }

  // 4. Formateo de Medias Móviles
  const rollMatch = label.match(/roll_mean_(\d+)d/i) || label.match(/media (\d+)d/i);
  if (rollMatch) {
    label = label.replace(rollMatch[0], dict['roll_mean_'] + rollMatch[1] + 'd');
  }

  // 5. Limpieza final
  label = label.replace(/_/g, ' ').trim();
  
  return label.charAt(0).toUpperCase() + label.slice(1);
}

let currentContFs = 'NO2', currentZoneFs = 'zbe', fsChart = null;

function renderForesight() {
  const key = `${currentContFs}_${currentZoneFs}_d1`;
  const tData = targetsData[key] || {};
  const fs = tData.foresight || null;
  const t = translations[currentLang];
  
  const narrativeDiv = document.getElementById('fs-narrative');
  if (!fs) {
    narrativeDiv.innerHTML = `<span style='color:var(--muted)'>${t.fsNoData}</span>`;
    if (fsChart) { fsChart.destroy(); fsChart = null; }
    return;
  }
  
  let narrativeText = fs.narrative || t.fsNoNarrative;
  if (narrativeText && typeof narrativeText === 'object') {
    narrativeText = narrativeText[currentLang] || narrativeText['es'] || t.fsNoNarrative;
  }
  if (!narrativeText && fs.error) {
    narrativeText = `Error: ${fs.error}`;
  }
  narrativeText = adaptNarrativeDate(narrativeText);

  // Construir vista estructurada: chips de features + narrativa LLM
  const posTop = fs.positive_top || [];
  const negTop = fs.negative_top || [];
  let allFactors = [...posTop.map(f => ({...f, dir: 'pos'})), ...negTop.map(f => ({...f, dir: 'neg'}))];
  allFactors.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const topFactors = allFactors.slice(0, 6);

  const chipHtml = topFactors.map(f => {
    const label = getFeatureLabel(f.feature);
    const isPos = f.dir === 'pos';
    const sign = isPos ? '▲' : '▼';
    const chipColor = isPos
      ? 'background:rgba(220,38,38,0.12);border:1px solid rgba(220,38,38,0.3);color:var(--red)'
      : 'background:rgba(5,150,105,0.12);border:1px solid rgba(5,150,105,0.3);color:var(--green)';
    const valStr = isPos ? `+${Math.abs(f.value).toFixed(2)}` : `-${Math.abs(f.value).toFixed(2)}`;
    return `<div style="display:inline-flex;align-items:center;gap:6px;${chipColor};padding:5px 10px;border-radius:20px;font-size:12px;font-family:'IBM Plex Mono',monospace;white-space:nowrap">
      <span>${sign}</span>
      <span style="font-weight:600">${label}</span>
      <span style="opacity:0.7">(${valStr})</span>
    </div>`;
  }).join('');

  const featureSectionTitle = currentLang === 'eu' ? 'Eragin nagusiak' : 'Variables más relevantes';
  const narrativeSectionTitle = currentLang === 'eu' ? 'Analisia' : 'Análisis';

  narrativeDiv.innerHTML = `
    <div style="margin-bottom:16px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent);margin-bottom:10px">${featureSectionTitle}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">${chipHtml || '<span style="color:var(--muted);font-size:12px">Sin datos de features</span>'}</div>
    </div>
    <div style="border-top:1px solid var(--border);padding-top:14px;margin-top:4px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:8px">${narrativeSectionTitle}</div>
      <div style="font-size:14px;line-height:1.7;color:var(--text);white-space:pre-line">${narrativeText || ''}</div>
    </div>
  `;
  
  // Datos para el gráfico de barras (reusa allFactors ya calculado arriba)
  let factors = allFactors.slice(0, 8); // Top 8 features
  
  const ctx = document.getElementById('fsChart').getContext('2d');
  if (fsChart) fsChart.destroy();
  
  const labels = factors.map(f => {
    const friendly = getFeatureLabel(f.feature);
    return friendly.length > 35 ? friendly.slice(0, 35) + '...' : friendly;
  });
  const values = factors.map(f => f.value);
  const colors = factors.map(f => f.value > 0 ? getCssVar('--red') : getCssVar('--green'));
  
  const text = getCssVar('--text'); const muted = getCssVar('--muted'); const grid = getCssVar('--border');
  
  fsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: t.fsImpactLabel,
        data: values,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { ticks: { color: muted }, grid: { color: grid } },
        y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 } }, grid: { display: false } }
      }
    }
  });
}

function selectContFs(cont, btn) { currentContFs = cont; document.querySelectorAll('#contTabsForesight .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderForesight(); }
function selectZoneFs(zone, btn) { currentZoneFs = zone; document.querySelectorAll('#zoneTabsForesight .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderForesight(); }

// End Foresight Block

// ── TRAFFIC YOY CHART ──────────────────────────────────────────────────────
let trafficYoYChartInstance = null;
function renderTrafficYoY() {
  const ctx = document.getElementById('trafficYoYChart');
  if (!ctx || trafficYoyData.length === 0) return;

  const labels = trafficYoyData.map(d => {
    const date = parseDateUTC(d.date);
    return date.toLocaleDateString(currentLang === 'es' ? 'es-ES' : 'eu-ES', { day: 'numeric', month: 'short', timeZone: 'UTC' });
  });
  
  const currentSeries = trafficYoyData.map(d => d.current);
  const previousSeries = trafficYoyData.map(d => d.previous);

  // Etiqueta honesta según la fase real del año anterior comparado.
  const phases = new Set(trafficYoyData.map(d => d.previous_phase).filter(Boolean));
  const tYoy = translations[currentLang];
  let previousLabel = tYoy.trafficPrevious;
  if (phases.size === 1) {
    previousLabel = phases.has('pre_zbe') ? tYoy.trafficPreviousPre : tYoy.trafficPreviousYear1;
  } else if (phases.size > 1) {
    previousLabel = tYoy.trafficPreviousMixed;
  }

  // Aplicar media móvil para suavizar (7 días)
  const currentSmooth = movingAvg(currentSeries, 7);
  const previousSmooth = movingAvg(previousSeries, 7);

  if (trafficYoYChartInstance) trafficYoYChartInstance.destroy();

  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  
  trafficYoYChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: translations[currentLang].trafficCurrent,
          data: currentSmooth,
          borderColor: getCssVar('--observed'),
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: previousLabel,
          data: previousSmooth,
          borderColor: getCssVar('--muted'),
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0.3,
          alpha: 0.5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { color: getCssVar('--text'), font: { size: 11 } } },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.parsed.y.toLocaleString()} veh/día`;
            }
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: getCssVar('--muted'), font: { size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
        y: { 
          title: { display: true, text: translations[currentLang].trafficVolume, color: getCssVar('--muted'), font: { size: 10 } },
          grid: { color: getCssVar('--border'), drawBorder: false }, 
          ticks: { color: getCssVar('--muted'), font: { size: 10 } } 
        }
      }
    }
  });
}

// End Traffic YoY Block

// ── INIT ───────────────────────────────────────────────────────────────────
window.onload = function() {
  updateI18n();
  updateRiskDate();
  renderPipelineBanner();
  
  if (localStorage.getItem('vitoria_theme') === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    currentTheme = 'dark';
    updateI18n(); // Re-actualizar textos de botones que dependen del tema
  }

  renderSummaryCards();
  renderFig1();
  renderFig2();
  renderFig3();
  renderDidTable();
  renderDashboard3();
  renderMetricsTable();
  renderV9Cards();
  renderForesight();
  renderTrafficYoY();
  updateV9Images();
  
  // Si estamos en la pestaña de tráfico, sincronizar lenguaje
  const iframe = document.getElementById('trafficIframe');
  if (iframe) {
      iframe.src = `traffic_map.html?lang=${currentLang}`;
  }
  console.log("DASHBOARD INITIALIZED SUCCESSFULLY");
};
</script>
<footer class="footer">
  <span id="pipelineStatus">__PIPELINE_STATUS__</span>
</footer>
</body>
</html>"""

# Sustitución e Inyección Final
content = html_template.replace('__CF_DATA_PLACEHOLDER__', cf_json_str)
content = content.replace('__PERF_DATA_PLACEHOLDER__', perf_json_str)
content = content.replace('__MANANA_DATA_PLACEHOLDER__', manana_json_str)
content = content.replace('__METRICS_DATA_PLACEHOLDER__', metrics_json_str)
content = content.replace('__STATIONS_DATA_PLACEHOLDER__', stations_json_str)
content = content.replace('__TARGETS_DATA_PLACEHOLDER__', targets_json_str)
content = content.replace('__V9_DATA_PLACEHOLDER__', v9_json_str)
content = content.replace('__META_DATA_PLACEHOLDER__', meta_json_str)
content = content.replace('__SUMMARY_DATA_PLACEHOLDER__', sum_json_str)
content = content.replace('__DID_DATA_PLACEHOLDER__', did_json_str)
content = content.replace('__TRAFFIC_YOY_DATA_PLACEHOLDER__', traffic_yoy_json_str)
content = content.replace('__EVENT_STUDY_TREND_PLACEHOLDER__', event_study_trend_json_str)
content = content.replace('__RISK_THRESHOLDS_PLACEHOLDER__', json.dumps(RISK_THRESHOLDS))
content = content.replace('__IMG_BASE_PATH__', img_base_path)
content = content.replace('__PRED_DATE_PLACEHOLDER__', prediction_date_str)

# Pie con la frescura real del pipeline (se escribe en run_pipeline.py).
pipeline_status_text = "Estado del pipeline: no disponible"
status = pipeline_status_data
if status:
    pipeline_status_text = (
        f"Datos hasta {status.get('source_date', 'N/D')} · "
        f"Predicción {status.get('prediction_date', 'N/D')} · "
        f"Generado {status.get('generated_at', 'N/D')}"
    )
content = content.replace('__PIPELINE_STATUS__', pipeline_status_text)
content = content.replace('__STATION_QC_PLACEHOLDER__', json.dumps(station_qc))
content = content.replace('__CAUSAL_VALIDATION_PLACEHOLDER__', json.dumps(causal_validation))
content = content.replace('__DRIFT_REPORT_PLACEHOLDER__', json.dumps(drift_report))
content = content.replace('__PIPELINE_STATUS_DATA_PLACEHOLDER__', json.dumps(pipeline_status_data))

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Listo! Archivo {output_html} generado correctamente.")
