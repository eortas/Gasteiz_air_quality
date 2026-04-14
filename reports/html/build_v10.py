import sys
import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import shutil

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"

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
try:
    if csv_file.exists():
        df_cv = pd.read_csv(csv_file)
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
                    "pure": round(sub_pure['gap_pct'].mean(), 1),
                    "unit": "µg/m³" if cont != "ICA" else "",
                    "lags": round(sub_lags['gap_pct'].mean(), 1) if not sub_lags.empty else 0
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

# ==============================================================================
# 4. PREDICCIONES DE MAÑANA
# ==============================================================================
print("Leyendo predicciones de manana desde predictions_latest.json...")
pred_json_path = PROCESSED_DIR / "predictions_latest.json"
manana_data = {'zbe': {}, 'out': {}}
prediction_date_str = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")

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
            if key in targets:
                # Prioridad a 'refined' (v2), si no 'prediction' (v1)
                val_v2 = targets[key].get("refined")
                val_v1 = targets[key].get("prediction", 0)
                manana_data[zone][cont] = val_v2 if val_v2 is not None else val_v1
            else:
                manana_data[zone][cont] = 0

    targets_json_str = json.dumps(targets)

    print(f"  OK Predicciones para {prediction_date_str} cargadas (usando refinamiento v2 si disponible).")

except Exception as e:
    print(f"  WARN No se pudo leer predictions_latest.json: {e}")
    targets_json_str = "{}"
    for z in ['zbe', 'out']:
        for c in ['NO2', 'PM10', 'PM2.5', 'ICA']:
            manana_data[z][c] = 0

# ==============================================================================
# 4. BACKTESTING
# ==============================================================================
print("Calculando backtesting con modelos v8...")
perf_data = {'zbe': {}, 'out': {}}

try:
    df_feat = pd.read_parquet(DATASET_PATH)
    df_feat["date"] = pd.to_datetime(df_feat["date"], utc=True)

    pred_full_path = PROCESSED_DIR / "features_latest.parquet"
    if pred_full_path.exists():
        df_latest = pd.read_parquet(pred_full_path)
        df_latest["date"] = pd.to_datetime(df_latest["date"], utc=True)
        if not df_feat.empty and not df_latest.empty:
            if df_latest["date"].iloc[-1] >= df_feat["date"].iloc[-1]:
                df_feat = df_latest

    if not df_feat.empty:
        df_feat = df_feat.sort_values("date").reset_index(drop=True)
        last_rows = df_feat.tail(9).copy()
        inputs_backtest  = last_rows.iloc[0:7]
        targets_backtest = last_rows.iloc[1:8]
        
        fechas = targets_backtest['date'].dt.strftime('%d %b').tolist()
        if len(fechas) > 0:
            fechas[-1] = "Ayer"
        
        for zone in ['zbe', 'out']:
            perf_data[zone]["labels"] = fechas
            for cont in ['NO2', 'PM10', 'PM2.5']:
                target_name = f"{cont}_{zone}_d1"
                model_path = MODELS_DIR / f"lgbm_v8_{target_name}.pkl"
                feat_path  = MODELS_DIR / f"lgbm_v8_{target_name}_features.json"
                
                try:
                    model = joblib.load(model_path)
                    features = json.loads(feat_path.read_text(encoding="utf-8"))
                    
                    X_backtest = inputs_backtest.reindex(columns=features, fill_value=0).fillna(0)
                    preds_7d = model.predict(X_backtest)
                    
                    contam_col = f"{cont}_{zone}"
                    if contam_col in targets_backtest.columns:
                        real_vals = [None if pd.isna(v) else round(v, 1) for v in targets_backtest[contam_col]]
                    else:
                        real_vals = [None] * 7
                        
                    while len(real_vals) < 7:
                        real_vals.insert(0, None)
                    
                    perf_data[zone][cont] = {
                        "real": real_vals,
                        "pred": [round(max(0, p), 1) for p in preds_7d]
                    }
                    
                except Exception as e:
                    perf_data[zone][cont] = {"real": [None]*7, "pred": [0]*7}

        print("  OK Predicciones y Backtest listos.")
    else:
        raise ValueError("El DataFrame de features está vacío.")

except Exception as e:
    print(f"  WARN No se encontro parquet o error general en backtesting: {e}")
    for z in ['zbe', 'out']:
        perf_data[z] = {"labels": ["1", "2", "3", "4", "5", "6", "Ayer"]}
        for c in ['NO2', 'PM10', 'PM2.5']:
            perf_data[z][c] = {"real": [0]*7, "pred": [0]*7}
            if z in manana_data and c in manana_data[z]:
                manana_data[z][c] = 0

perf_json_str = json.dumps(perf_data)
manana_json_str = json.dumps(manana_data)

try:
    metrics_raw = json.loads((MODELS_DIR / "metrics_v8.json").read_text(encoding="utf-8"))
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
        
        for stn, meta in STATION_COORDS.items():
            pred_src = pred_zbe if meta["zone"] == "ZBE" else pred_out
            stations_data[stn] = {
                **meta,
                "NO2":  _get_stn_val(latest, stn, "NO2"),
                "PM10": _get_stn_val(latest, stn, "PM10"),
                "PM25": _get_stn_val(latest, stn, "PM25"),
                "ICA":  _get_stn_val(latest, stn, "ICA"),
                "pred_NO2":  round(pred_src.get("NO2", 0), 1),
                "pred_PM10": round(pred_src.get("PM10", 0), 1),
                "pred_PM25": round(pred_src.get("PM2.5", 0), 1),
            }
    else:
        raise ValueError("station_daily.csv está vacío.")
        
except Exception as e:
    print(f"  WARN Error cargando station_daily para el mapa: {e}")
    for stn, meta in STATION_COORDS.items():
        stations_data[stn] = {**meta, "NO2": None, "PM10": None, "PM25": None, "ICA": None,
                              "pred_NO2": 0, "pred_PM10": 0, "pred_PM25": 0}

stations_json_str = json.dumps(stations_data)

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
    --red:       #ef5350;
    --yellow:    #ffd54f;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body { background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; min-height: 100vh; }

  /* Contenedor fijo para header y pestañas */
  .header-wrapper { position: sticky; top: 0; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
  
  /* Barra superior (Azul) + Bandas (Blanca y Roja francesa) */
  .main-header { position: relative; display: flex; justify-content: space-between; align-items: center; padding: 16px 48px; background: #0055A4; color: #fff; border: none; }
  .main-header::after { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 6px; background: linear-gradient(to bottom, #ffffff 50%, #EF4135 50%); }

  .main-header-title { font-size: 18px; font-weight: 700; }
  .header-right { display: flex; align-items: center; gap: 24px; }
  .main-header-author { font-size: 13px; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9; }
  
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
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
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
  .v10-risk-badge { display: inline-block; font-size: 32px; font-weight: 800; padding: 12px 30px; border-radius: 8px; margin-bottom: 30px; font-family: 'IBM Plex Mono', monospace; border: 2px solid; }
  .v10-risk-grid { display: flex; justify-content: center; gap: 70px; }
  .v10-risk-item .val { font-size: 28px; font-weight: 700; color: var(--text); font-family: 'IBM Plex Mono', monospace;}
  .v10-risk-item .lab { font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 8px; font-family: 'IBM Plex Mono', monospace;}
  
  .perf-chart-wrap, .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; height: 350px; position: relative; }
  .perf-chart-wrap canvas, .chart-wrap canvas { display: block; width: 100% !important; height: 100% !important; }

  @media (max-width: 900px) {
    .header, .controls, .charts-section, .did-section, .top-nav { padding-left: 20px; padding-right: 20px; }
    .summary-grid { grid-template-columns: 1fr 1fr; }
    h1 { font-size: 20px; }
    .v10-risk-grid { flex-direction: column; gap: 20px; }
    .main-header { padding-left: 20px; padding-right: 20px; flex-direction: column; align-items: flex-start; gap: 12px; }
    .header-right { width: 100%; justify-content: space-between; }
  }

  @media screen and (max-width: 768px) {
      .chart-wrap, div[style*="position: relative"], canvas { min-height: 350px !important; width: 100% !important; }
  }

  #stationMap { height: 540px; border-radius: 4px; border: 1px solid var(--border); }
  .map-header { padding: 32px 48px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
  .map-legend { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; min-width: 170px; }
  .map-legend-item { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); }
  .map-legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .map-wrap { padding: 24px 48px 40px; }
  
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
      </div>
    </div>
  </div>

  <div class="top-nav">
    <button class="top-nav-btn active" onclick="switchMainView('v10', this)" data-i18n="nav1">1. Predicción Operativa</button>
    <button class="top-nav-btn" onclick="switchMainView('v8', this)" data-i18n="nav2">2. Monitor Interactivo Diario</button>
    <button class="top-nav-btn" onclick="switchMainView('v9', this)" data-i18n="nav3">3. Validación Causal</button>
    <button class="top-nav-btn" onclick="switchMainView('map', this)" data-i18n="nav4">4. Mapa de Estaciones</button>
    <button class="top-nav-btn" onclick="switchMainView('traffic', this)" data-i18n="nav5">5. Mapa de Tráfico</button>
    <button class="top-nav-btn" onclick="switchMainView('foresight', this)">6. Foresight AI</button>
  </div>
</div>

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
    <table><thead><tr><th data-i18n="didCont">Contaminante</th><th>β₃</th><th data-i18n="didPVal">p-valor</th><th>IC 95%</th><th data-i18n="didRel">Efecto relativo</th><th>R²</th><th>n obs</th><th data-i18n="didSig">Significativo</th></tr></thead><tbody id="didTable"></tbody></table>
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
    </div>
  </div>

  <div class="controls">
    <span class="controls-label" data-i18n="contaminant">Contaminante</span>
    <div class="tab-group" id="contTabsV10">
      <button class="tab active" onclick="selectContV10('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContV10('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContV10('PM2.5', this)">PM2.5</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label" data-i18n="zone">ZONA </span>
    <div class="tab-group" id="zoneTabsV10">
      <button class="tab active" onclick="selectZoneV10('out', this)" data-i18n="zoneOut">FUERA DE ZBE</button>
      <button class="tab" onclick="selectZoneV10('zbe', this)" data-i18n="zoneIn">ZBE</button>
    </div>
  </div>

  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header">
        <div class="fig-title" data-i18n="auditTitle"><strong>Auditoría del Modelo:</strong> Predicción vs Real (Últimos 7 días)</div>
      </div>
      <div class="perf-chart-wrap">
        <canvas id="perfChart"></canvas>
      </div>
    </div>
    
    <div class="did-section" style="padding: 0;">
      <div class="did-title" data-i18n="backtestTitle"><strong>Validación de Ayer</strong> — Backtesting de precisión</div>
      <table>
        <thead><tr><th data-i18n="colParam">Parámetro</th><th data-i18n="colPred">Predicción Ayer (v8)</th><th data-i18n="colReal">Medición Real</th><th data-i18n="colDev">Desviación</th><th data-i18n="colStatus">Estado</th></tr></thead>
        <tbody id="backtestTable"></tbody>
      </table>
    </div>
  </div>

  <div class="did-section" style="padding: 0 0 40px;">
    <div class="did-title" style="padding: 0 48px 16px;" data-i18n="perfTitle"><strong>Rendimiento del Modelo v8</strong> — Métricas Cross-Validation (5 Folds)</div>
    <div style="padding: 0 48px; overflow-x: auto;">
      <table id="metricsTable">
        <thead><tr><th data-i18n="colContZone">Contaminante + Zona</th><th data-i18n="colRMSE">RMSE (µg/m³)</th><th data-i18n="colMAE">MAE (µg/m³)</th><th data-i18n="colR2">R²</th><th data-i18n="colMAPE">MAPE %</th><th data-i18n="colNFeatures">N. Features</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
    <p style="padding: 12px 48px 0; font-size:11px; color:var(--muted); font-family:'IBM Plex Mono',monospace;">
      <span data-i18n="mapeNote">MAPE calculado solo sobre filas con valor observado > 1 µg/m³.</span>&nbsp;&nbsp;<span data-i18n="mapeThreshold">Umbral aceptación: MAPE < 25%, R² > 0.35.</span>
    </p>
  </div>
</div>

<div id="view-map" class="view-container">
  <div class="map-header" style="padding: 16px 48px 16px;">
    <div class="header-left">
      <div class="label-tag" data-i18n="mapTag">Vitoria-Gasteiz — Red Sensores Municipales</div>
      <h1 data-i18n="mapTitle">Mapa de <span>Estaciones</span></h1>
      <p class="subtitle" data-i18n="mapSubtitle">Media diaria de ayer por estación y predicción para mañana. Haz click sobre un punto para ver el detalle.</p>
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
  <div class="traffic-iframe-container">
    <iframe id="trafficIframe" src="traffic_map.html" title="Mapa de Tráfico"></iframe>
  </div>
</div>

<div id="view-foresight" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag">XAI — EXPLAINABLE AI</div>
      <h1>Foresight <span>AI</span></h1>
      <p class="subtitle">Análisis avanzado de los factores meteorológicos y de contexto que influencian la predicción del modelo para mañana (SHAP Values).</p>
    </div>
  </div>
  
  <div class="controls">
    <span class="controls-label">Contaminante</span>
    <div class="tab-group" id="contTabsForesight">
      <button class="tab active" onclick="selectContFs('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContFs('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContFs('PM2.5', this)">PM2.5</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label">ZONA</span>
    <div class="tab-group" id="zoneTabsForesight">
      <button class="tab active" onclick="selectZoneFs('zbe', this)">ZBE</button>
      <button class="tab" onclick="selectZoneFs('out', this)">FUERA DE ZBE</button>
    </div>
  </div>

  <div class="charts-section" style="display: grid; grid-template-columns: minmax(300px, 1.2fr) minmax(300px, 1.8fr); gap: 40px;">
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 30px;">
      <h3 style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--accent); letter-spacing:1px; margin-bottom:16px; text-transform:uppercase;">Síntesis Narrativa (LLM)</h3>
      <div id="fs-narrative" style="font-size:15px; line-height:1.7; color:var(--text); white-space: pre-line;">Cargando narrativa...</div>
    </div>
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 30px; display: flex; flex-direction: column;">
      <h3 style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); letter-spacing:1px; margin-bottom:12px; text-transform:uppercase;">Top Factores de Impacto (SHAP)</h3>
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

let currentTheme = 'light';
let mapInstance = null;
let mapTileLayer = null;
let _mapInitialized = false;

let currentLang = localStorage.getItem('vitoria_lang') || 'eu';
if (currentLang !== 'es' && currentLang !== 'eu') currentLang = 'eu';
const predDate = "__PRED_DATE_PLACEHOLDER__";

const getCssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SUMMARY_STATS = __SUMMARY_DATA_PLACEHOLDER__;
const DID_RESULTS = __DID_DATA_PLACEHOLDER__;

function movingAvg(arr, windowSize) {
  return arr.map((_, i, a) => {
    let start = Math.max(0, i - Math.floor(windowSize/2));
    let end = Math.min(a.length, i + Math.floor(windowSize/2) + 1);
    let sub = a.slice(start, end).filter(v => v !== null && !isNaN(v));
    return sub.length ? sub.reduce((s, v) => s + v, 0) / sub.length : null;
  });
}

function fmt(v) { return (v === null || v === undefined) ? 'N/D' : v.toFixed(1); }

const translations = {
  es: {
    mainTitle: "Vitoria-Gasteiz — Análisis Calidad del Aire y ZBE",
    nav1: "1. Predicción Operativa",
    nav2: "2. Monitor Interactivo Diario",
    nav3: "3. Validación Causal",
    nav4: "4. Mapa de Estaciones",
    nav5: "5. Mapa de Tráfico",
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
    auditTitle: "<strong>Auditoría del Modelo:</strong> Predicción vs Real (Últimos 7 días)",
    backtestTitle: "<strong>Validación de Ayer</strong> — Backtesting de Precisión",
    colParam: "Parametro",
    colPred: "Predicción Ayer (v8)",
    colReal: "Medición Real",
    colDev: "Desviación",
    colStatus: "Estado",
    perfTitle: "<strong>Rendimiento Modelo v8</strong> — Métricas de Validación Cruzada (5 Fold)",
    colContZone: "Contaminante + Zona",
    colR2: "R²",
    colNFeatures: "N. Features",
    mapeNote: "MAPE calculado solo sobre filas con valor observado > 1 µg/m³.",
    mapeThreshold: "Umbral de aceptación: MAPE < 25%, R² > 0.35.",
    mapTag: "Vitoria-Gasteiz — Red Sensores Municipales",
    mapTitle: "Mapa de <span>Estaciones</span>",
    mapSubtitle: "Medias diarias de ayer y predicción para mañana. Haga clic en un punto para ver detalles.",
    mapLegendTitle: "Semáforo ICA",
    icaGood: "Buena (≤25)",
    icaMod: "Moderada (25-50)",
    icaBad: "Mala (50-75)",
    icaVeryBad: "Muy Mala (>75)",
    trafficTitle: "Mapa de Tráfico",
    sumObs: "Observado (media)",
    sumCF: "Contrafactual (meteo-puro)",
    sumEffect: "Efecto ZBE (meteo-puro)",
    sumRange: "Rango efecto ZBE",
    sumPeriod: "Sep 2025 → Mar 2026",
    sumBound: "Límite conservador",
    sumMeteoLag: "Meteo-puro vs Con-Lags",
    sumRobRed: "✓ Reducción robusta",
    sumUncertain: "⚠ Incierto",
    sumIncrease: "✕ Incremento neto",
    v8Fig1: "<strong>Figura 1</strong> — Observado vs Counterfactual",
    v8Fig2: "<strong>Figura 2</strong> — Banda de incertidumbre del efecto ZBE",
    v8Fig3: "<strong>Figura 3</strong> — Gap medio por contaminante y zona",
    v9Fig1: "<strong>Figura 1</strong> — Control Sintético (Serie Suavizada)",
    v9Fig2: "<strong>Figura 2</strong> — Event Study DiD",
    v10Tomorrow: "mañana",
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
    v10Good: "🟢 BUENA",
    v10Mod: "🟡 MODERADA",
    v10Bad: "🔴 MALA",
    backReal: "Medición Real",
    backPred: "Predicción",
    backWait: "⏳ Esperando sensores municipales",
    backExcel: "✓ Precisión Excelente",
    backGood: "✓ Precisión Buena",
    backAccept: "✓ Precisión Aceptable",
    backReview: "🔴 Revisar Modelo",
    mapDailyAyer: "Medias diarias de ayer",
    mapPredManana: "▶ Predicción mañana",
    mapZone: "Zona",
    mapNote: "⚠ Los valores son medias diarias, no lecturas horarias",
    colRMSE: "RMSE (µg/m³)",
    colMAE: "MAE (µg/m³)",
    colMAPE: "MAPE %",
  },
  eu: {
    mainTitle: "Vitoria-Gasteiz — Airearen Kalitatearen eta EGEaren Analisia",
    nav1: "1. Iragarpen Operatiboa",
    nav2: "2. Eguneroko Monitore Interaktiboa",
    nav3: "3. Baliozkotze Kausala",
    nav4: "4. Estazioen Mapa",
    nav5: "5. Trafikoaren Mapa",
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
    v10Title: "Biharko aurreikusitako aire-kalitatea",
    v10Calculating: "KALKULATZEN...",
    auditTitle: "<strong>Ereduaren Ikuskapena:</strong> Aurreikusitakoa vs Erreala (Azken 7 egunak)",
    backtestTitle: "<strong>Atzoko Baliozkotzea</strong> — Doitasunaren Backtestinga",
    colParam: "Parametroa",
    colPred: "Atzoko Aurreikuspena (v8)",
    colReal: "Neurketa Erreala",
    colDev: "Desbideratzea",
    colStatus: "Egoera",
    perfTitle: "<strong>v8 Ereduaren Errendimendua</strong> — Baliozkotze Gurutzatuko Metrikak (5 Fold)",
    colContZone: "Kutsatzailea + Eremua",
    colR2: "R²",
    colNFeatures: "Ezaugarri Kopurua",
    mapeNote: "MAPE behatutako balioa > 1 µg/m³ duten errenkadetan soilik kalkulatua.",
    mapeThreshold: "Onarpen-atalasea: MAPE < %25, R² > 0.35.",
    mapTag: "Vitoria-Gasteiz — Udaltzaingoaren sentsoreen sarea",
    mapTitle: "Estazioen <span>Mapa</span>",
    mapSubtitle: "Atzoko eguneroko batez bestekoak eta biharko aurreikuspenak. Egin klik puntu batean xehetasunak ikusteko.",
    mapLegendTitle: "ICA Semaforoa",
    icaGood: "Ona (≤25)",
    icaMod: "Ertaina (25-50)",
    icaBad: "Txarra (50-75)",
    icaVeryBad: "Oso txarra (>75)",
    trafficTitle: "Trafikoaren Mapa",
    sumObs: "Behatua (batez bestekoa)",
    sumCF: "Kontrafaktuala (meteo-purua)",
    sumEffect: "EGEaren efektua (meteo-purua)",
    sumRange: "EGE efektuaren heina",
    sumPeriod: "2025 Ira → 2026 Mar",
    sumBound: "Muga kontserbadorea",
    sumMeteoLag: "Meteo-purua vs Lag-ekin",
    sumRobRed: "✓ Murrizketa sendoa",
    sumUncertain: "⚠ Ziurgabea",
    sumIncrease: "✕ Gehikuntza garbia",
    v9Adj: "Egokitze Metrika (EGE aurretik)",
    v9Impact: "EGE osteko eragin estimatua",
    v9Diag: "Diagnostiko Kausala",
    v9Trend: "Pre-Trend Test-a",
    v9ParallelOK: "✓ OK Parallel Trends",
    v10Good: "🟢 ONA",
    v10Mod: "🟡 ERTAINA",
    v10Bad: "🔴 TXARRA",
    backReal: "Neurketa Erreala",
    backPred: "Aurreikuspena",
    backWait: "⏳ Sentsoreen zain",
    backExcel: "✓ Doitasun bikaina",
    backGood: "✓ Doitasun ona",
    backAccept: "✓ Doitasun onargarria",
    backReview: "🔴 Eredua berrikusi",
    mapDailyAyer: "Atzoko eguneroko batez bestekoak",
    mapPredManana: "▶ Biharko aurreikuspena",
    mapZone: "Eremua",
    mapNote: "⚠ Balioak eguneroko batez bestekoak dira, ez orduko irakurketak",
    colRMSE: "RMSE (µg/m³)",
    colMAE: "MAE (µg/m³)",
    colMAPE: "MAPE %",
    v8Fig1: "<strong>1. Irudia</strong> — Behatua vs Kontrafaktuala",
    v8Fig2: "<strong>2. Irudia</strong> — EGE efektuaren ziurgabetasun-banda",
    v8Fig3: "<strong>3. Irudia</strong> — Gap-a batez beste, kutsatzaile eta eremuka",
    v9Fig1: "<strong>1. Irudia</strong> — Kontrol Sintetikoa (Serie Leundua)",
    v9Fig2: "<strong>2. Irudia</strong> — Event Study DiD",
    v10Tomorrow: "bihar",
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
    didNo: "Ez"
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

function updateRiskDate() {
    const el = document.getElementById('riskDateSpan');
    if (el) {
        const t = translations[currentLang];
        el.innerHTML = `<span style="text-transform: uppercase;">${t.v10Tomorrow}</span> ${predDate}`;
    }
}

function toggleLang() {
  currentLang = currentLang === 'es' ? 'eu' : 'es';
  localStorage.setItem('vitoria_lang', currentLang);
  updateI18n();
  updateRiskDate();
  
  // Re-renderizar componentes que dependen de traducciones
  renderSummaryCards();
  renderFig1();
  renderFig2();
  renderFig3();
  renderDidTable();
  renderV9Cards();
  renderMetricsTable();
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
      const url = currentTheme === 'dark' 
          ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' 
          : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
      mapTileLayer.setUrl(url);
      if (mapInstance) mapInstance.closePopup();
  }
}

// ===========================================================================
// JAVASCRIPT ORIGINAL DE D1 (ADAPTADO A VARIABLES CSS)
// ===========================================================================
let currentCont = 'NO2'; let currentZone = 'zbe'; let fig1Chart = null, fig2Chart = null, fig3Chart = null;

const monthsEU = ['urt.', 'ots.', 'mar.', 'api.', 'mai.', 'eka.', 'uzt.', 'abu.', 'ira.', 'urr.', 'aza.', 'abe.'];
function formatDate(d) {
  const date = new Date(d);
  if (currentLang === 'eu') {
    return `${date.getDate()} ${monthsEU[date.getMonth()]}`;
  }
  return date.toLocaleDateString('es-ES', { day:'numeric', month:'short' });
}
function getClass(val) { if (val < -5)  return 'neg'; if (val > 5)   return 'pos'; return 'neutral'; }

function renderSummaryCards() {
  const s = SUMMARY_STATS[`${currentCont}_${currentZone}`]; if (!s) return;
  const t = translations[currentLang];
  const low = Math.min(s.pure, s.lags), high = Math.max(s.pure, s.lags), gap_abs = ((s.obs - s.cf_pure)).toFixed(2);
  const cards = [ 
      { label: t.sumObs, value: s.obs.toFixed(2), sub: s.unit, cls: 'neutral', range: t.sumPeriod }, 
      { label: t.sumCF, value: s.cf_pure.toFixed(2), sub: `${s.unit} ${t.sumSinZBE}`, cls: 'neutral', range: t.sumBound }, 
      { label: t.sumEffect, value: `${s.pure > 0 ? '+' : ''}${s.pure.toFixed(1)}%`, sub: `${gap_abs} ${s.unit} ${t.sumAbsoluto}`, cls: getClass(s.pure), range: t.sumBound }, 
      { label: t.sumRange, value: `[${low.toFixed(1)}%, ${high.toFixed(1)}%]`, sub: t.sumMeteoLag, cls: low < 0 && high < 0 ? 'neg' : (low > 0 && high > 0 ? 'pos' : 'neutral'), range: low < 0 && high < 0 ? t.sumRobRed : (low < 0 ? t.sumUncertain : t.sumIncrease) } 
  ];
  document.getElementById('summaryCards').innerHTML = cards.map(c => `<div class="summary-card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div><div class="card-sub">${c.sub}</div><div class="card-range">${c.range}</div></div>`).join('');
}

function renderFig1() {
  const key = `${currentCont}_${currentZone}`; 
  if (!cfData || !cfData[key] || !cfData[key]['METEO-PURO']) {
    console.warn("No data for Fig1:", key);
    if(fig1Chart) fig1Chart.destroy();
    return;
  }
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'], dates = pure.dates.map(formatDate); const obsSmooth = movingAvg(pure.observed, 7);
  const t = translations[currentLang];
  const ctx = document.getElementById('fig1').getContext('2d');
  if (fig1Chart) fig1Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');
  
  fig1Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_fillUpper', data: pure.counterfactual, borderColor: 'transparent', backgroundColor: getCssVar('--observed') + '20', fill: '+1', pointRadius: 0, tension: 0.3 }, { label: t.fig1Obs, data: obsSmooth, borderColor: getCssVar('--observed'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false, order: 1 }, { label: t.fig1CFPure, data: movingAvg(pure.counterfactual, 7), borderColor: getCssVar('--cf-pure'), backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, tension: 0.3, fill: false, order: 2 }, { label: t.fig1CFLags, data: movingAvg(lags.counterfactual, 7), borderColor: getCssVar('--cf-lags'), backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [3,3], pointRadius: 0, tension: 0.3, fill: false, order: 3 } ] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: text, font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2)} µg/m³`; } } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: grid } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(1) + ' µg/m³' }, grid: { color: grid }, title: { display: true, text: `µg/m³`, color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig2() {
  const key = `${currentCont}_${currentZone}`; 
  if (!cfData || !cfData[key] || !cfData[key]['METEO-PURO']) {
    console.warn("No data for Fig2:", key);
    if(fig2Chart) fig2Chart.destroy();
    return;
  }
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'], dates = pure.dates.map(formatDate); const gapPureSmooth = movingAvg(pure.gap_pct, 14), gapLagsSmooth = movingAvg(lags.gap_pct, 14), zero = dates.map(() => 0);
  const t = translations[currentLang];
  const ctx = document.getElementById('fig2').getContext('2d');
  if (fig2Chart) fig2Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');

  fig2Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_bandUpper', data: gapPureSmooth.map((p, i) => Math.max(p, gapLagsSmooth[i])), borderColor: 'transparent', backgroundColor: getCssVar('--accent') + '26', fill: '+1', pointRadius: 0, tension: 0.4 }, { label: '_bandLower', data: gapPureSmooth.map((p, i) => Math.min(p, gapLagsSmooth[i])), borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0, tension: 0.4 }, { label: t.fig2NoEffect, data: zero, borderColor: text, borderWidth: 1, borderDash: [2,4], pointRadius: 0, fill: false, order: 10 }, { label: t.fig2GapPure, data: gapPureSmooth, borderColor: getCssVar('--cf-pure'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false, order: 1 }, { label: t.fig2GapLags, data: gapLagsSmooth, borderColor: getCssVar('--cf-lags'), backgroundColor: 'transparent', borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.4, fill: false, order: 2 } ] }, options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: text, font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}%`; } } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: grid } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(0) + '%' }, grid: { color: grid }, title: { display: true, text: 'Gap %', color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig3() {
  const keys = Object.keys(SUMMARY_STATS); const vals = keys.map(k => SUMMARY_STATS[k].pure);
  const labels = keys.map(k => k.replace('PM2.5','PM₂.₅').replace('_',' '));
  const colors = vals.map(v => v < -5 ? getCssVar('--green')+'CC' : v > 5 ? getCssVar('--red')+'CC' : getCssVar('--yellow')+'CC');
  const borders = vals.map(v => v < -5 ? getCssVar('--green') : v > 5 ? getCssVar('--red') : getCssVar('--yellow'));
  const t = translations[currentLang];
  const ctx = document.getElementById('fig3').getContext('2d');
  if (fig3Chart) fig3Chart.destroy();
  
  const text = getCssVar('--muted'); const grid = getCssVar('--border'); const bg = getCssVar('--surface2'); const title = getCssVar('--text');

  fig3Chart = new Chart(ctx, { type: 'bar', data: { labels: labels, datasets: [{ label: t.fig3Label, data: vals, backgroundColor: colors, borderColor: borders, borderWidth: 1, borderRadius: 3 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { backgroundColor: bg, borderColor: grid, borderWidth: 1, titleColor: title, bodyColor: text, titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => ` Efecto: ${ctx.parsed.y.toFixed(1)}%` } } }, scales: { x: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 11 } }, grid: { display: false } }, y: { ticks: { color: text, font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v + '%' }, grid: { color: grid }, title: { display: true, text: 'Gap %', color: text, font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderDidTable() {
  const t = translations[currentLang];
  const rows = Object.entries(DID_RESULTS).map(([cont, d]) => {
    const sig = d.significant;
    const bCls = d.beta3 < -0.1 ? 'beta-neg' : d.beta3 > 0.1 ? 'beta-pos' : 'beta-neu';
    const sign = d.beta3 > 0 ? '+' : '';
    const pFmt = d.pvalue < 0.0001 ? '<0.0001' : d.pvalue.toFixed(4);
    const sigText = sig ? t.didYes : t.didNo;
    
    return `<tr>
      <td><strong>${cont}</strong></td>
      <td class="${bCls}">${sign}${d.beta3.toFixed(3)} µg/m³</td>
      <td class="${sig ? 'sig-yes' : 'sig-no'}">${pFmt} ${sig ? '✓' : ''}</td>
      <td>[${d.ci_low.toFixed(3)}, ${d.ci_high.toFixed(3)}]</td>
      <td>${d.r2.toFixed(3)}</td>
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
  document.getElementById('summaryCardsV9').innerHTML = `<div class="summary-card"><div class="card-label">${t.v9Adj}</div><div class="card-value neutral">R² = ${data.preR2}</div></div><div class="summary-card"><div class="card-label">${t.v9Impact}</div><div class="card-value ${colorClass}">${data.gap}</div></div><div class="summary-card"><div class="card-label">${t.v9Diag}</div><div class="card-value neutral" style="font-size: 16px; margin-top:8px;">${data.desc}</div></div><div class="summary-card"><div class="card-label">${t.v9Trend}</div><div class="card-value neg" style="font-size: 16px; margin-top:8px;">${t.v9ParallelOK}</div></div>`;
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
let currentContV10 = 'NO2', currentZoneV10 = 'out', perfChart = null;

function renderDashboard3() {
  const zoneData = manana[currentZoneV10] || {};
  const no2 = zoneData['NO2'] || 0;
  const pm25 = zoneData['PM2.5'] || 0;
  const pm10 = zoneData['PM10'] || 0;
  const t = translations[currentLang];

  let badge = document.getElementById('riskBadge');
  let riskLevel = 0; 
  if (no2 >= 40 || pm10 >= 20 || pm25 >= 10) riskLevel = 1;
  if (no2 >= 90 || pm10 >= 40 || pm25 >= 20) riskLevel = 2;

  if (riskLevel === 0) { badge.innerText = t.v10Good; badge.style.color = "var(--green)"; badge.style.borderColor = "var(--green)"; badge.style.backgroundColor = getCssVar('--green')+"1A"; }
  else if (riskLevel === 1) { badge.innerText = t.v10Mod; badge.style.color = "var(--yellow)"; badge.style.borderColor = "var(--yellow)"; badge.style.backgroundColor = getCssVar('--yellow')+"1A"; }
  else { badge.innerText = t.v10Bad; badge.style.color = "var(--red)"; badge.style.borderColor = "var(--red)"; badge.style.backgroundColor = getCssVar('--red')+"1A"; }

  document.getElementById('val-no2').innerHTML = `${no2.toFixed(1)} <span style="font-size:10px; color:var(--accent); vertical-align:middle; border:1px solid var(--accent); padding:1px 4px; border-radius:3px; margin-left:5px">V2 REFINADO</span>`;
  document.getElementById('val-pm25').innerHTML = `${pm25.toFixed(1)} <span style="font-size:10px; color:var(--accent); vertical-align:middle; border:1px solid var(--accent); padding:1px 4px; border-radius:3px; margin-left:5px">V2 REFINADO</span>`;
  document.getElementById('val-pm10').innerHTML = `${pm10.toFixed(1)} <span style="font-size:10px; color:var(--accent); vertical-align:middle; border:1px solid var(--accent); padding:1px 4px; border-radius:3px; margin-left:5px">V2 REFINADO</span>`;

  const d = perfStats[currentZoneV10][currentContV10] || {labels:[], real:[], pred:[]};
  const ctx = document.getElementById('perfChart').getContext('2d');
  if(perfChart) perfChart.destroy();
  
  const labels = (perfStats[currentZoneV10].labels || []).map(l => l === 'Ayer' ? (currentLang === 'es' ? 'Ayer' : 'Atzo') : l);
  const text = getCssVar('--muted'); const grid = getCssVar('--border');

  perfChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: labels,
          datasets: [
              { label: t.backReal, data: d.real, borderColor: getCssVar('--observed'), backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.4 },
              { label: t.backPred, data: d.pred, borderColor: getCssVar('--accent'), backgroundColor: 'transparent', borderDash: [5,5], borderWidth: 2, pointRadius: 4, tension: 0.4 }
          ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { color: text, font: { family: 'IBM Plex Mono'} } }, }, scales: { x: { ticks: { color: text }, grid: { color: grid } }, y: { ticks: { color: text }, grid: { color: grid } } } }
  });

  const lastRealDraw = d.real[d.real.length - 1]; 
  const lastPred = d.pred[d.pred.length - 1] || 0;
  let realText, errorText, interpret, interpretColor;
  
  if (lastRealDraw === null || lastRealDraw === undefined) {
      realText = currentLang === 'es' ? "N/D (Pendiente)" : "E/D (Zain)";
      errorText = "N/D";
      interpret = t.backWait;
      interpretColor = "var(--muted)";
  } else {
      const lastReal = lastRealDraw;
      const error = lastPred > 0 ? (((lastReal - lastPred) / lastPred) * 100).toFixed(1) : 0;
      realText = `${lastReal.toFixed(1)} µg/m³`;
      errorText = `${error > 0 ? '+' : ''}${error}%`;
      const errorColor = error > 0 ? "var(--red)" : "var(--green)";
      const absError = Math.abs(error);
      if (absError <= 10) { interpretColor = "var(--green)"; interpret = t.backExcel; } 
      else if (absError <= 20) { interpretColor = "var(--green)"; interpret = t.backGood; } 
      else if (absError <= 30) { interpretColor = "var(--green)"; interpret = t.backAccept; } 
      else { interpretColor = "var(--yellow)"; interpret = t.backReview; }

      document.getElementById('backtestTable').innerHTML = `
        <tr>
          <td><strong>${currentContV10} (${currentZoneV10.toUpperCase()})</strong></td>
          <td>${lastPred.toFixed(1)} µg/m³</td>
          <td>${realText}</td>
          <td style="color:${errorColor}; font-weight:bold;">${errorText}</td>
          <td style="color:${interpretColor}; font-weight:bold;">${interpret}</td>
        </tr>
      `;
  }
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
    const metaStr = meta ? `<div style="font-size:9px; color:var(--accent); margin-top:2px">+${meta.improvement_pct}% mejora meta-model</div>` : '';

    return `<tr>
      <td><strong>${cont}</strong> <span style="color:var(--muted)">${zone.toUpperCase()}</span>${metaStr}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${m.cv_rmse.toFixed(2)}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${m.cv_mae.toFixed(2)}</td>
      <td style="color:${r2Color};font-family:'IBM Plex Mono',monospace">${m.cv_r2.toFixed(3)}</td>
      <td style="color:${mapeColor};font-family:'IBM Plex Mono',monospace;font-weight:bold">${m.cv_mape.toFixed(1)}%</td>
      <td style="color:var(--muted);font-family:'IBM Plex Mono',monospace">${m.n_features}</td>
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
  
  const tileUrl = currentTheme === 'dark' 
      ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' 
      : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
      
  mapTileLayer = L.tileLayer(tileUrl, {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: 'abcd', maxZoom: 19
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
        <div class="popup-section-title">${t.mapPredManana}</div>
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
let currentContFs = 'NO2', currentZoneFs = 'zbe', fsChart = null;

function renderForesight() {
  const key = `${currentContFs}_${currentZoneFs}_d1`;
  const tData = targetsData[key] || {};
  const fs = tData.foresight || null;
  
  const narrativeDiv = document.getElementById('fs-narrative');
  if (!fs) {
    narrativeDiv.innerHTML = "<span style='color:var(--muted)'>No hay análisis disponible para esta selección.</span>";
    if (fsChart) { fsChart.destroy(); fsChart = null; }
    return;
  }
  
  let narrativeText = fs.narrative || "Este modelo no ha generado narrativa LLM (posiblemente porque no es el target principal o falla la API).";
  if (!narrativeText && fs.error) {
    narrativeText = `Error al calcular SHAP: ${fs.error}`;
  }
  narrativeDiv.textContent = narrativeText;
  
  const posTop = fs.positive_top || [];
  const negTop = fs.negative_top || [];
  let factors = [...posTop, ...negTop].sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
  factors = factors.slice(0, 8); // Top 8 features
  
  const ctx = document.getElementById('fsChart').getContext('2d');
  if (fsChart) fsChart.destroy();
  
  const labels = factors.map(f => f.feature.length > 22 ? f.feature.slice(0, 22) + '...' : f.feature);
  const values = factors.map(f => f.value);
  const colors = factors.map(f => f.value > 0 ? getCssVar('--red') : getCssVar('--green'));
  
  const text = getCssVar('--text'); const muted = getCssVar('--muted'); const grid = getCssVar('--border');
  
  fsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Impacto en µg/m³',
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

// ── INIT ───────────────────────────────────────────────────────────────────
window.onload = function() {
  updateI18n();
  updateRiskDate();
  
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
  updateV9Images();
  
  // Si estamos en la pestaña de tráfico, sincronizar lenguaje
  const iframe = document.getElementById('trafficIframe');
  if (iframe) {
      iframe.src = `traffic_map.html?lang=${currentLang}`;
  }
  console.log("DASHBOARD INITIALIZED SUCCESSFULLY");
};
</script>
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
content = content.replace('__IMG_BASE_PATH__', img_base_path)
content = content.replace('__PRED_DATE_PLACEHOLDER__', prediction_date_str)

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Listo! Archivo {output_html} generado correctamente.")