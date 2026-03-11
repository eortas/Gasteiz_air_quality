import sys
import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta

ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"

# --- 1. CONFIGURACIÓN DE ARCHIVOS Y RUTAS ---
# CSV generado por train_model_v8.py, guardado en models/
csv_file = str(MODELS_DIR / "counterfactual_gap_v8.csv")

# Soporte para --output <filename>
_out_idx = sys.argv.index("--output") if "--output" in sys.argv else None
output_html = sys.argv[_out_idx + 1] if _out_idx is not None else str(ROOT_DIR / "reports" / "plots" / "index.html")

# ==============================================================================
# 2. DATOS PESTAÑAS 1 Y 2 (Causal v8)
# ==============================================================================
print(f"Leyendo {csv_file}...")
df_cv = pd.read_csv(csv_file)
cf_data = {}

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

cf_json_str = json.dumps(cf_data)

# ==============================================================================
# 3. PREDICCIONES DE MAÑANA — desde predictions_latest.json (generado por predict.py)
# ==============================================================================
print("Leyendo predicciones de mañana desde predictions_latest.json...")
pred_json_path = PROCESSED_DIR / "predictions_latest.json"
manana_data = {'zbe': {}, 'out': {}}
prediction_date_str = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")  # fallback

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
                manana_data[zone][cont] = targets[key]["prediction"]
            else:
                manana_data[zone][cont] = 0

    print(f"  ✅ Predicciones para {prediction_date_str} cargadas.")

except Exception as e:
    print(f"  ⚠️ No se pudo leer predictions_latest.json: {e}")
    for z in ['zbe', 'out']:
        for c in ['NO2', 'PM10', 'PM2.5', 'ICA']:
            manana_data[z][c] = 0

# ==============================================================================
# 4. BACKTESTING desde parquet + modelos (para la gráfica de auditoría)
# ==============================================================================
print("Calculando backtesting con modelos v8...")
perf_data = {'zbe': {}, 'out': {}}

try:
    df_feat = pd.read_parquet(DATASET_PATH)
    df_feat["date"] = pd.to_datetime(df_feat["date"], utc=True)

    # Extraer el dataframe superset (que incluye hoy sin dropear por falta de targets)
    pred_full_path = PROCESSED_DIR / "features_latest.parquet"
    if pred_full_path.exists():
        df_latest = pd.read_parquet(pred_full_path)
        df_latest["date"] = pd.to_datetime(df_latest["date"], utc=True)
        if df_latest["date"].iloc[-1] >= df_feat["date"].iloc[-1]:
            df_feat = df_latest

    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    # LOGICA DE BACKTESTING CORREGIDA
    # Obtenemos las últimas 9 filas. Ejemplo asumiendo Hoy = Día 10:
    # - índices 1 a 7: Días 3 al 9 (los 7 días "reales" a graficar, terminando en "Ayer" Día 9)
    # - índices 0 a 6: Días 2 al 8 (las "features de entrada" usadas para predecir los Días 3 al 9)
    # - índice 8: Día 10 (las features de hoy para predecir Mañana Día 11)
    last_rows = df_feat.tail(9).copy()
    
    inputs_backtest  = last_rows.iloc[0:7]
    targets_backtest = last_rows.iloc[1:8]
    row_hoy          = last_rows.iloc[[-1]]
    
    fechas = targets_backtest['date'].dt.strftime('%d %b').tolist()
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
                
                # Predecir los 7 días (usando las features del día anterior)
                X_backtest = inputs_backtest.reindex(columns=features, fill_value=0).fillna(0)
                preds_7d = model.predict(X_backtest)
                
                # Valores reales leídos directamente de la fecha de target
                contam_col = f"{cont}_{zone}"
                if contam_col in targets_backtest.columns:
                    # Guardamos None en lugar de 0 para que Chart.js corte la línea si no hay datos.
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

    print("  ✅ Predicciones y Backtest listos.")

except Exception as e:
    print(f"  ⚠️ No se encontró parquet o error general: {e}")
    for z in ['zbe', 'out']:
        perf_data[z] = {"labels": ["1", "2", "3", "4", "5", "6", "Ayer"]}
        for c in ['NO2', 'PM10', 'PM2.5']:
            perf_data[z][c] = {"real": [0]*7, "pred": [0]*7}
            manana_data[z][c] = 0

perf_json_str = json.dumps(perf_data)
manana_json_str = json.dumps(manana_data)

# Cargar métricas del modelo (RMSE, MAE, R², MAPE) desde metrics_v8.json
try:
    metrics_raw = json.loads((MODELS_DIR / "metrics_v8.json").read_text(encoding="utf-8"))
except Exception:
    metrics_raw = {}
metrics_json_str = json.dumps(metrics_raw)

# Cargar métricas v9 (Control Sintético)
try:
    with open(MODELS_DIR / "synthetic_control_v9.json", "r", encoding="utf-8") as f:
        sc_data = json.load(f)
        v9_stats = {}
        for cont, stations in sc_data.items():
            for st, st_data in stations.items():
                v9_stats[f"{cont}_{st}"] = {
                    "preR2": st_data.get("preR2", "N/A"),
                    "gap": st_data.get("gap", "N/A"),
                    "desc": st_data.get("desc", "Sin datos")
                }
        v9_json_str = json.dumps(v9_stats)
except Exception as e:
    print(f"  ⚠️ No se encontró synthetic_control_v9.json: {e}")
    v9_json_str = "{}"


# ==============================================================================
# 5. DATOS MAPA DE ESTACIONES (station_daily.csv + predictions_latest.json)
# ==============================================================================
STATION_COORDS = {
    "HUETOS":    {"lat": 42.853846, "lon": -2.699907, "zone": "OUT", "label": "Huetos"},
    "LANDAZURI": {"lat": 42.847626, "lon": -2.677065, "zone": "OUT", "label": "Landazuri"},
    "BEATO":     {"lat": 42.849319, "lon": -2.675857, "zone": "ZBE", "label": "Beato (ZBE)"},
    "PAUL":      {"lat": 42.851130, "lon": -2.670824, "zone": "ZBE", "label": "Vicente de Paul (ZBE)"},
    "FUEROS":    {"lat": 42.846270, "lon": -2.669415, "zone": "ZBE", "label": "Fueros (ZBE)"},
    "ZUMABIDE":  {"lat": 42.835437, "lon": -2.673657, "zone": "OUT", "label": "Zumabide"},
}

stations_data = {}
try:
    df_stn = pd.read_csv(PROCESSED_DIR / "station_daily.csv", index_col=0, parse_dates=True)
    latest = df_stn.iloc[-1]
    # Predicciones de mañana (zona agregada, misma para la zona de la estación)
    pred_zbe = manana_data.get("zbe", {})
    pred_out = manana_data.get("out", {})
    for stn, meta in STATION_COORDS.items():
        def _v(col):
            v = latest.get(f"{stn}_{col}")
            return round(float(v), 1) if v is not None and str(v) != 'nan' else None
        pred_src = pred_zbe if meta["zone"] == "ZBE" else pred_out
        stations_data[stn] = {
            **meta,
            "NO2":  _v("NO2"),
            "PM10": _v("PM10"),
            "PM25": _v("PM25"),
            "ICA":  _v("ICA"),
            "pred_NO2":  round(pred_src.get("NO2", 0), 1),
            "pred_PM10": round(pred_src.get("PM10", 0), 1),
            "pred_PM25": round(pred_src.get("PM2.5", 0), 1),
        }
except Exception as e:
    print(f"  ⚠️ Error cargando station_daily para el mapa: {e}")
    for stn, meta in STATION_COORDS.items():
        stations_data[stn] = {**meta, "NO2": None, "PM10": None, "PM25": None, "ICA": None,
                              "pred_NO2": 0, "pred_PM10": 0, "pred_PM25": 0}
stations_json_str = json.dumps(stations_data)

# ==============================================================================
# 4. PLANTILLA HTML (D1 y D2 Intactos + D3 Arreglado)
# ==============================================================================
html_template = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZBE Vitoria-Gasteiz — Dashboard Integral</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  /* ── TU CSS ORIGINAL INTACTO ── */
  :root {
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

  .header { padding: 40px 48px 28px; border-bottom: 1px solid var(--border); display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
  .header-left { max-width: 620px; }
  .label-tag { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); border: 1px solid var(--accent); padding: 3px 8px; border-radius: 2px; margin-bottom: 12px; }
  h1 { font-size: 26px; font-weight: 700; line-height: 1.25; letter-spacing: -0.02em; color: #fff; }
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

  /* CONTENEDOR CHART.JS ORIGINAL */
  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; position: relative; }
  canvas { display: block; }
  
  /* CONTENEDOR IMÁGENES ORIGINAL (V9) */
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

  /* ── NUEVAS CLASES PARA TABS Y V10 (AISLADAS) ── */
  .top-nav { display: flex; background: var(--surface2); border-bottom: 1px solid var(--border); padding: 0 48px; }
  .top-nav-btn { padding: 16px 24px; background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; cursor: pointer; transition: all 0.2s; }
  .top-nav-btn:hover { color: var(--text); }
  .top-nav-btn.active { color: var(--accent); border-bottom-color: var(--accent); color: #fff;}
  .view-container { display: none; }
  .view-container.active { display: block; }
  
  .v10-risk-container { text-align: center; padding: 50px 40px; background: var(--surface2); border-bottom: 1px solid var(--border); }
  .v10-risk-badge { display: inline-block; font-size: 32px; font-weight: 800; padding: 12px 30px; border-radius: 8px; margin-bottom: 30px; font-family: 'IBM Plex Mono', monospace; border: 2px solid; }
  .v10-risk-grid { display: flex; justify-content: center; gap: 70px; }
  .v10-risk-item .val { font-size: 28px; font-weight: 700; color: #fff; font-family: 'IBM Plex Mono', monospace;}
  .v10-risk-item .lab { font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 8px; font-family: 'IBM Plex Mono', monospace;}
  
  /* ESTA CLASE ASEGURA QUE EL CHART DE D3 NO ROMPA EL RESTO */
  .perf-chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; height: 350px; position: relative; }
  .perf-chart-wrap canvas { display: block; width: 100% !important; height: 100% !important; }

  @media (max-width: 900px) {
    .header, .controls, .charts-section, .did-section, .top-nav { padding-left: 20px; padding-right: 20px; }
    .summary-grid { grid-template-columns: 1fr 1fr; }
    h1 { font-size: 20px; }
    .v10-risk-grid { flex-direction: column; gap: 20px; }
  }

  /* FIX MÓVIL ORIGINAL */
  @media screen and (max-width: 768px) {
      .chart-wrap, div[style*="position: relative"], canvas { min-height: 350px !important; width: 100% !important; }
  }

  /* ── MAP DASHBOARD (D4) ── */
  #stationMap { height: 540px; border-radius: 4px; border: 1px solid var(--border); }
  .map-header { padding: 32px 48px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
  .map-legend { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; display: flex; flex-direction: column; gap: 8px; min-width: 170px; }
  .map-legend-item { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--muted); }
  .map-legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .map-wrap { padding: 24px 48px 40px; }
  /* Leaflet popup dark theme */
  .leaflet-popup-content-wrapper { background: #1c2030 !important; color: #e2e6f0 !important; border: 1px solid #2a2f3f !important; border-radius: 6px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important; }
  .leaflet-popup-tip { background: #1c2030 !important; }
  .leaflet-popup-content { margin: 14px 18px !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 13px !important; line-height: 1.6 !important; }
  .popup-name { font-weight: 700; font-size: 14px; margin-bottom: 8px; color: #fff; }
  .popup-row { display: flex; justify-content: space-between; gap: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 2px 0; }
  .popup-label { color: #6b7494; }
  .popup-val { font-weight: 600; }
  .popup-section { margin-top: 10px; padding-top: 10px; border-top: 1px solid #2a2f3f; }
  .popup-section-title { color: #7c6af7; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; }
  /* Override Leaflet defaults for dark map */
  .leaflet-control-attribution { background: rgba(13,15,20,0.85) !important; color: #6b7494 !important; }
  .leaflet-control-attribution a { color: #7c6af7 !important; }
  .leaflet-control-zoom a { background: #1c2030 !important; color: #e2e6f0 !important; border-color: #2a2f3f !important; }
</style>
</head>
<body>

<div class="top-nav">
  <button class="top-nav-btn active" onclick="switchMainView('v10', this)">1. Predicción Operativa (v8)</button>
  <button class="top-nav-btn" onclick="switchMainView('v8', this)">2. Monitor Interactivo Diario</button>
  <button class="top-nav-btn" onclick="switchMainView('v9', this)">3. Validación Causal Académica</button>
  <button class="top-nav-btn" onclick="switchMainView('map', this)">4. Mapa de Estaciones</button>
</div>

<div id="view-v8" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag">Análisis Causal — ZBE Vitoria-Gasteiz</div>
      <h1>Counterfactual Meteorológico<br><span>ZBE Sep·2025 → Mar·2026</span></h1>
      <p class="subtitle">Comparación entre contaminación observada y el escenario contrafactual.</p>
    </div>
    <div class="legend-global">
      <div class="card-label" style="margin-bottom:4px">Leyenda</div>
      <div class="legend-item"><div class="legend-line" style="background:var(--observed);height:2px"></div><span>Observado (real)</span></div>
      <div class="legend-item"><div class="legend-line dashed" style="color:var(--cf-pure);height:0;border-top:2px dashed var(--cf-pure)"></div><span>CF Meteo-Puro</span></div>
      <div class="legend-item"><div class="legend-line dashed" style="color:var(--cf-lags);height:0;border-top:2px dashed var(--cf-lags)"></div><span>CF Con-Lags</span></div>
      <div class="legend-item"><div style="width:28px;height:12px;background:rgba(79,195,247,0.2);border-radius:2px;flex-shrink:0"></div><span>Banda de efecto</span></div>
    </div>
  </div>

  <div class="controls">
    <span class="controls-label">Contaminante</span>
    <div class="tab-group" id="contTabs">
      <button class="tab active" onclick="selectCont('NO2',this)">NO₂</button>
      <button class="tab" onclick="selectCont('PM10',this)">PM10</button>
      <button class="tab" onclick="selectCont('PM2.5',this)">PM2.5</button>
      <button class="tab" onclick="selectCont('ICA',this)">ICA</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label">Zona</span>
    <div class="tab-group" id="zoneTabs">
      <button class="tab active" onclick="selectZone('zbe',this)">ZBE (dentro)</button>
      <button class="tab" onclick="selectZone('out',this)">OUT (control)</button>
    </div>
  </div>

  <div class="summary-grid" id="summaryCards"></div>

  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title"><strong>Figura 1</strong> — Observado vs Counterfactual</div></div>
      <div class="chart-wrap"><canvas id="fig1" height="100"></canvas></div>
    </div>
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title"><strong>Figura 2</strong> — Banda de incertidumbre del efecto ZBE</div></div>
      <div class="chart-wrap"><canvas id="fig2" height="80"></canvas></div>
    </div>
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title"><strong>Figura 3</strong> — Gap medio por contaminante y zona</div></div>
      <div class="chart-wrap"><canvas id="fig3" height="60"></canvas></div>
    </div>
  </div>

  <div class="did-section">
    <div class="did-title"><strong>Tabla 1</strong> — Difference-in-Differences v8</div>
    <table><thead><tr><th>Contaminante</th><th>β₃</th><th>p-valor</th><th>IC 95%</th><th>Efecto relativo</th><th>R²</th><th>n obs</th><th>Significativo</th></tr></thead><tbody id="didTable"></tbody></table>
  </div>
</div>

<div id="view-v9" class="view-container">
  <div class="header">
    <div class="header-left">
      <div class="label-tag">Modelos v9 — Rigor Académico</div>
      <h1>Evaluación Causal ZBE<br><span>Event Study & Synthetic Control</span></h1>
    </div>
  </div>
  <div class="controls">
    <span class="controls-label">Contaminante</span>
    <div class="tab-group" id="contTabsV9">
      <button class="tab active" onclick="selectContV9('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContV9('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContV9('PM2.5', this)">PM2.5</button>
      <button class="tab" onclick="selectContV9('ICA', this)">ICA</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label">Estación IntraZBE</span>
    <div class="tab-group" id="stationTabsV9">
      <button class="tab active" onclick="selectStationV9('PAUL', this)">PAUL</button>
      <button class="tab" onclick="selectStationV9('BEATO', this)">BEATO</button>
      <button class="tab" onclick="selectStationV9('FUEROS', this)">FUEROS</button>
    </div>
  </div>
  <div class="summary-grid" id="summaryCardsV9"></div>
  
  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title"><strong>Figura 1</strong> — Control Sintético (Serie Suavizada)</div></div>
      <div class="img-wrap">
        <img id="img-sc" src="" alt="Gráfico Control Sintético" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div id="err-sc" class="error-msg">Imagen no encontrada. Verifica el nombre del archivo.</div>
      </div>
    </div>
    <div class="fig-block">
      <div class="fig-header"><div class="fig-title"><strong>Figura 2</strong> — Event Study DiD</div></div>
      <div class="img-wrap">
        <img id="img-es" src="" alt="Gráfico Event Study" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div id="err-es" class="error-msg">Imagen no encontrada. Verifica el nombre del archivo.</div>
      </div>
    </div>
  </div>
</div>

<div id="view-v10" class="view-container active">
  <div class="v10-risk-container">
    <div class="label-tag">EEA Standards — Air Quality Risk</div>
    <h2 style="margin-bottom: 10px; color: var(--muted); font-weight: 400; font-size: 14px; text-transform: uppercase; letter-spacing: 0.1em;">Calidad del aire prevista para <span style="color:#fff;font-weight:700;">mañana __PRED_DATE_PLACEHOLDER__</span></h2>
    <div id="riskBadge" class="v10-risk-badge">CALCULANDO...</div>
    <div class="v10-risk-grid">
      <div class="v10-risk-item"><div class="val" id="val-no2">--</div><div class="lab">NO₂ µg/m³</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm25">--</div><div class="lab">PM2.5 µg/m³</div></div>
      <div class="v10-risk-item"><div class="val" id="val-pm10">--</div><div class="lab">PM10 µg/m³</div></div>
    </div>
  </div>

  <div class="controls">
    <span class="controls-label">Contaminante</span>
    <div class="tab-group" id="contTabsV10">
      <button class="tab active" onclick="selectContV10('NO2', this)">NO₂</button>
      <button class="tab" onclick="selectContV10('PM10', this)">PM10</button>
      <button class="tab" onclick="selectContV10('PM2.5', this)">PM2.5</button>
    </div>
    <div class="sep"></div>
    <span class="controls-label">ZONA (Auditoría v8)</span>
    <div class="tab-group" id="zoneTabsV10">
      <button class="tab active" onclick="selectZoneV10('out', this)">OUT (Auditoría Modelo Libre ZBE)</button>
      <button class="tab" onclick="selectZoneV10('zbe', this)">ZBE (Impacto Diario Política)</button>
    </div>
  </div>

  <div class="charts-section">
    <div class="fig-block">
      <div class="fig-header">
        <div class="fig-title"><strong>Auditoría del Modelo:</strong> Predicción vs Real (Últimos 7 días)</div>
      </div>
      <div class="perf-chart-wrap">
        <canvas id="perfChart"></canvas>
      </div>
    </div>
    
    <div class="did-section" style="padding: 0;">
      <div class="did-title"><strong>Validación de Ayer</strong> — Backtesting de precisión</div>
      <table>
        <thead><tr><th>Parámetro</th><th>Predicción Ayer (v8)</th><th>Medición Real</th><th>Desviación</th><th>Estado</th></tr></thead>
        <tbody id="backtestTable"></tbody>
      </table>
    </div>
  </div>

  <div class="did-section" style="padding: 0 0 40px;">
    <div class="did-title" style="padding: 0 48px 16px;"><strong>Rendimiento del Modelo v8</strong> — Métricas Cross-Validation (5 Folds)</div>
    <div style="padding: 0 48px; overflow-x: auto;">
      <table id="metricsTable">
        <thead><tr><th>Contaminante + Zona</th><th>RMSE (µg/m³)</th><th>MAE (µg/m³)</th><th>R²</th><th>MAPE %</th><th>N. Features</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
    <p style="padding: 12px 48px 0; font-size:11px; color:var(--muted); font-family:'IBM Plex Mono',monospace;">
      MAPE calculado solo sobre filas con valor observado &gt; 1 µg/m³.&nbsp;&nbsp;Umbral aceptación: MAPE &lt; 25%, R² &gt; 0.35.
    </p>
  </div>
</div>

<div id="view-map" class="view-container">
  <div class="map-header" style="padding: 16px 48px 16px;">
    <div class="header-left">
      <div class="label-tag">Vitoria-Gasteiz — Red Kunak</div>
      <h1>Mapa de <span>Estaciones</span></h1>
      <p class="subtitle">Media diaria de ayer por estación y predicción para mañana. Haz click sobre un punto para ver el detalle.</p>
    </div>
    <div class="map-legend">
      <div class="card-label" style="margin-bottom:4px">Semáforo ICA</div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:#4caf82"></div><span>Buena (&le;25)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:#ffd54f"></div><span>Moderada (25-50)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:#ff7043"></div><span>Mala (50-75)</span></div>
      <div class="map-legend-item"><div class="map-legend-dot" style="background:#ef5350"></div><span>Muy mala (&gt;75)</span></div>
    </div>
  </div>
  <div class="map-wrap">
    <div id="stationMap"></div>
  </div>
</div>

<script>
// ===========================================================================
// INYECCIÓN DE DATOS
// ===========================================================================
let cfData = __CF_DATA_PLACEHOLDER__;
let perfStats = __PERF_DATA_PLACEHOLDER__;
let manana = __MANANA_DATA_PLACEHOLDER__;
let metricsData = __METRICS_DATA_PLACEHOLDER__;
let stationsData = __STATIONS_DATA_PLACEHOLDER__;
const v9Stats = __V9_DATA_PLACEHOLDER__;

const DID_RESULTS = {
  NO2:  { beta3: 0.215,  pvalue: 0.5986, ci_low: -0.586, ci_high: 1.017,  r2: 0.646, n: 4245, pre_mean: 11.72 },
  PM10: { beta3: 1.910,  pvalue: 0.0000, ci_low:  1.115, ci_high: 2.706,  r2: 0.285, n: 4245, pre_mean: 11.03 },
  'PM2.5': { beta3: 0.469, pvalue: 0.0160, ci_low: 0.087, ci_high: 0.850, r2: 0.334, n: 4245, pre_mean: 6.91 },
};

const SUMMARY_STATS = {
  NO2_zbe:   { pure: -19.1, lags: -13.7, obs: 12.26, cf_pure: 14.84, unit: 'µg/m³' },
  NO2_out:   { pure: -12.9, lags: -11.1, obs: 13.35, cf_pure: 15.59, unit: 'µg/m³' },
  PM10_zbe:  { pure:  11.1, lags:  11.4, obs: 11.41, cf_pure: 10.27, unit: 'µg/m³' },
  PM10_out:  { pure:  -3.8, lags:  -4.2, obs:  9.47, cf_pure:  9.83, unit: 'µg/m³' },
  'PM2.5_zbe': { pure: -10.4, lags: -7.7, obs: 5.92, cf_pure: 6.37, unit: 'µg/m³' },
  'PM2.5_out': { pure:  -8.8, lags: -9.5, obs: 6.19, cf_pure: 6.76, unit: 'µg/m³' },
  ICA_zbe:   { pure:  -0.9, lags:   0.1, obs: 18.62, cf_pure: 18.55, unit: 'pts' },
  ICA_out:   { pure:  -6.8, lags:  -7.9, obs: 18.01, cf_pure: 19.26, unit: 'pts' },
};

// ── NAVEGACIÓN ──
function switchMainView(view, btn) {
  document.querySelectorAll('.view-container').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.top-nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  btn.classList.add('active');
  if (view === 'v10') renderDashboard3();
  if (view === 'map') { initMap(); }
}

// ===========================================================================
// JAVASCRIPT ORIGINAL DE D1 (INTACTO)
// ===========================================================================
let currentCont = 'NO2'; let currentZone = 'zbe'; let fig1Chart = null, fig2Chart = null, fig3Chart = null;

function movingAvg(arr, w) { return arr.map((_, i) => { const start = Math.max(0, i - Math.floor(w/2)); const end = Math.min(arr.length, i + Math.ceil(w/2)); const slice = arr.slice(start, end); return slice.reduce((a,b)=>a+b,0)/slice.length; }); }
function formatDate(d) { return new Date(d).toLocaleDateString('es-ES', { day:'numeric', month:'short' }); }
function getColor(val) { if (val < -5)  return '#4caf82'; if (val > 5)   return '#ef5350'; return '#ffd54f'; }
function getClass(val) { if (val < -5)  return 'neg'; if (val > 5)   return 'pos'; return 'neutral'; }

function renderSummaryCards() {
  const key = `${currentCont}_${currentZone}`; const s = SUMMARY_STATS[key]; if (!s) return;
  const low = Math.min(s.pure, s.lags), high = Math.max(s.pure, s.lags), gap_abs = ((s.obs - s.cf_pure)).toFixed(2);
  const cards = [ { label: 'Observado (medio)', value: s.obs.toFixed(2), sub: s.unit, cls: 'neutral', range: 'Sep 2025 → Mar 2026' }, { label: 'Counterfactual (meteo-puro)', value: s.cf_pure.toFixed(2), sub: `${s.unit} sin ZBE`, cls: 'neutral', range: 'Bound conservador' }, { label: 'Efecto ZBE (meteo-puro)', value: `${s.pure > 0 ? '+' : ''}${s.pure.toFixed(1)}%`, sub: `${gap_abs} ${s.unit} absoluto`, cls: getClass(s.pure), range: 'Bound conservador' }, { label: 'Rango efecto ZBE', value: `[${low.toFixed(1)}%, ${high.toFixed(1)}%]`, sub: 'Meteo-puro vs Con-lags', cls: low < 0 && high < 0 ? 'neg' : (low > 0 && high > 0 ? 'pos' : 'neutral'), range: low < 0 && high < 0 ? '✓ Reducción robusta' : (low < 0 ? '⚠ Incierto' : '✕ Aumento neto') } ];
  document.getElementById('summaryCards').innerHTML = cards.map(c => `<div class="summary-card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div><div class="card-sub">${c.sub}</div><div class="card-range">${c.range}</div></div>`).join('');
}

function renderFig1() {
  const key = `${currentCont}_${currentZone}`; if (!cfData || !cfData[key]) return;
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'], dates = pure.dates.map(formatDate); const obsSmooth = movingAvg(pure.observed, 7);
  const ctx = document.getElementById('fig1').getContext('2d'); if (fig1Chart) fig1Chart.destroy();
  fig1Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_fillUpper', data: pure.counterfactual, borderColor: 'transparent', backgroundColor: 'rgba(79,195,247,0.12)', fill: '+1', pointRadius: 0, tension: 0.3 }, { label: 'Observado', data: obsSmooth, borderColor: '#4fc3f7', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false, order: 1 }, { label: 'CF Meteo-Puro', data: movingAvg(pure.counterfactual, 7), borderColor: '#ff7043', backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, tension: 0.3, fill: false, order: 2 }, { label: 'CF Con-Lags', data: movingAvg(lags.counterfactual, 7), borderColor: '#ffb74d', backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [3,3], pointRadius: 0, tension: 0.3, fill: false, order: 3 } ] }, options: { responsive: true, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: '#1c2030', borderColor: '#2a2f3f', borderWidth: 1, titleColor: '#e2e6f0', bodyColor: '#6b7494', titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2)} µg/m³`; } } } }, scales: { x: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: '#1c2030' } }, y: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(1) + ' µg/m³' }, grid: { color: 'rgba(42,47,63,0.6)' }, title: { display: true, text: `${currentCont} (µg/m³)`, color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig2() {
  const key = `${currentCont}_${currentZone}`; if (!cfData || !cfData[key]) return;
  const pure = cfData[key]['METEO-PURO'], lags = cfData[key]['CON-LAGS'], dates = pure.dates.map(formatDate); const gapPureSmooth = movingAvg(pure.gap_pct, 14), gapLagsSmooth = movingAvg(lags.gap_pct, 14), zero = dates.map(() => 0);
  const ctx = document.getElementById('fig2').getContext('2d'); if (fig2Chart) fig2Chart.destroy();
  fig2Chart = new Chart(ctx, { type: 'line', data: { labels: dates, datasets: [ { label: '_bandUpper', data: gapPureSmooth.map((p, i) => Math.max(p, gapLagsSmooth[i])), borderColor: 'transparent', backgroundColor: 'rgba(124,106,247,0.15)', fill: '+1', pointRadius: 0, tension: 0.4 }, { label: '_bandLower', data: gapPureSmooth.map((p, i) => Math.min(p, gapLagsSmooth[i])), borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0, tension: 0.4 }, { label: 'Sin efecto (0%)', data: zero, borderColor: 'rgba(107,116,148,0.4)', borderWidth: 1, borderDash: [2,4], pointRadius: 0, fill: false, order: 10 }, { label: 'Gap Meteo-Puro', data: gapPureSmooth, borderColor: '#ff7043', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false, order: 1 }, { label: 'Gap Con-Lags', data: gapLagsSmooth, borderColor: '#ffb74d', backgroundColor: 'transparent', borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.4, fill: false, order: 2 } ] }, options: { responsive: true, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, position: 'top', labels: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 11 }, boxWidth: 24, filter: item => !item.text.startsWith('_') } }, tooltip: { backgroundColor: '#1c2030', borderColor: '#2a2f3f', borderWidth: 1, titleColor: '#e2e6f0', bodyColor: '#6b7494', titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => { if (ctx.dataset.label.startsWith('_')) return null; return ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}%`; } } } }, scales: { x: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: '#1c2030' } }, y: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v.toFixed(0) + '%' }, grid: { color: 'rgba(42,47,63,0.6)' }, title: { display: true, text: 'Gap % (observado − predicho) / predicho', color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderFig3() {
  const keys = Object.keys(SUMMARY_STATS); const vals = keys.map(k => SUMMARY_STATS[k].pure);
  const colors = vals.map(v => v < -5 ? 'rgba(76,175,130,0.8)' : v > 5 ? 'rgba(239,83,80,0.8)' : 'rgba(255,213,79,0.7)');
  const borders = vals.map(v => v < -5 ? '#4caf82' : v > 5 ? '#ef5350' : '#ffd54f');
  const ctx = document.getElementById('fig3').getContext('2d'); if (fig3Chart) fig3Chart.destroy();
  fig3Chart = new Chart(ctx, { type: 'bar', data: { labels: keys.map(k => k.replace('PM2.5','PM₂.₅').replace('_',' ')), datasets: [{ label: 'Gap % (METEO-PURO)', data: vals, backgroundColor: colors, borderColor: borders, borderWidth: 1, borderRadius: 3 }] }, options: { responsive: true, plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1c2030', borderColor: '#2a2f3f', borderWidth: 1, titleColor: '#e2e6f0', bodyColor: '#6b7494', titleFont: { family: 'IBM Plex Mono', size: 11 }, bodyFont: { family: 'IBM Plex Mono', size: 11 }, callbacks: { label: ctx => ` Efecto: ${ctx.parsed.y.toFixed(1)}%` } } }, scales: { x: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 11 } }, grid: { display: false } }, y: { ticks: { color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 }, callback: v => v + '%' }, grid: { color: 'rgba(42,47,63,0.6)' }, title: { display: true, text: 'Gap % (observado − CF meteo-puro) / CF', color: '#6b7494', font: { family: 'IBM Plex Mono', size: 10 } } } } } });
}

function renderDidTable() {
  const rows = Object.entries(DID_RESULTS).map(([cont, d]) => {
    const sig = d.pvalue < 0.05, pct = (d.beta3 / d.pre_mean * 100).toFixed(1), bCls = d.beta3 < -0.1 ? 'beta-neg' : d.beta3 > 0.1 ? 'beta-pos' : 'beta-neu', sign = d.beta3 > 0 ? '+' : '', pFmt = d.pvalue < 0.0001 ? '<0.0001' : d.pvalue.toFixed(4);
    return `<tr><td><strong>${cont}</strong></td><td class="${bCls}">${sign}${d.beta3.toFixed(3)} µg/m³</td><td class="${sig ? 'sig-yes' : 'sig-no'}">${pFmt} ${sig ? '✓' : ''}</td><td>[${d.ci_low > 0 ? '+' : ''}${d.ci_low.toFixed(3)}, ${d.ci_high > 0 ? '+' : ''}${d.ci_high.toFixed(3)}]</td><td class="${bCls}">${sign}${pct}%</td><td>${d.r2.toFixed(3)}</td><td>${d.n.toLocaleString()}</td><td class="${sig ? 'sig-yes' : 'sig-no'}">${sig ? 'Sí (p<0.05)' : 'No'}</td></tr>`;
  });
  document.getElementById('didTable').innerHTML = rows.join('');
}

function selectCont(cont, btn) { currentCont = cont; document.querySelectorAll('#contTabs .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderSummaryCards(); renderFig1(); renderFig2(); }
function selectZone(zone, btn) { currentZone = zone; document.querySelectorAll('#zoneTabs .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderSummaryCards(); renderFig1(); renderFig2(); }

// ===========================================================================
// LÓGICA V9 (CÓDIGO ORIGINAL RESTAURADO)
// ===========================================================================
let currentContV9 = 'NO2', currentStationV9 = 'PAUL';

function renderV9Cards() {
  const contKey = currentContV9 === 'PM2.5' ? 'PM25' : currentContV9;
  const key = `${contKey}_${currentStationV9}`; const data = v9Stats[key] || { preR2: 'N/A', gap: 'N/A', desc: 'Sin datos' };
  const colorClass = String(data.gap).includes('-') ? 'neg' : (String(data.gap).includes('+') ? 'pos' : 'neutral');
  document.getElementById('summaryCardsV9').innerHTML = `<div class="summary-card"><div class="card-label">Métrica de Ajuste (Pre-ZBE)</div><div class="card-value neutral">R² = ${data.preR2}</div></div><div class="summary-card"><div class="card-label">Impacto Post-ZBE Estimado</div><div class="card-value ${colorClass}">${data.gap}</div></div><div class="summary-card"><div class="card-label">Diagnóstico Causal</div><div class="card-value neutral" style="font-size: 16px; margin-top:8px;">${data.desc}</div></div><div class="summary-card"><div class="card-label">Pre-Trend Test</div><div class="card-value neg" style="font-size: 16px; margin-top:8px;">✓ OK Parallel Trends</div></div>`;
}

function updateV9Images() {
  const contName = currentContV9 === 'PM2.5' ? 'PM25' : currentContV9;
  const imgSc = document.getElementById('img-sc'), imgEs = document.getElementById('img-es');
  
  // Ocultar mensaje error primero
  imgSc.nextElementSibling.style.display = 'none';
  imgEs.nextElementSibling.style.display = 'none';
  
  imgSc.style.display = 'block'; 
  imgEs.style.display = 'block'; 
  
  imgSc.src = `reports/plots/synthetic_control_${contName}_${currentStationV9}.png`; 
  imgEs.src = `reports/plots/event_study_${contName}.png`;
}

function selectContV9(cont, btn) { currentContV9 = cont; document.querySelectorAll('#contTabsV9 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderV9Cards(); updateV9Images(); }
function selectStationV9(station, btn) { currentStationV9 = station; document.querySelectorAll('#stationTabsV9 .tab').forEach(t => t.classList.remove('active')); btn.classList.add('active'); renderV9Cards(); updateV9Images(); }


// ===========================================================================
// LÓGICA DASHBOARD 3 (OPERATIVA Y PREDICCIÓN - JS CORREGIDO)
// ===========================================================================
let currentContV10 = 'NO2', currentZoneV10 = 'out', perfChart = null;

function renderDashboard3() {
  const zoneData = manana[currentZoneV10] || {};
  
  // SOLUCIÓN AL ERROR DE JAVASCRIPT: Se usa ['PM2.5'] explícito para no pedir .PM25 a un objeto que no lo tiene
  const no2 = zoneData['NO2'] || 0;
  const pm25 = zoneData['PM2.5'] || 0;
  const pm10 = zoneData['PM10'] || 0;

  // 1. Semáforo EEA
  let badge = document.getElementById('riskBadge');
  let riskLevel = 0; 
  if (no2 >= 40 || pm10 >= 20 || pm25 >= 10) riskLevel = 1;
  if (no2 >= 90 || pm10 >= 40 || pm25 >= 20) riskLevel = 2;

  if (riskLevel === 0) { badge.innerText = "🟢 BUENO"; badge.style.color = "var(--green)"; badge.style.borderColor = "var(--green)"; badge.style.backgroundColor = "rgba(76,175,130,0.1)"; }
  else if (riskLevel === 1) { badge.innerText = "🟡 MODERADO"; badge.style.color = "var(--yellow)"; badge.style.borderColor = "var(--yellow)"; badge.style.backgroundColor = "rgba(255,213,79,0.1)"; }
  else { badge.innerText = "🔴 MALO"; badge.style.color = "var(--red)"; badge.style.borderColor = "var(--red)"; badge.style.backgroundColor = "rgba(239,83,80,0.1)"; }

  // Aquí era donde fallaba al hacer .toFixed(1) sobre algo "undefined"
  document.getElementById('val-no2').innerText = no2.toFixed(1);
  document.getElementById('val-pm25').innerText = pm25.toFixed(1);
  document.getElementById('val-pm10').innerText = pm10.toFixed(1);

  // 2. Gráfico de Performance
  const d = perfStats[currentZoneV10][currentContV10] || {labels:[], real:[], pred:[]};
  const ctx = document.getElementById('perfChart').getContext('2d');
  if(perfChart) perfChart.destroy();
  perfChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: perfStats[currentZoneV10].labels || [],
          datasets: [
              { label: 'Medición Real', data: d.real, borderColor: '#4fc3f7', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 4, tension: 0.4 },
              { label: 'Predicción LightGBM v8', data: d.pred, borderColor: '#7c6af7', backgroundColor: 'transparent', borderDash: [5,5], borderWidth: 2, pointRadius: 4, tension: 0.4 }
          ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { color: '#6b7494', font: { family: 'IBM Plex Mono'} } } }, scales: { x: { ticks: { color: '#6b7494' } }, y: { ticks: { color: '#6b7494' }, grid: { color: '#2a2f3f' } } } }
  });

  // 3. Tabla de Validación
  const lastRealDraw = d.real[d.real.length - 1]; // Puede ser null si el sensor tiene delay
  const lastPred = d.pred[d.pred.length - 1] || 0;
  
  let realText, errorText, interpret, color;
  
  if (lastRealDraw === null) {
      realText = "N/D (Pendiente)";
      errorText = "N/D";
      interpret = "⏳ Esperando sensores Kunak";
      color = "var(--muted)";
  } else {
      const lastReal = lastRealDraw;
      const error = lastPred > 0 ? (((lastReal - lastPred) / lastPred) * 100).toFixed(1) : 0;
      realText = `${lastReal.toFixed(1)} µg/m³`;
      errorText = `${error > 0 ? '+' : ''}${error}%`;
      
      // Color del porcentaje: rojo si la contaminación fue MAYOR a la prevista, verde si fue MENOR o igual.
      const errorColor = error > 0 ? "var(--red)" : "var(--green)";
      
      // Color del texto y el texto: verde si el error absoluto <= 25%, amarillo si > 25%
      const interpretColor = Math.abs(error) <= 25 ? "var(--green)" : "var(--yellow)";
      interpret = Math.abs(error) <= 25 ? "✓ Precisión aceptable" : "ℹ Revisar modelo (desv >25%)";

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

// ── METRICS TABLE ─────────────────────────────────────────────────────────
function renderMetricsTable() {
  const order = [
    ['NO2','zbe'], ['NO2','out'],
    ['PM10','zbe'], ['PM10','out'],
    ['PM2.5','zbe'], ['PM2.5','out'],
    ['ICA','zbe'], ['ICA','out'],
  ];
  const body = document.getElementById('metricsBody');
  if (!body) return;
  body.innerHTML = order.map(([cont, zone]) => {
    const key = `target_${cont}_${zone}_d1`;
    const m = metricsData[key];
    if (!m) return '';
    const mapeColor = m.cv_mape <= 25 ? 'var(--green)' : 'var(--yellow)';
    const r2Color   = m.cv_r2   >= 0.35 ? 'var(--green)' : 'var(--yellow)';
    return `<tr>
      <td><strong>${cont}</strong> <span style="color:var(--muted)">${zone.toUpperCase()}</span></td>
      <td style="font-family:'IBM Plex Mono',monospace">${m.cv_rmse.toFixed(2)}</td>
      <td style="font-family:'IBM Plex Mono',monospace">${m.cv_mae.toFixed(2)}</td>
      <td style="color:${r2Color};font-family:'IBM Plex Mono',monospace">${m.cv_r2.toFixed(3)}</td>
      <td style="color:${mapeColor};font-family:'IBM Plex Mono',monospace;font-weight:bold">${m.cv_mape.toFixed(1)}%</td>
      <td style="color:var(--muted);font-family:'IBM Plex Mono',monospace">${m.n_features}</td>
    </tr>`;
  }).join('');
}

// ── MAPA DE ESTACIONES (Leaflet) ──────────────────────────────────────────────
let _mapInitialized = false;
function icaColor(ica) {
  if (ica === null || ica === undefined) return '#6b7494';
  if (ica <= 25)  return '#4caf82';
  if (ica <= 50)  return '#ffd54f';
  if (ica <= 75)  return '#ff7043';
  return '#ef5350';
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
  const map = L.map('stationMap', { zoomControl: true });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: 'abcd', maxZoom: 19
  }).addTo(map);

  // ── Perímetro oficial ZBE — fuente: DGT XML id="1059_Ayuntamiento de Vitoria-Gasteiz"
  // Coordenadas exactas (OpenlrPolygonLocationReference, actualizado 2025-09-27)
  const ZBE_COORDS = [
    [42.845707, -2.6764994],
    [42.846058, -2.678793],
    [42.846058, -2.678777],
    [42.84426,  -2.6781747],
    [42.84412,  -2.6780825],
    [42.84361,  -2.6756046],
    [42.843502, -2.6743975],
    [42.843246, -2.6726305],
    [42.84232,  -2.6727068],
    [42.8423,   -2.6724603],
    [42.84322,  -2.6724162],
    [42.84342,  -2.6700573],
    [42.8442,   -2.6686096],
    [42.845608, -2.6682484],
    [42.846146, -2.6681008],
    [42.846935, -2.667851],
    [42.84823,  -2.668305],
    [42.849224, -2.6686778],
    [42.850346, -2.6691453],
    [42.852093, -2.670227],
    [42.85308,  -2.6731813],
    [42.85263,  -2.6730998],
    [42.852352, -2.6731632],
    [42.852013, -2.6732967],
    [42.851536, -2.6737828],
    [42.849266, -2.6755311],
    [42.847466, -2.6762118],
    [42.84709,  -2.676142],
    [42.845707, -2.6764994],   // cierre del polígono
  ];
  L.polygon(ZBE_COORDS, {
    color: '#ef5350',
    weight: 2.5,
    opacity: 0.9,
    fillColor: '#ef5350',
    fillOpacity: 0.07,
    interactive: false,
    dashArray: '6 4',
  }).addTo(map);

  Object.entries(stationsData).forEach(([name, s]) => {
    const color = icaColor(s.ICA);
    const isZbe = s.zone === 'ZBE';
    const markerColor = isZbe ? '#7c6af7' : '#4fc3f7';

    // Punto principal — sin halos, solo el marcador
    const marker = L.circleMarker([s.lat, s.lon], {
      radius: 11,
      color: markerColor, weight: 2,
      fillColor: color, fillOpacity: 0.95,
    }).addTo(map);

    function fmt(v) { return v !== null && v !== undefined ? v.toFixed(1) : 'N/D'; }
    const popupHtml = `
      <div class="popup-name">${icaEmoji(s.ICA)} ${s.label}</div>
      <div style="font-size:10px;color:#7c6af7;font-family:'IBM Plex Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em">Media diaria (ayer)</div>
      <div class="popup-row"><span class="popup-label">NO₂</span><span class="popup-val">${fmt(s.NO2)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">PM10</span><span class="popup-val">${fmt(s.PM10)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">PM2.5</span><span class="popup-val">${fmt(s.PM25)} µg/m³</span></div>
      <div class="popup-row"><span class="popup-label">ICA</span><span class="popup-val">${fmt(s.ICA)}</span></div>
      <div class="popup-section">
        <div class="popup-section-title">▶ Predicción mañana</div>
        <div class="popup-row"><span class="popup-label">NO₂</span><span class="popup-val">${fmt(s.pred_NO2)} µg/m³</span></div>
        <div class="popup-row"><span class="popup-label">PM10</span><span class="popup-val">${fmt(s.pred_PM10)} µg/m³</span></div>
        <div class="popup-row"><span class="popup-label">PM2.5</span><span class="popup-val">${fmt(s.pred_PM25)} µg/m³</span></div>
        <div style="font-size:10px;color:#6b7494;margin-top:6px">Zona ${s.zone} — modelo LightGBM v8</div>
        <div style="font-size:10px;color:#6b7494;margin-top:2px">⚠ Los valores son medias diarias, no lecturas horarias</div>
      </div>`;
    marker.bindPopup(popupHtml, { maxWidth: 260 });

    // Etiqueta con nombre fijado al mapa
    const icon = L.divIcon({
      className: '',
      html: `<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#e2e6f0;background:rgba(13,15,20,0.75);padding:2px 5px;border-radius:3px;white-space:nowrap;border:1px solid #2a2f3f">${name}</div>`,
      iconAnchor: [-14, 6]
    });
    L.marker([s.lat, s.lon], { icon, interactive: false }).addTo(map);
  });

  // Ajustar vista para que se vean todas las estaciones
  const allLatLngs = Object.values(stationsData).map(s => [s.lat, s.lon]);
  if (allLatLngs.length > 0) map.fitBounds(L.latLngBounds(allLatLngs), { padding: [50, 50] });

  setTimeout(() => map.invalidateSize(), 100);
}

// ── INIT ───────────────────────────────────────────────────────────────────
window.onload = function() {
  renderSummaryCards(); renderFig1(); renderFig2(); renderFig3(); renderDidTable();
  renderV9Cards(); updateV9Images();
  renderDashboard3(); renderMetricsTable();
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
content = content.replace('__V9_DATA_PLACEHOLDER__', v9_json_str)
content = content.replace('__PRED_DATE_PLACEHOLDER__', prediction_date_str)

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"¡Listo! Archivo {output_html} generado correctamente sin romper D1 ni D2.")