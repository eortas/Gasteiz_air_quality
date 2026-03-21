import pandas as pd
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent.parent
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"
OUTPUT_HTML = ROOT_DIR / "reports" / "html" / "traffic_map.html"

def build_traffic_dashboard():
    print("Cargando sensores...")
    sensors_df = pd.read_csv(TRAFFIC_DIR / "sensors.csv", dtype={"code": str})
    
    print("Cargando datos de tráfico 2026...")
    traffic_df = pd.read_csv(TRAFFIC_DIR / "trafico_2026.csv", dtype={"code": str})
    traffic_df["start_date"] = pd.to_datetime(traffic_df["start_date"], utc=True)
    traffic_df["date"] = traffic_df["start_date"].dt.date
    
    # Obtener los dos últimos días con datos
    available_dates = sorted(traffic_df["date"].unique())
    if len(available_dates) < 2:
        print("Advertencia: Menos de 2 días de datos disponibles.")
        today_date = available_dates[-1]
        yesterday_date = today_date # Fallback
    else:
        today_date = available_dates[-1]
        yesterday_date = available_dates[-2]
    
    print(f"Procesando datos para {today_date} (Hoy) y {yesterday_date} (Ayer)...")
    
    def get_daily_stats(date):
        day_df = traffic_df[traffic_df["date"] == date]
        if day_df.empty:
            return pd.DataFrame()
            
        stats = day_df.groupby("code").agg({
            "volume": "sum",
            "occupancy": "mean",
            "load": "mean"
        }).reset_index()
        
        # Encontrar el tramo con carga máxima para cada sensor
        peak_info = []
        for code, group in day_df.groupby("code"):
            if group["load"].isna().all():
                peak_info.append({"code": code, "peak_h": None, "peak_l": None, "peak_v": None})
                continue
                
            peak_row = group.loc[group["load"].idxmax()]
            peak_info.append({
                "code": code,
                "peak_h": peak_row["start_date"].strftime("%H:%M"),
                "peak_l": round(float(peak_row["load"]), 1),
                "peak_v": int(peak_row["volume"])
            })
            
        peak_df = pd.DataFrame(peak_info)
        return pd.merge(stats, peak_df, on="code", how="left")

    stats_today = get_daily_stats(today_date)
    stats_yesterday = get_daily_stats(yesterday_date)
    
    # Combinar con sensores
    sensors_list = []
    for _, sensor in sensors_df.iterrows():
        code = sensor["code"]
        
        s_today = stats_today[stats_today["code"] == code]
        s_yesterday = stats_yesterday[stats_yesterday["code"] == code]
        
        if s_today.empty and s_yesterday.empty:
            continue
            
        def get_val(df, col):
            val = df[col].iloc[0] if not df.empty else None
            return round(float(val), 2) if val is not None else None

        v_t, o_t, l_t = get_val(s_today, "volume"), get_val(s_today, "occupancy"), get_val(s_today, "load")
        v_y, o_y, l_y = get_val(s_yesterday, "volume"), get_val(s_yesterday, "occupancy"), get_val(s_yesterday, "load")
        
        peak_t = {
            "h": s_today["peak_h"].iloc[0] if not s_today.empty else None,
            "l": s_today["peak_l"].iloc[0] if not s_today.empty else None,
            "v": int(s_today["peak_v"].iloc[0]) if not s_today.empty and not pd.isna(s_today["peak_v"].iloc[0]) else None
        }
        peak_y = {
            "h": s_yesterday["peak_h"].iloc[0] if not s_yesterday.empty else None,
            "l": s_yesterday["peak_l"].iloc[0] if not s_yesterday.empty else None,
            "v": int(s_yesterday["peak_v"].iloc[0]) if not s_yesterday.empty and not pd.isna(s_yesterday["peak_v"].iloc[0]) else None
        }

        # Filtrar si todos los valores son 0 en ambos días
        if (v_t == 0 or v_t is None) and (o_t == 0 or o_t is None) and (l_t == 0 or l_t is None) and \
           (v_y == 0 or v_y is None) and (o_y == 0 or o_y is None) and (l_y == 0 or l_y is None):
            continue

        sensors_list.append({
            "code": code,
            "name": sensor["name"],
            "lat": sensor["lat"],
            "lon": sensor["lon"],
            "today": {"v": v_t, "o": o_t, "l": l_t, "peak": peak_t},
            "yesterday": {"v": v_y, "o": o_y, "l": l_y, "peak": peak_y}
        })

    # HTML Template
    html_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tráfico Vitoria-Gasteiz — Mapa Sensores</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f4f6f8;
            --surface: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --accent: #4f46e5;
            --green: #059669;
            --yellow: #d97706;
            --red: #dc2626;
            --border: #dcdfe4;
        }
        body { margin: 0; font-family: 'IBM Plex Sans', sans-serif; background: var(--bg); color: var(--text); }
        .header { background: #0055A4; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .header h1 { margin: 0; font-size: 18px; font-weight: 700; }
        .controls { padding: 12px 24px; background: white; border-bottom: 1px solid var(--border); display: flex; gap: 12px; align-items: center; }
        .btn-toggle { padding: 6px 16px; border-radius: 4px; border: 1px solid var(--border); background: white; cursor: pointer; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
        .btn-toggle.active { background: var(--accent); color: white; border-color: var(--accent); }
        #map { height: calc(100vh - 110px); width: 100%; }
        .leaflet-popup-content-wrapper { border-radius: 8px; }
        .popup-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; display: block; }
        .popup-row { display: flex; justify-content: space-between; gap: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; margin-bottom: 2px; }
        .legend { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.4); line-height: 1.5; font-size: 12px; }
        .legend i { width: 12px; height: 12px; float: left; margin-right: 8px; opacity: 0.7; border-radius: 50%; border: 1px solid #999; }
    </style>
</head>
<body>
    <div class="header">
        <h1 data-i18n="title">Vitoria-Gasteiz — Mapa de Sensores de Tráfico</h1>
        <div style="font-size: 12px; font-family: 'IBM Plex Mono', monospace;" data-i18n="subtitle">Real-time-ish feed</div>
    </div>
    <div class="controls">
        <span style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--muted);" data-i18n="viewLabel">Ver datos de:</span>
        <button id="btn-yesterday" class="btn-toggle" onclick="setView('yesterday')"><span data-i18n="dayPrefix">Día </span>__YESTERDAY__</button>
        <button id="btn-today" class="btn-toggle active" onclick="setView('today')"><span data-i18n="dayPrefix">Día </span>__TODAY__</button>
    </div>
    <div id="map"></div>

    <script>
        const sensors = __SENSORS_DATA__;
        let currentView = 'today';
        
        const urlParams = new URLSearchParams(window.location.search);
        let currentLang = urlParams.get('lang') || 'es';

        const translations = {
          es: {
            title: "Vitoria-Gasteiz — Mapa de Sensores de Tráfico",
            subtitle: "Real-time-ish feed",
            viewLabel: "Ver datos de:",
            dayPrefix: "Día ",
            carga: "Carga:",
            ocupacion: "Ocupación:",
            volumen: "Volumen total:",
            picoTitle: "Pico de Carga (Tramo Máx)",
            hora: "Hora:",
            cargaPico: "Carga pico:",
            volumenPico: "Volumen pico:",
            legendTitle: "Carga Tráfico (%)",
            legendFluido: "< 15% (Fluido)",
            legendMod: "15-30% (Moderado)",
            legendDenso: "> 30% (Denso)",
            legendNoData: "Sin datos"
          },
          eu: {
            title: "Gasteiz — Trafiko Sentsoreen Mapa",
            subtitle: "Real-time-ish feed",
            viewLabel: "Ikusi datuak:",
            dayPrefix: "Eguna ",
            carga: "Karga:",
            ocupacion: "Okupazioa:",
            volumen: "Bolumen osoa:",
            picoTitle: "Karga Gailurra (Gehienezko Tartea)",
            hora: "Ordua:",
            cargaPico: "Karga gailurra:",
            volumenPico: "Bolumen gailurra:",
            legendTitle: "Trafiko Karga (%)",
            legendFluido: "< %15 (Arina)",
            legendMod: "%15-30 (Ertaina)",
            legendDenso: "> %30 (Densea)",
            legendNoData: "Daturik gabe"
          }
        };

        function updateI18n() {
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                if (translations[currentLang][key]) {
                    el.innerText = translations[currentLang][key];
                }
            });
            if (legendInstance) {
                legendInstance.remove();
                legendInstance.addTo(map);
            }
        }

        let map = L.map('map').setView([42.8467, -2.6716], 14);
        
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }).addTo(map);

        let markers = L.layerGroup().addTo(map);

        function getColor(load) {
            if (load === null) return '#999';
            return load > 30 ? 'var(--red)' :
                   load > 15 ? 'var(--yellow)' :
                               'var(--green)';
        }

        function updateMarkers() {
            markers.clearLayers();
            const t = translations[currentLang];
            sensors.forEach(s => {
                const data = s[currentView];
                const color = getColor(data.l);
                
                const marker = L.circleMarker([s.lat, s.lon], {
                    radius: 8,
                    fillColor: color,
                    color: "#fff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                });

                const popupContent = `
                    <div class="popup-title">${s.name} (${s.code})</div>
                    <div class="popup-row"><span>${t.carga}</span> <b>${data.l !== null ? data.l + '%' : 'N/A'}</b></div>
                    <div class="popup-row"><span>${t.ocupacion}</span> <b>${data.o !== null ? data.o + '%' : 'N/A'}</b></div>
                    <div class="popup-row"><span>${t.volumen}</span> <b>${data.v !== null ? data.v : 'N/A'}</b></div>
                    
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border);">
                        <div style="font-size: 10px; color: var(--accent); font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">${t.picoTitle}</div>
                        <div class="popup-row"><span>${t.hora}</span> <b>${data.peak.h || 'N/A'}</b></div>
                        <div class="popup-row"><span>${t.cargaPico}</span> <b>${data.peak.l !== null ? data.peak.l + '%' : 'N/A'}</b></div>
                        <div class="popup-row"><span>${t.volumenPico}</span> <b>${data.peak.v || 'N/A'}</b></div>
                    </div>
                `;
                
                marker.bindPopup(popupContent);
                marker.addTo(markers);
            });
        }

        function setView(view) {
            currentView = view;
            document.getElementById('btn-today').classList.toggle('active', view === 'today');
            document.getElementById('btn-yesterday').classList.toggle('active', view === 'yesterday');
            updateMarkers();
        }

        let legendInstance = L.control({position: 'bottomright'});
        legendInstance.onAdd = function (map) {
            const t = translations[currentLang];
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML += `<b>${t.legendTitle}</b><br>`;
            div.innerHTML += `<i style="background: var(--green)"></i> ${t.legendFluido}<br>`;
            div.innerHTML += `<i style="background: var(--yellow)"></i> ${t.legendMod}<br>`;
            div.innerHTML += `<i style="background: var(--red)"></i> ${t.legendDenso}<br>`;
            div.innerHTML += `<i style="background: #999"></i> ${t.legendNoData}`;
            return div;
        };
        legendInstance.addTo(map);

        // Inicialización
        updateI18n();
        updateMarkers();
    </script>
</body>
</html>"""

    # Reemplazar placeholders
    final_html = html_template.replace("__SENSORS_DATA__", json.dumps(sensors_list))
    final_html = final_html.replace("__TODAY__", str(today_date))
    final_html = final_html.replace("__YESTERDAY__", str(yesterday_date))
    
    OUTPUT_HTML.write_text(final_html, encoding="utf-8")
    print(f"Dashboard generado con éxito en: {OUTPUT_HTML}")

if __name__ == "__main__":
    build_traffic_dashboard()
