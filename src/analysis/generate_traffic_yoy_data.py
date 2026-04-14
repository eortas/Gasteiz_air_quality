import pandas as pd
import json
from pathlib import Path

# Configuración
ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
TRAFFIC_DIR = ROOT_DIR / "data/raw/traffic"
REPORTS_DIR = ROOT_DIR / "reports"
ZBE_DATE = "2025-09-01"

# Sensores del perímetro ZBE
PERIPHERAL_SENSORS = [
    "02164", "02197", "03012", "03014", "03016", "03241", "03017", "03242",
    "03125", "03120", "03024", "03027", "03023", "03025", "02036", "02158",
    "02148", "02159", "02042"
]

def generate_yoy():
    print("Iniciando generación de datos YoY de tráfico perimetral...")
    all_data = []
    
    for year in [2024, 2025, 2026]:
        file_path = TRAFFIC_DIR / f"trafico_{year}.csv"
        if not file_path.exists():
            continue
            
        print(f"  Procesando {year}...")
        # Carga optimizada
        df = pd.read_csv(file_path, usecols=["code", "start_date", "volume"])
        df["code"] = df["code"].astype(str).str.zfill(5)
        df = df[df["code"].isin(PERIPHERAL_SENSORS)]
        
        if not df.empty:
            df["start_date"] = pd.to_datetime(df["start_date"])
            all_data.append(df)
            
    if not all_data:
        print("[ERROR] No se encontraron datos de tráfico.")
        return

    df_full = pd.concat(all_data)
    
    # 1. Suma diaria por sensor (Vehículos/Día)
    df_daily_sensor = df_full.groupby(["code", df_full["start_date"].dt.date])["volume"].sum().reset_index()
    df_daily_sensor.columns = ["code", "date", "volume"]
    df_daily_sensor["date"] = pd.to_datetime(df_daily_sensor["date"])
    
    # 2. Media del grupo perimetral por día
    daily_avg = df_daily_sensor.groupby("date")["volume"].mean().reset_index()
    daily_avg = daily_avg.sort_values("date").set_index("date")
    
    # Periodo de análisis: desde implantación ZBE
    zbe_ts = pd.Timestamp(ZBE_DATE)
    post_zbe = daily_avg[daily_avg.index >= zbe_ts].copy()
    
    results = []
    for dt in post_zbe.index:
        # Alineación por día de semana (-52 semanas = -364 días)
        prev_dt = dt - pd.Timedelta(days=364)
        
        current_val = post_zbe.loc[dt, "volume"]
        prev_val = daily_avg.loc[prev_dt, "volume"] if prev_dt in daily_avg.index else None
        
        results.append({
            "date": dt.strftime("%Y-%m-%d"),
            "current": round(float(current_val), 1),
            "previous": round(float(prev_val), 1) if prev_val is not None else None
        })
        
    output_path = REPORTS_DIR / "peripheral_traffic_yoy.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"[OK] Datos YoY guardados en {output_path}")
    print(f"     Total días procesados: {len(results)}")

if __name__ == "__main__":
    generate_yoy()
