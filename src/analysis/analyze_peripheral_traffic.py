import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuración
ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
TRAFFIC_DIR = ROOT_DIR / "data/raw/traffic"
REPORTS_DIR = ROOT_DIR / "reports"
ZBE_DATE = "2025-09-01"

# Sensores proporcionados por el usuario (Perímetro ZBE)
PERIPHERAL_SENSORS = [
    "02164", "02197", "03012", "03014", "03016", "03241", "03017", "03242",
    "03125", "03120", "03024", "03027", "03023", "03025", "02036", "02158",
    "02148", "02159", "02042"
]

def analyze():
    print(f"Análisis de Tráfico Perimetral ZBE")
    print(f"====================================")
    
    all_data = []
    years = [2024, 2025, 2026]
    
    for year in years:
        file_path = TRAFFIC_DIR / f"trafico_{year}.csv"
        if not file_path.exists():
            print(f"  [WARN] No se encontró {file_path}")
            continue
            
        print(f"  Cargando datos de {year}...")
        # Leemos solo columnas necesarias para ahorrar memoria
        df = pd.read_csv(file_path, usecols=["code", "start_date", "volume"])
        
        # Filtrar por sensores del perímetro
        df["code"] = df["code"].astype(str).str.zfill(5)
        df = df[df["code"].isin(PERIPHERAL_SENSORS)]
        
        if not df.empty:
            df["start_date"] = pd.to_datetime(df["start_date"])
            all_data.append(df)
            
    if not all_data:
        print("No hay datos para analizar.")
        return
        
    df_all = pd.concat(all_data)
    
    # Agregación diaria por sensor
    df_daily = df_all.groupby(["code", df_all["start_date"].dt.date])["volume"].sum().reset_index()
    df_daily["start_date"] = pd.to_datetime(df_daily["start_date"])
    
    # Marcamos periodos
    zbe_ts = pd.Timestamp(ZBE_DATE)
    df_daily["period"] = df_daily["start_date"].apply(lambda x: "Post-ZBE" if x >= zbe_ts else "Pre-ZBE")
    
    # ── 1. ESTADÍSTICAS POR SENSOR ───────────────────────────────────────────
    stats = df_daily.groupby(["code", "period"])["volume"].mean().unstack()
    stats["diff_abs"] = stats["Post-ZBE"] - stats["Pre-ZBE"]
    stats["diff_pct"] = (stats["diff_abs"] / stats["Pre-ZBE"]) * 100
    
    print("\nImpacto por Sensor (Tráfico Medio Diario):")
    print(stats[["Pre-ZBE", "Post-ZBE", "diff_pct"]].sort_values("diff_pct", ascending=False))
    
    # Guardar CSV
    stats.to_csv(REPORTS_DIR / "peripheral_traffic_stats.csv")
    print(f"\n[OK] Estadísticas guardadas en reports/peripheral_traffic_stats.csv")
    
    # ── 2. TENDENCIA DEL GRUPO ───────────────────────────────────────────────
    group_daily = df_daily.groupby("start_date")["volume"].mean() # Media de los sensores cada día
    group_roll = group_daily.rolling(window=7, center=True).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(group_roll.index, group_roll.values, color="#2196F3", linewidth=2, label="Tráfico Medio (7d roll)")
    plt.axvline(zbe_ts, color="red", linestyle="--", alpha=0.7, label="Implementación ZBE")
    
    # Añadir flecha de cambio
    pre_avg = group_daily[group_daily.index < zbe_ts].mean()
    post_avg = group_daily[group_daily.index >= zbe_ts].mean()
    change = ((post_avg - pre_avg) / pre_avg) * 100
    
    plt.title(f"Evolución Tráfico en Perímetro ZBE\nImpacto Medio: {change:+.1f}%", fontsize=14, fontweight='bold')
    plt.ylabel("Vehículos/Día (Media del grupo)")
    plt.xlabel("Fecha")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Guardar gráfico
    output_png = REPORTS_DIR / "plots/peripheral_traffic_impact.png"
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Gráfico guardado en {output_png}")
    
    print(f"\nRESUMEN GLOBAL:")
    print(f"-------------------------------")
    print(f"Tráfico Medio Pre-ZBE:  {pre_avg:.0f} veh/día")
    print(f"Tráfico Medio Post-ZBE: {post_avg:.0f} veh/día")
    print(f"Variación:              {change:+.2f}%")

if __name__ == "__main__":
    analyze()
