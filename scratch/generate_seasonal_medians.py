"""
Genera los archivos de medianas estacionales (por mes) a partir del dataset existente.
Esto permite que predict.py use medianas del mes actual para imputar NaN
sin necesidad de re-entrenar los modelos completos.
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(r"c:\Users\ortas\OneDrive\Documentos\Vitoria_AG")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TARGETS = [
    "NO2_zbe_d1", "NO2_out_d1",
    "PM10_zbe_d1", "PM10_out_d1",
    "PM2.5_zbe_d1", "PM2.5_out_d1",
]

print("Generando medianas estacionales a partir del dataset...")

# Cargar el dataset de features
df = pd.read_parquet(PROCESSED / "features_daily.parquet")
df["date"] = pd.to_datetime(df["date"], utc=True)
df["_month"] = df["date"].dt.month
print(f"Dataset: {len(df)} filas, rango: {df['date'].min().date()} -> {df['date'].max().date()}")

for target in TARGETS:
    feat_path = MODELS / f"lgbm_v8_{target}_features.json"
    if not feat_path.exists():
        print(f"  {target}: sin archivo de features, saltando")
        continue
    
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    
    # Calcular medianas por mes
    seasonal_medians = {}
    for month in range(1, 13):
        month_data = df[df["_month"] == month]
        if len(month_data) >= 5:
            available_feats = [f for f in features if f in month_data.columns]
            month_meds = month_data[available_feats].median().fillna(0)
            seasonal_medians[str(month)] = {
                feat: float(month_meds[feat])
                for feat in available_feats if feat in month_meds.index
            }
    
    out_path = MODELS / f"lgbm_v8_{target}_medians_seasonal.json"
    with open(out_path, "w") as f:
        json.dump(seasonal_medians, f)
    
    print(f"  {target}: {len(seasonal_medians)} meses guardados -> {out_path.name}")
    
    # Mostrar diferencia de medianas clave para NO2
    if "NO2" in target:
        global_path = MODELS / f"lgbm_v8_{target}_medians.json"
        if global_path.exists():
            global_meds = json.loads(global_path.read_text(encoding="utf-8"))
            key_feats_check = ["NO2_zbe", "NO2_out", "NO2_zbe_roll_mean_7d", "HDD", "HDD_acum_7d"]
            print(f"    Comparación medianas clave (global vs junio):")
            for kf in key_feats_check:
                if kf in global_meds and "6" in seasonal_medians:
                    g = global_meds[kf]
                    s = seasonal_medians["6"].get(kf, "N/A")
                    if isinstance(s, (int, float)):
                        print(f"      {kf:40s} global={g:.2f}  junio={s:.2f}  ratio={s/g:.2f}")

print("\n[OK] Medianas estacionales generadas.")
