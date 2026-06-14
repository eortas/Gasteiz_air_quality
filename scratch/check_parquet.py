import pandas as pd
from pathlib import Path

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_parquet(PROCESSED_DIR / "features_daily.parquet")
df["date"] = pd.to_datetime(df["date"], utc=True)
sub = df[(df["date"] >= "2026-06-01") & (df["date"] <= "2026-06-13")]

print("--- DATES WITH NULL TRAFFIC VOLUME ---")
print(sub[sub["traffic_volume"].isnull()]["date"])





