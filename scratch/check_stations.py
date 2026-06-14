import pandas as pd
from pathlib import Path

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
AIR_DIR = ROOT_DIR / "data" / "raw" / "air"

df = pd.read_csv(AIR_DIR / "kunak_2026.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("Unique stations in kunak_2026.csv:")
print(df["estacion"].unique())
print("Date range in kunak_2026.csv:")
print(df["timestamp"].min(), "to", df["timestamp"].max())
