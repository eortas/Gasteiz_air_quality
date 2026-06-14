import pandas as pd
from pathlib import Path

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
AIR_DIR = ROOT_DIR / "data" / "raw" / "air"

df = pd.read_csv(AIR_DIR / "kunak_2026.csv")
print(df.tail(20))
