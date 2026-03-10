# ver_outliers.py — ejecuta esto para ver los días concretos
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "features_daily.parquet")

contaminants = ["NO2_zbe", "NO2_out", "PM10_zbe", "PM10_out",
                "PM2.5_zbe", "PM2.5_out", "ICA_zbe", "ICA_out"]

for col in contaminants:
    s = df[col]
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    mask = (s < Q1 - 3*IQR) | (s > Q3 + 3*IQR)
    if mask.sum() > 0:
        print(f"\n── {col} (Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}) ──")
        print(df.loc[mask, ["date", col]].to_string(index=False))