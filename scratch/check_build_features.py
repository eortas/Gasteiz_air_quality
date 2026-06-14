import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path("c:/Users/ortas/OneDrive/Documentos/Vitoria_AG")
sys.path.append(str(ROOT_DIR / "src" / "features"))

# We can just load the intermediate files or run parts of build_features_v6.py
import build_features_v6 as bf

air = bf.load_air_daily()
traffic = bf.load_traffic_daily()
weather = bf.load_weather_daily()

df = bf.merge_daily(air, traffic, weather)
df = bf.add_temporal_features(df)
df = bf.add_lags_and_rolling(df)
df = bf.add_targets(df)

lag_cols = [c for c in df.columns if "_lag_" in c]
print("\nFirst 4 lag columns: ", lag_cols[:4])
print("\nLast 5 rows of lag_cols[:4]:")
print(df[["date"] + lag_cols[:4]].tail(5))

# Simulate dropna target and dropna lag
df_target_dropped = df.dropna(subset=target_cols, how="all")
print(f"\nAfter target dropna, last date is: {df_target_dropped['date'].max()}")

df_lag_dropped = df_target_dropped.dropna(subset=lag_cols[:4], how="any")
print(f"After lag dropna, last date is: {df_lag_dropped['date'].max()}")

