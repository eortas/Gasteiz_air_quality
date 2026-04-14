import joblib
import pandas as pd
from pathlib import Path
import json

target = "NO2_zbe_d1"
model = joblib.load(f"lgbm_v8_{target}.pkl")
features = json.loads(Path(f"lgbm_v8_{target}_features.json").read_text())

# Load latest row
df = pd.read_parquet("../data/processed/features_daily.parquet")
X = df.iloc[[-1]].reindex(columns=features, fill_value=0).fillna(0)

contribs = model.predict(X, pred_contrib=True)
print("Contribs shape:", contribs.shape)
print("Base value:", contribs[0, -1])

features_and_contribs = list(zip(features, contribs[0, :-1]))
features_and_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
for f, c in features_and_contribs[:10]:
    print(f"{f}: {c}")
