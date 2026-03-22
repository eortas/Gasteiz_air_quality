"""
train_model_v8.py
==================
Entrena 4 modelos LightGBM especializados (NO2, PM10, PM2.5, ICA - solo d1)

CAMBIOS v8 respecto a v7 (nivel paper):

  (7) Efectos fijos temporales C(month)?C(year) en el DiD:
     - Controla shocks comunes del per?odo (anticiclones, intrusi?n sahariana,
       episodios regionales de contaminaci?n compartidos por ZBE y OUT)
     - Granularidad mes?a?o: 22 dummies para el rango mar-2024 -> dic-2025
     - M?s riguroso que C(month) solo porque distingue el mismo mes en a?os distintos

  (8) Clustering de errores por fecha en el DiD:
     - Sustituye HC3 por cov_type="cluster", groups=date
     - Las observaciones del mismo d?a (ZBE y OUT) no son independientes
     - P-valores m?s conservadores y honestos -> m?s cre?bles en revisi?n

  (9) Counterfactual dual (bound inferior / bound superior):
     - Versi?n METEO-PURO: solo meteorolog?a + calendario, sin lags del
       contaminante. Es el bound conservador: efecto m?nimo garantizado.
     - Versi?n CON-LAGS: incluye lags pre-ZBE del contaminante. Es el bound
       superior: efecto m?ximo posible (puede absorber parte del efecto ZBE
       si los lags tienen memoria del per?odo post).
     - El efecto real de la ZBE est? entre ambos bounds.
     - Ambas versiones se guardan en counterfactual_gap_v8.csv

Sin cambios en:
  - Arquitectura LightGBM (mismos hiperpar?metros)
  - TimeSeriesSplit 5 folds
  - HDD features forzadas
  - Prueba del Domingo ajustada

Uso:
    python src/ml/train_model_v8.py
    python src/ml/train_model_v8.py --tune
    python src/ml/train_model_v8.py --skip-cv   # solo an?lisis causal
"""

import sys
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ??? RUTAS ????????????????????????????????????????????????????????????????????
ROOT_DIR      = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH  = PROCESSED_DIR / "features_daily.parquet"

# ??? CONFIG ???????????????????????????????????????????????????????????????????
TARGETS        = ["NO2", "PM10", "PM2.5", "ICA"]
ZBE_DATE       = pd.Timestamp("2025-09-01", tz="UTC")
N_SPLITS       = 5
HORIZON        = "d1"
TOP_N_FEATURES = 80
EARLY_STOPPING_ROUNDS = 100

HDD_FEATURES_REQUIRED = [
    "HDD", "HDD_acum_7d", "HDD_acum_14d", "HDD_lag_1d", "HDD_lag_7d",
    "fc_HDD_d1", "fc_HDD_d2", "fc_HDD_d3",
    "dia_muy_frio", "es_invierno_estricto", "domingo_invierno", "es_domingo",
]

# Covariables meteorol?gicas para DiD y counterfactual
METEO_COVARIATES = [
    "HDD", "HDD_acum_7d", "temperature_2m", "precipitation",
    "wind_speed_10m", "boundary_layer_height", "cloud_cover",
    "relative_humidity_2m", "sunshine_duration",
    "is_weekend", "day_of_week", "month", "season",
]

# (9) Features permitidas en el counterfactual METEO-PURO
# Solo meteorolog?a y calendario - sin lags del contaminante
METEO_PURE_FEATURES = [
    "HDD", "HDD_acum_7d", "HDD_acum_14d", "HDD_lag_1d", "HDD_lag_7d",
    "temperature_2m", "precipitation", "wind_speed_10m", "wind_u", "wind_v",
    "boundary_layer_height", "cloud_cover", "relative_humidity_2m",
    "sunshine_duration", "pressure_msl", "temp_range", "dia_muy_frio",
    "is_weekend", "is_workday", "day_of_week", "month", "season",
    "day_of_year", "week_of_year", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "es_invierno_estricto", "es_verano", "es_domingo", "domingo_invierno",
    "exp_traffic_volume_d1", "exp_traffic_occupancy_d1",
    "traffic_volume_lag_7d", "traffic_volume_lag_14d",
]

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

TUNE_GRID = {
    "num_leaves":        [31, 50, 63, 90, 127],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "min_child_samples": [10, 20, 30, 40, 50],
    "subsample":         [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
    "reg_alpha":         [0.0, 0.05, 0.1, 0.5],
    "reg_lambda":        [0.0, 0.05, 0.1, 0.5],
}

N_ITER_RANDOM  = 100
MAPE_THRESHOLD = 5.0

# ??? HELPERS ??????????????????????????????????????????????????????????????????
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

def section(title):
    log(); log("=" * 65); log(f"  {title}"); log("=" * 65)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def mape(y_true, y_pred):
    mask = y_true > MAPE_THRESHOLD
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ??? 1. CARGAR DATOS ??????????????????????????????????????????????????????????
def load_dataset():
    section("1. Cargando dataset")

    if not DATASET_PATH.exists():
        log(f"  ? No se encuentra {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATASET_PATH)
    log(f"  Filas    : {len(df):,}")

    all_target_cols = [c for c in df.columns
                       if c.startswith("target_") and c.endswith(f"_{HORIZON}")]
    log(f"  Targets  : {len(all_target_cols)} (solo {HORIZON})")

    raw_contaminants = ["NO2", "PM10", "PM2.5", "ICA",
                        "humedad", "presion", "temperatura", "viento_dir", "viento_vel"]
    datetime_cols = [c for c in df.columns
                     if pd.api.types.is_datetime64_any_dtype(df[c])]
    feature_cols = [c for c in df.columns
                    if not c.startswith("target_")
                    and c != "date"
                    and c not in raw_contaminants
                    and c not in datetime_cols]

    null_pct = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if null_pct[c] <= 0.5]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

    hdd_available = [f for f in HDD_FEATURES_REQUIRED if f in feature_cols]
    hdd_missing   = [f for f in HDD_FEATURES_REQUIRED if f not in feature_cols]
    log(f"  Features HDD disponibles : {len(hdd_available)} / {len(HDD_FEATURES_REQUIRED)}")
    if hdd_missing:
        log(f"  [WARN]  HDD features ausentes: {hdd_missing}")

    log(f"  Features totales : {len(feature_cols)}")
    return df, feature_cols, all_target_cols


# ??? 2. VERIFICAR FEATURES ????????????????????????????????????????????????????
def verify_features(df, feature_cols):
    section("2. Verificando features del dataset v6")

    hdd_cols = [c for c in feature_cols if "HDD" in c or "hdd" in c.lower()
                or c in ["dia_muy_frio", "es_invierno_estricto",
                         "domingo_invierno", "es_domingo", "es_verano"]]
    log(f"  Features calderas/HDD : {len(hdd_cols)}")
    for c in hdd_cols:
        if c in df.columns:
            log(f"    {c:<30} mean={df[c].mean():.3f}  std={df[c].std():.3f}")

    return df, feature_cols


# ??? 3. SELECCI?N DE FEATURES ????????????????????????????????????????????????
def select_features_permutation(X_train, y_train, X_val, y_val,
                                  feature_cols, target_name):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    log(f"\n  Calculando permutation importance para {target_name}...")
    model = LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 300})
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])

    perm       = permutation_importance(model, X_val, y_val, n_repeats=5,
                                        random_state=42,
                                        scoring="neg_root_mean_squared_error",
                                        n_jobs=-1)
    sorted_idx = np.argsort(perm.importances_mean)[::-1]
    n_select   = max(min(TOP_N_FEATURES, (perm.importances_mean > 0).sum()), 20)
    selected   = [feature_cols[i] for i in sorted_idx[:n_select]]

    hdd_forced = [f for f in HDD_FEATURES_REQUIRED
                  if f in feature_cols and f not in selected]
    if hdd_forced:
        selected = selected + hdd_forced
        log(f"  HDD features forzadas : {hdd_forced}")

    log(f"  Features seleccionadas: {len(selected)}")
    log(f"  Top 5: {selected[:5]}")
    return selected


# ??? 4. ENTRENAR UN MODELO ????????????????????????????????????????????????????
def train_single(X_train, y_train, X_val, y_val, tune=False):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    callbacks = [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                 log_evaluation(-1)]
    params = LGBM_PARAMS.copy()

    if tune:
        best_rmse, best_params = np.inf, params.copy()
        for gp in ParameterSampler(TUNE_GRID, n_iter=N_ITER_RANDOM, random_state=42):
            p = {**params, **gp, "n_estimators": 1000}
            m = LGBMRegressor(**p)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            r = rmse(y_val.values, m.predict(X_val))
            if r < best_rmse:
                best_rmse, best_params = r, p
        params = best_params
        log(f"    Mejor RMSE tuning: {best_rmse:.4f}")

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


# ??? 5. ENTRENAR TODOS (CV) ???????????????????????????????????????????????????
def train_all(df, feature_cols, target_cols, tune=False):
    section(f"3. Entrenamiento LightGBM v8 - {HORIZON} ? TimeSeriesSplit ({N_SPLITS} Folds)")

    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    folds  = list(tscv.split(df[feature_cols].fillna(0)))
    X_full = df[feature_cols].fillna(0)

    all_metrics, all_importances, all_selected = {}, {}, {}

    pbar = tqdm(target_cols, desc="Entrenando", unit="target")
    for target_col in pbar:
        pbar.set_description(f"Procesando {target_col}")
        y_full       = df[target_col].fillna(df[target_col].median())
        fold_metrics = {"rmse": [], "mae": [], "r2": [], "mape": []}

        last_tr, last_va = folds[-1]
        selected = select_features_permutation(
            X_full.iloc[last_tr], y_full.iloc[last_tr],
            X_full.iloc[last_va], y_full.iloc[last_va],
            feature_cols, target_col,
        )
        all_selected[target_col] = selected
        X_sel = df[selected].fillna(0)

        final_model = None
        for tr_idx, va_idx in folds:
            model = train_single(X_sel.iloc[tr_idx], y_full.iloc[tr_idx],
                                 X_sel.iloc[va_idx],   y_full.iloc[va_idx], tune=tune)
            pred = model.predict(X_sel.iloc[va_idx])
            fold_metrics["rmse"].append(rmse(y_full.iloc[va_idx].values, pred))
            fold_metrics["mae"].append(mae(y_full.iloc[va_idx].values, pred))
            fold_metrics["r2"].append(r2(y_full.iloc[va_idx].values, pred))
            fold_metrics["mape"].append(mape(y_full.iloc[va_idx].values, pred))
            final_model = model

        all_metrics[target_col] = {
            "cv_rmse":    round(np.mean(fold_metrics["rmse"]), 4),
            "cv_mae":     round(np.mean(fold_metrics["mae"]), 4),
            "cv_r2":      round(np.mean(fold_metrics["r2"]), 4),
            "cv_mape":    round(np.nanmean(fold_metrics["mape"]), 2),
            "n_features": len(selected),
        }
        all_importances[target_col] = dict(zip(selected, final_model.feature_importances_))
        joblib.dump(final_model,
            MODELS_DIR / f"lgbm_v8_{target_col.replace('target_', '')}.pkl")
        with open(MODELS_DIR / f"lgbm_v8_{target_col.replace('target_', '')}_features.json", "w") as f:
            json.dump(selected, f)

    return all_metrics, all_importances, all_selected


# ??? (9) COUNTERFACTUAL DUAL ???????????????????????????????????????????????????
def train_counterfactual_dual(df, feature_cols):
    """
    Counterfactual dual - bound inferior y superior del efecto ZBE.

    METEO-PURO (bound conservador):
        Solo meteorolog?a + calendario + tr?fico hist?rico esperado.
        Sin lags del propio contaminante. Efecto m?nimo garantizado.

    CON-LAGS (bound superior):
        Meteo + calendario + lags del contaminante en per?odo pre-ZBE.
        Puede absorber memoria del contaminante. Efecto m?ximo posible.

    El efecto real de la ZBE est? entre ambos bounds.
    """
    section("(9) Counterfactual Dual (Bound Inferior / Bound Superior)")

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    pre_mask  = df["date"] < ZBE_DATE
    post_mask = df["date"] >= ZBE_DATE
    n_pre, n_post = pre_mask.sum(), post_mask.sum()

    log(f"  D?as pre-ZBE  (train): {n_pre}")
    log(f"  D?as post-ZBE (test) : {n_post}")

    if n_pre < 60 or n_post == 0:
        log("  [WARN]  Datos insuficientes para counterfactual")
        return []

    lag_features  = [c for c in feature_cols
                     if any(t in c for t in ["_lag_", "_roll_mean_", "_roll_std_"])
                     and c in df.columns]
    pure_features = [c for c in METEO_PURE_FEATURES if c in df.columns]

    fit_callbacks = [early_stopping(50, verbose=False), log_evaluation(-1)]
    cf_params     = {**LGBM_PARAMS, "n_estimators": 500}

    results = []

    for cont in TARGETS:
        for zone in ["zbe", "out"]:
            target_raw = f"{cont}_{zone}"
            y_col      = f"target_{cont}_{zone}_d1"
            if y_col not in df.columns:
                y_col = target_raw
            if y_col not in df.columns:
                continue

            df_pre  = df[pre_mask].dropna(subset=[y_col])
            df_post = df[post_mask].dropna(subset=[y_col])
            if df_pre.empty or df_post.empty:
                continue

            log(f"\n  ?? {target_raw} ??????????????????????????????????????????")
            observed = df_post[y_col].values

            version_results = {}

            for version_name, extra_feats in [
                ("METEO-PURO", []),
                ("CON-LAGS",   lag_features),
            ]:
                all_feats = list(dict.fromkeys(pure_features + extra_feats))
                feat_ok   = [c for c in all_feats
                             if c in df.columns
                             and df_pre[c].notna().mean() > 0.5]

                X_pre  = df_pre[feat_ok].fillna(0)
                y_pre  = df_pre[y_col]
                X_post = df_post[feat_ok].fillna(0)

                split  = int(len(X_pre) * 0.8)
                X_tr   = X_pre.iloc[:split]
                X_va   = X_pre.iloc[split:] if split < len(X_pre) - 5 else X_pre
                y_tr   = y_pre.iloc[:split]
                y_va   = y_pre.iloc[split:] if split < len(y_pre) - 5 else y_pre

                model_cf = LGBMRegressor(**cf_params)
                model_cf.fit(X_tr, y_tr,
                             eval_set=[(X_va, y_va)],
                             callbacks=fit_callbacks)

                val_r2_v  = r2(y_va.values, model_cf.predict(X_va))
                pred_post = model_cf.predict(X_post)
                gap       = observed - pred_post
                gap_mean  = np.mean(gap)
                gap_pct   = gap_mean / np.mean(pred_post) * 100 if np.mean(pred_post) > 0 else 0

                log(f"  [{version_name}]  n_features={len(feat_ok)}  R?_val={val_r2_v:.3f}")
                log(f"    Observado     : {np.mean(observed):.2f} ?g/m?")
                log(f"    Predicho (CF) : {np.mean(pred_post):.2f} ?g/m?")
                log(f"    Gap medio     : {gap_mean:+.2f} ?g/m?  ({gap_pct:+.1f}%)")

                version_results[version_name] = {"gap_mean": gap_mean, "gap_pct": gap_pct}

                joblib.dump(model_cf,
                    MODELS_DIR / f"lgbm_v8_cf_{target_raw}_{version_name.replace('-','_').lower()}.pkl")

                for date, obs, pred, g in zip(
                    df_post["date"].values, observed, pred_post, gap
                ):
                    results.append({
                        "date":           pd.Timestamp(date).date(),
                        "contaminant":    cont,
                        "zone":           zone,
                        "version":        version_name,
                        "observed":       round(float(obs), 3),
                        "counterfactual": round(float(pred), 3),
                        "gap":            round(float(obs - pred), 3),
                        "gap_pct":        round(float((obs - pred) / pred * 100)
                                                if pred > 0 else 0, 2),
                    })

            # Resumen del rango de bounds
            if "METEO-PURO" in version_results and "CON-LAGS" in version_results:
                g_pure = version_results["METEO-PURO"]["gap_pct"]
                g_lags = version_results["CON-LAGS"]["gap_pct"]
                low, high = min(g_pure, g_lags), max(g_pure, g_lags)
                log(f"")
                log(f"  ? RANGO EFECTO ZBE en {target_raw}:")
                log(f"     Bound conservador (METEO-PURO) : {g_pure:+.1f}%")
                log(f"     Bound optimista   (CON-LAGS)   : {g_lags:+.1f}%")
                if low < 0 and high < 0:
                    log(f"     [OK] Ambos bounds negativos -> reducci?n robusta")
                    log(f"        Efecto estimado: entre {abs(high):.1f}% y {abs(low):.1f}% de reducci?n")
                elif low < 0 <= high:
                    log(f"     [WARN]  Bounds en direcciones opuestas -> efecto incierto")
                    log(f"        Necesitas datos de verano 2026 para confirmar")
                else:
                    log(f"     [WARN]  Ambos bounds positivos -> aumento neto en {target_raw}")
                    log(f"        Fuente no-tr?fico domina (hip?tesis calderas)")

    if results:
        gap_df = pd.DataFrame(results)
        gap_df.to_csv(MODELS_DIR / "counterfactual_gap_v8.csv", index=False)
        log(f"\n  [OK] counterfactual_gap_v8.csv guardado")
        log(f"     Filtra version='METEO-PURO' o 'CON-LAGS' para comparar bounds")

    return results


# ??? (7)(8) DiD CON EFECTOS FIJOS Y CLUSTERING ??????????????????????????????????
def did_analysis_v8(df):
    """
    Difference-in-Differences v8:
        pollution_it = ?0 + ?1?Post + ?2?ZBE + ?3?(Post?ZBE)
                     + ??_my?C(month_year)   ? efectos fijos (7)
                     + ???meteo
                     + ?_it

    SE clusterizados por fecha (8) - m?s conservadores que HC3.
    ?3 = efecto causal neto de la ZBE.
    """
    section("(7)(8) Difference-in-Differences v8 - Efectos Fijos + Clustering")

    try:
        import statsmodels.formula.api as smf
        use_sm = True
        log("  Motor: statsmodels OLS")
        log("  (7) Efectos fijos: C(month_year)  - 22 dummies mes?a?o")
        log("  (8) SE clusterizados por fecha")
    except ImportError:
        use_sm = False
        log("  [WARN]  statsmodels no instalado - DiD manual 2?2")
        log("     pip install statsmodels")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # (7) Variable mes?a?o para efectos fijos
    df["month_year"] = (df["date"].dt.year.astype(str) + "_"
                        + df["date"].dt.month.astype(str).str.zfill(2))

    # Construir panel largo (date ? zone)
    panel_rows = []
    for cont in ["NO2", "PM10", "PM2.5"]:
        col_zbe = f"{cont}_zbe"
        col_out = f"{cont}_out"
        if col_zbe not in df.columns or col_out not in df.columns:
            continue

        for _, row in df.iterrows():
            base = {
                "date":        row["date"],
                "month_year":  row["month_year"],
                "Post":        int(row["date"] >= ZBE_DATE),
                "contaminant": cont,
            }
            for cov in METEO_COVARIATES:
                if cov in df.columns:
                    base[cov] = row[cov] if not pd.isna(row.get(cov, np.nan)) else 0

            if not pd.isna(row.get(col_zbe, np.nan)):
                panel_rows.append({**base, "ZBE": 1, "pollution": row[col_zbe]})
            if not pd.isna(row.get(col_out, np.nan)):
                panel_rows.append({**base, "ZBE": 0, "pollution": row[col_out]})

    panel = pd.DataFrame(panel_rows)
    panel["Post_x_ZBE"] = panel["Post"] * panel["ZBE"]

    n_month_year = panel["month_year"].nunique()
    log(f"\n  Panel: {len(panel):,} obs ? {panel['date'].nunique()} fechas")
    log(f"  Dummies mes?a?o ((7)): {n_month_year}")
    log(f"  Clusters fecha  ((8)): {panel['date'].nunique()}")

    did_results = {}

    for cont in ["NO2", "PM10", "PM2.5"]:
        sub = panel[panel["contaminant"] == cont].dropna(subset=["pollution"])
        if len(sub) < 50:
            log(f"\n  [WARN]  {cont}: datos insuficientes ({len(sub)} obs)")
            continue

        log(f"\n  ?? DiD v8: {cont} ??????????????????????????????????????????")

        cov_available = [c for c in METEO_COVARIATES
                         if c in sub.columns and sub[c].notna().mean() > 0.7]

        if use_sm:
            cov_str = " + ".join(cov_available) if cov_available else "1"
            formula = (
                f"pollution ~ Post + ZBE + Post_x_ZBE"
                f" + C(month_year)"        # (7) efectos fijos mes?a?o
                f" + {cov_str}"
            )

            try:
                model = smf.ols(formula, data=sub).fit(
                    cov_type="cluster",    # (8) SE clusterizados
                    cov_kwds={"groups": sub["date"]}
                )

                beta3   = model.params.get("Post_x_ZBE", np.nan)
                pval    = model.pvalues.get("Post_x_ZBE", np.nan)
                ci      = model.conf_int()
                ci_low  = ci.loc["Post_x_ZBE", 0] if "Post_x_ZBE" in ci.index else np.nan
                ci_high = ci.loc["Post_x_ZBE", 1] if "Post_x_ZBE" in ci.index else np.nan
                r2_val  = model.rsquared

                log(f"  ?1 (Post)           : {model.params.get('Post', 0):+.3f}")
                log(f"  ?2 (ZBE)            : {model.params.get('ZBE', 0):+.3f}")
                log(f"  ?3 (Post ? ZBE)     : {beta3:+.3f}  ? EFECTO CAUSAL ZBE")
                log(f"  p-valor ?3          : {pval:.4f}  "
                    f"{'[OK] p<0.05' if pval < 0.05 else '[WARN]  no significativo'}")
                log(f"  IC 95% ?3           : [{ci_low:+.3f}, {ci_high:+.3f}]")
                log(f"  R?                  : {r2_val:.3f}")
                log(f"  Efectos fijos (7)     : {n_month_year} dummies C(month_year)")
                log(f"  SE (8)                : clusterizados por fecha")

                mean_pre = sub[(sub["Post"] == 0) & (sub["ZBE"] == 1)]["pollution"].mean()
                if mean_pre > 0 and not np.isnan(beta3):
                    pct = beta3 / mean_pre * 100
                    log(f"  Efecto relativo     : {pct:+.1f}% vs pre-ZBE ({mean_pre:.2f} ?g/m?)")

                log(f"")
                if not np.isnan(pval) and pval < 0.05:
                    if beta3 < 0:
                        log(f"  [OK] ZBE redujo {cont} en {abs(beta3):.2f} ?g/m? - CAUSAL")
                        log(f"     (controlando estacionalidad mes?a?o, meteo y clustering)")
                    else:
                        log(f"  [WARN]  {cont} AUMENT? {beta3:.2f} ?g/m? en ZBE vs control")
                        log(f"     -> Fuente no-tr?fico (calderas residenciales) domina")
                        log(f"     -> O desplazamiento de tr?fico al l?mite de la ZBE")
                else:
                    log(f"  ?  Efecto no significativo (p={pval:.3f})")
                    log(f"     Con datos de verano 2026 el poder estad?stico mejorar?")

                did_results[cont] = {
                    "beta3":       round(float(beta3), 4) if not np.isnan(beta3) else None,
                    "pvalue":      round(float(pval), 4)  if not np.isnan(pval)  else None,
                    "ci_low":      round(float(ci_low), 4)   if not np.isnan(ci_low)  else None,
                    "ci_high":     round(float(ci_high), 4)  if not np.isnan(ci_high) else None,
                    "r2":          round(float(r2_val), 4),
                    "n_obs":       len(sub),
                    "significant": bool(not np.isnan(pval) and pval < 0.05),
                    "fe_dummies":  n_month_year,
                    "se_type":     "clustered_by_date",
                    "formula":     formula,
                }

            except Exception as e:
                log(f"  ? Error OLS: {e}")
                use_sm = False

        if not use_sm:
            # DiD manual 2?2 como fallback
            zbe_pre  = sub[(sub["Post"] == 0) & (sub["ZBE"] == 1)]["pollution"].mean()
            zbe_post = sub[(sub["Post"] == 1) & (sub["ZBE"] == 1)]["pollution"].mean()
            out_pre  = sub[(sub["Post"] == 0) & (sub["ZBE"] == 0)]["pollution"].mean()
            out_post = sub[(sub["Post"] == 1) & (sub["ZBE"] == 0)]["pollution"].mean()
            did_m    = (zbe_post - zbe_pre) - (out_post - out_pre)
            did_pct  = did_m / zbe_pre * 100 if zbe_pre > 0 else 0

            log(f"  {'':15} {'Pre-ZBE':>12} {'Post-ZBE':>12} {'?':>10}")
            log(f"  {'ZBE':15} {zbe_pre:>12.2f} {zbe_post:>12.2f} {zbe_post-zbe_pre:>+10.2f}")
            log(f"  {'OUT (control)':15} {out_pre:>12.2f} {out_post:>12.2f} {out_post-out_pre:>+10.2f}")
            log(f"  {'DiD (?3)':15} {'':>12} {'':>12} {did_m:>+10.2f}  ({did_pct:+.1f}%)")
            did_results[cont] = {"beta3": round(did_m, 4), "pvalue": None, "n_obs": len(sub)}

    did_path = MODELS_DIR / "did_results_v8.json"
    with open(did_path, "w", encoding="utf-8") as f:
        json.dump(did_results, f, indent=2, ensure_ascii=False)
    log(f"\n  [OK] did_results_v8.json guardado")

    return did_results


# ??? FUNCIONES AUXILIARES ?????????????????????????????????????????????????????
def print_metrics_summary(all_metrics, all_selected):
    section("4. Resumen de m?tricas (Media Cross-Validation)")
    log(f"  {'Target':<22} {'CV RMSE':>10} {'CV R2':>8} {'CV MAPE%':>9} {'Features':>9}")
    log(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*9} {'-'*9}")
    for cont in TARGETS:
        cm = {k: v for k, v in all_metrics.items() if f"_{cont}_" in k}
        if cm:
            log(f"\n  -- {cont} --")
            for tc, m in sorted(cm.items()):
                log(f"  {tc.replace('target_',''):<22} {m['cv_rmse']:>10.3f} "
                    f"{m['cv_r2']:>8.3f} {m['cv_mape']:>9.1f} {m.get('n_features','-'):>9}")

    log(f"\n  -- Comparativa v7 -> v8 --")
    v7_ref = {
        "NO2_out_d1":   (4.710, 0.390), "NO2_zbe_d1":   (3.993, 0.375),
        "PM10_out_d1":  (4.354, 0.394), "PM10_zbe_d1":  (4.314, 0.365),
        "PM2.5_out_d1": (3.233, 0.445), "PM2.5_zbe_d1": (2.698, 0.479),
        "ICA_out_d1":   (8.147, 0.396), "ICA_zbe_d1":   (6.956, 0.374),
    }
    log(f"  {'Target':<22} {'v7 RMSE':>9} {'v8 RMSE':>9} {'?':>8} {'v7 R?':>7} {'v8 R?':>7}")
    log(f"  {'-'*66}")
    for tc, m in sorted(all_metrics.items()):
        key = tc.replace("target_", "")
        if key in v7_ref:
            v7r, v7r2 = v7_ref[key]
            dr = m["cv_rmse"] - v7r
            log(f"  {key:<22} {v7r:>9.3f} {m['cv_rmse']:>9.3f} "
                f"{dr:>+8.3f}{'[OK]' if dr < 0 else '[WARN] '} {v7r2:>7.3f} {m['cv_r2']:>7.3f}")


def analyze_hdd_importance(all_importances):
    section("5. HDD en Feature Importance")
    target_key = "target_NO2_zbe_d1"
    if target_key not in all_importances:
        return
    imp_sorted = sorted(all_importances[target_key].items(), key=lambda x: x[1], reverse=True)
    log(f"  Top 15 para NO2_zbe_d1:")
    hdd_ranks = {}
    for rank, (feat, val) in enumerate(imp_sorted[:15], 1):
        is_hdd = any(h in feat for h in ["HDD", "dia_muy_frio", "invierno", "domingo"])
        log(f"  {rank:<4} {feat:<45} {val:>10.1f}{'  ?' if is_hdd else ''}")
        if is_hdd:
            hdd_ranks[feat] = rank
    if hdd_ranks:
        top10 = {k: v for k, v in hdd_ranks.items() if v <= 10}
        if top10:
            log(f"  ? HDD en Top 10: {list(top10.keys())}")
        else:
            log(f"  HDD fuera del Top 10 (mejor: #{min(hdd_ranks.values())})")


def prueba_domingo_ajustada(df, all_importances):
    section("6. Prueba del Domingo ajustada por HDD")
    if "NO2_zbe" not in df.columns:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    inv     = df[df["date"].dt.month.isin([11, 12, 1, 2])]
    dom_inv = inv[inv["date"].dt.dayofweek == 6]
    pre     = dom_inv[dom_inv["date"] <  ZBE_DATE]
    post    = dom_inv[dom_inv["date"] >= ZBE_DATE]
    if pre.empty or post.empty:
        log("  [WARN]  Datos insuficientes")
        return
    delta_raw = post["NO2_zbe"].mean() - pre["NO2_zbe"].mean()
    pct_raw   = delta_raw / pre["NO2_zbe"].mean() * 100
    log(f"  Domingos invierno PRE  (n={len(pre)}): {pre['NO2_zbe'].mean():.2f} ?g/m?")
    log(f"  Domingos invierno POST (n={len(post)}): {post['NO2_zbe'].mean():.2f} ?g/m?")
    log(f"  ? sin ajuste = {delta_raw:+.2f} ?g/m?  ({pct_raw:+.1f}%)")
    if "HDD" in df.columns:
        mask = df["HDD"].notna() & df["NO2_zbe"].notna()
        coef = np.polyfit(df.loc[mask, "HDD"].values,
                          df.loc[mask, "NO2_zbe"].values, 1)
        hdd_adj   = (post["HDD"].mean() - pre["HDD"].mean()) * coef[0]
        delta_adj = delta_raw - hdd_adj
        pct_adj   = delta_adj / pre["NO2_zbe"].mean() * 100
        log(f"  Sensibilidad NO2/HDD = {coef[0]:+.3f} ?g/m? por HDD")
        log(f"  ? ajustado           = {delta_adj:+.2f} ?g/m?  ({pct_adj:+.1f}%)")
        if pct_adj < -5:
            log(f"  [OK] ZBE reduce NO2 incluso ajustando temperatura")
        elif pct_adj > 5:
            log(f"  [WARN]  NO2 sube ajustando - fuente estructural (calderas)")
        else:
            log(f"  ?  Cambio explicado por temperatura")


def save_feature_importance(all_importances):
    section("7. Feature importance global")
    imp_df = pd.DataFrame(all_importances).fillna(0)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df = imp_df[["mean"]].sort_values("mean", ascending=False)
    for feat, row in imp_df.head(15).iterrows():
        is_hdd = any(h in str(feat) for h in ["HDD", "dia_muy_frio", "invierno", "domingo"])
        log(f"  {str(feat):<50} {row['mean']:>10.1f}{'  ?' if is_hdd else ''}")
    imp_df.reset_index().rename(columns={"index": "feature"}).to_csv(
        MODELS_DIR / "feature_importance_v8.csv", index=False)


def analyze_zbe_effect(df):
    section("8. Cambio observado pre/post ZBE")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    pre  = df[df["date"] < ZBE_DATE]
    post = df[df["date"] >= ZBE_DATE]
    for cont in TARGETS:
        for zone in ["zbe", "out"]:
            col = f"{cont}_{zone}"
            if col in df.columns:
                pm, pom = pre[col].mean(), post[col].mean()
                chg = (pom - pm) / pm * 100 if pm > 0 else 0
                log(f"  {col:<12}: pre={pm:.2f}  post={pom:.2f}  {chg:+.1f}% {'v' if chg<0 else '^'}")


def save_outputs(all_metrics, did_results=None):
    section("9. Guardando m?tricas y reporte")
    with open(MODELS_DIR / "metrics_v8.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    summary = {}
    for cont in TARGETS:
        cm = {k: v for k, v in all_metrics.items() if f"_{cont}_" in k}
        if cm:
            summary[cont] = {
                "avg_cv_rmse": round(np.mean([v["cv_rmse"] for v in cm.values()]), 4),
                "avg_cv_r2":   round(np.mean([v["cv_r2"]   for v in cm.values()]), 4),
            }
    if did_results:
        summary["did_v8"] = did_results
    with open(MODELS_DIR / "metrics_summary_v8.json", "w") as f:
        json.dump(summary, f, indent=2)
    (MODELS_DIR / "training_report_v8.txt").write_text(
        "\n".join(report_lines), encoding="utf-8")


# ??? MAIN ?????????????????????????????????????????????????????????????????????
def main():
    start   = time.time()
    tune    = "--tune"    in sys.argv
    skip_cv = "--skip-cv" in sys.argv

    log("=" * 65)
    log("  TRAIN MODEL v8 - Vitoria Air Quality (ZBE Causal - Paper Level)")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  (7) C(month_year)  (8) SE clusterizados  (9) Counterfactual dual")
    log(f"  Tuning: {'s?' if tune else 'no'}  |  Skip CV: {'s?' if skip_cv else 'no'}")
    log("=" * 65)

    df, feature_cols, target_cols = load_dataset()
    df, feature_cols = verify_features(df, feature_cols)

    all_metrics, all_importances, all_selected = {}, {}, {}

    if not skip_cv:
        all_metrics, all_importances, all_selected = train_all(
            df, feature_cols, target_cols, tune=tune)
        print_metrics_summary(all_metrics, all_selected)
        analyze_hdd_importance(all_importances)
        prueba_domingo_ajustada(df, all_importances)
        save_feature_importance(all_importances)

    train_counterfactual_dual(df, feature_cols)   # (9)
    did_results = did_analysis_v8(df)             # (7)(8)

    analyze_zbe_effect(df)
    save_outputs(all_metrics, did_results)

    elapsed = time.time() - start
    log("\n" + "=" * 65)
    log(f"  [OK] v8 COMPLETADO")
    log(f"  Time: {int(elapsed//60)} min {int(elapsed%60)} s")
    log(f"  counterfactual_gap_v8.csv  - observed vs CF (meteo-puro / con-lags)")
    log(f"  did_results_v8.json        - beta3, p-valor IC 95%")
    log(f"  training_report_v8.txt     - log completo")
    log("=" * 65)
    log("")
    log("  PARA EL PAPER:")
    log("  Tabla 1 - DiD: beta3 con IC 95% clusterizado y efectos fijos C(month_year)")
    log("  Figura 1 - observed vs counterfactual METEO-PURO para NO2_zbe")
    log("  Figura 2 - banda de incertidumbre: gap meteo-puro vs con-lags")
    log("  Nota - Con datos verano 2026 repetir para aislar efecto tr?fico puro")
    log("=" * 65)


if __name__ == "__main__":
    main()