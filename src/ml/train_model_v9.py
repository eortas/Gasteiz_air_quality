"""
train_model_v9.py
=================
Entrena y evalúa los modelos causales V9: Event Study Difference-in-Differences 
y Synthetic Control para estimar el impacto de la ZBE.

Lee de `data/processed/station_daily.csv` y genera los JSON en `models/` 
necesarios para pintar la pestaña 3 del Dashboard.

Uso:
    python src/ml/train_model_v9.py --station-data data/processed/station_daily.csv
"""

import sys
import argparse
import json
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

# Silenciar warnings matemáticos menores
warnings.filterwarnings('ignore')

# ─── RUTAS ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUT_ES_FILE = MODELS_DIR / "event_study_v9.json"
OUT_SC_FILE = MODELS_DIR / "synthetic_control_v9.json"

# Importar constantes centralizadas (Fix auditoría #17)
import sys
sys.path.insert(0, str(ROOT_DIR / "src"))
from config import ZBE_DATE_STR

CONTAMINANTS = ["NO2", "PM10", "PM2.5", "ICA"]
TREATED     = ["PAUL", "FUEROS"]
DONORS      = ["LANDAZURI", "HUETOS", "ZUMABIDE", "BEATO"]

QC_FILE = MODELS_DIR / "station_qc.json"


def load_qc_exclusions() -> set:
    """Pares estación_contaminante descartados por el control de calidad."""
    if not QC_FILE.exists():
        return set()
    try:
        qc = json.loads(QC_FILE.read_text(encoding="utf-8"))
        excluded = set(qc.get("excluded", []))
        if excluded:
            print(f"  [QC] Pares excluidos del análisis causal: {', '.join(sorted(excluded))}")
        return excluded
    except Exception as exc:
        print(f"  [WARN] No se pudo leer station_qc.json: {exc}")
        return set()


def load_data(path: Path) -> pd.DataFrame:
    print(f"  Cargando {path}...")
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Asegurarnos de usar PM25 y no PM2.5 en los nombres de columnas
    df.columns = [c.replace("PM2.5", "PM25") for c in df.columns]
    return df


# ─── SYNTHETIC CONTROL ────────────────────────────────────────────────────────
def run_synthetic_control(df: pd.DataFrame):
    """
    Usa regresión Ridge con las estaciones exteriores (Donors) 
    para predecir la estación interior (Treated) en el periodo pre-ZBE.
    Aplica esos pesos al periodo post-ZBE para sacar el "Control Sintético".
    """
    print("\n-- 1. Entrenando Control Sintetico (Ridge Regression) ----------")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    zbe_date = pd.Timestamp(ZBE_DATE_STR)
    
    # Rellenamos solo huecos cortos (máx. 3 días) y nunca extrapolamos el post:
    # interpolar el periodo tratado inventaría la serie observada y los donors.
    df = df.set_index("date").interpolate(
        method="time", limit=3, limit_area="inside"
    ).reset_index()

    # No usamos dropna global: cada modelo descarta después solo sus columnas.
    pre_df = df[df["date"] < zbe_date].copy()
    post_df = df[df["date"] >= zbe_date]

    if len(pre_df) < 30:
        print("  [WARN]️ Insuficientes datos pre-ZBE para Synthetic Control.")
        return {}

    sc_results = {cont: {} for cont in CONTAMINANTS}
    qc_excluded = load_qc_exclusions()

    for cont in CONTAMINANTS:
        cont_col = cont.replace("PM2.5", "PM25")
        
        for tr_station in TREATED:
            y_col = f"{tr_station}_{cont_col}"
            if y_col in qc_excluded:
                print(f"    [QC] {y_col}: excluido por cambio de régimen")
                continue
            X_cols = [f"{d}_{cont_col}" for d in DONORS]
            
            # Verificar si las columnas existen y tienen suficientes datos
            valid_cols = [c for c in [y_col] + X_cols if c in pre_df.columns]
            if y_col not in valid_cols or len(valid_cols) <= 1:
                continue

            available_donors = [c for c in X_cols if c in valid_cols]

            # Entrenamos solo con días pre-ZBE completos para este modelo.
            # Así una estación sin datos en 2024 no recorta el pre de las demás.
            train_df = pre_df[[y_col] + available_donors].dropna()
            if len(train_df) < 30:
                print(f"  [WARN] {y_col}: pre-ZBE insuficiente ({len(train_df)} días)")
                continue
            X_pre = train_df[available_donors]
            y_pre = train_df[y_col]
            
            model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
            model.fit(X_pre, y_pre)
            
            pre_r2 = r2_score(y_pre, model.predict(X_pre))
            
            # Aplicar modelo al periodo Post-ZBE
            # Rellenar NAs en post_df para los donors con la media de la fila
            X_post = post_df[available_donors].ffill().bfill()
            y_post = post_df[y_col]
            
            # Las fechas completas (Pre + Post)
            full_dates = pd.concat([pre_df["date"], post_df["date"]])
            full_y = pd.concat([pre_df[y_col], y_post])
            
            # La predicción para todo el periodo (el Sintético)
            y_pred_pre = model.predict(pre_df[available_donors].ffill().bfill())
            y_pred_post = model.predict(X_post)
            full_synthetic = np.concatenate([y_pred_pre, y_pred_post])

            # Calcular el Gap % en el periodo Post
            # Solo consideramos el post donte y_post tenga valores
            valid_idx = ~y_post.isna()
            obs_post = y_post[valid_idx]
            syn_post = y_pred_post[valid_idx]
            
            if len(obs_post) > 0 and syn_post.mean() > 0:
                gap_abs = obs_post.mean() - syn_post.mean()
                gap_pct = (gap_abs / syn_post.mean()) * 100
            else:
                gap_abs = 0.0
                gap_pct = 0.0

            # Determinando la descripción automática
            if gap_pct < -5.0:
                if pre_r2 > 0.4:
                    desc = f"[OK] Reducción robusta ({gap_abs:+.2f} unit)"
                else:
                    desc = f"[OK] Posible mejora ({gap_abs:+.2f} unit)"
            elif gap_pct > 5.0:
                if pre_r2 > 0.4:
                    desc = f"[WARN] Aumento comprobado ({gap_abs:+.2f} unit)"
                else:
                    desc = f"[WARN] Aumento incierto ({gap_abs:+.2f} unit)"
            else:
                desc = "~ Efecto neto nulo"

            sc_results[cont][tr_station] = {
                "preR2": f"{pre_r2:.2f}",
                "gap": f"{gap_pct:+.1f}%",
                "desc": desc,
                "series": {
                    "dates": full_dates.dt.strftime("%Y-%m-%d").tolist(),
                    "observed": [float(v) if pd.notna(v) else None for v in full_y],
                    "synthetic": [float(v) for v in full_synthetic]
                }
            }
            
            print(f"    [{cont}] {tr_station}: R2={pre_r2:.2f} | Gap={gap_pct:+.1f}%")

    with open(OUT_SC_FILE, "w", encoding="utf-8") as f:
        json.dump(sc_results, f, ensure_ascii=False)
    print(f"  OK Guardado: {OUT_SC_FILE.name}")


# --- EVENT STUDY (DiD) --------------------------------------------------------
def run_event_study(df: pd.DataFrame):
    """
    Calcula un Event Study puro simplificado con bootstrap CIs.
    Compara la media mensual de Treatment (IntraZBE) vs Control (ExtraZBE)
    relativo al mes 0 (implementación).
    
    Los intervalos de confianza se calculan por bootstrap (1000 resamplings)
    de los residuos diarios DiD dentro de cada mes.
    """
    print("\n-- 2. Entrenando Event Study DiD (bootstrap CIs) ---------------")
    
    N_BOOTSTRAP = 1000
    es_results = {cont: {} for cont in CONTAMINANTS}
    zbe_date = pd.Timestamp(ZBE_DATE_STR)
    
    # Configurar mes relativo (0 = Sep 2025)
    df["month_idx"] = (df["date"].dt.year - zbe_date.year) * 12 + (df["date"].dt.month - zbe_date.month)
    
    # Necesitamos al menos el mes -1 (agosto) como referencia
    if -1 not in df["month_idx"].values:
        print("  WARN No se encontro datos de referencia (mes -1) para el Event Study.")
        return

    rng = np.random.default_rng(42)

    # Ventana dinámica: mantenemos 12 meses pre como máximo, pero incluimos
    # todos los meses post disponibles para que la serie no se congele al
    # superar el mes +12.
    available_months = sorted(int(m) for m in df["month_idx"].dropna().unique())
    if not available_months:
        print("  WARN No hay meses válidos para el Event Study.")
        return
    first_month = max(-12, min(available_months))
    last_month = max(available_months)
    qc_excluded = load_qc_exclusions()

    for cont in CONTAMINANTS:
        cont_col = cont.replace("PM2.5", "PM25")
        
        # Agrupar estaciones Treated y control, descartando pares con QC adverso.
        treated_cols = [
            f"{st}_{cont_col}" for st in TREATED
            if f"{st}_{cont_col}" in df.columns and f"{st}_{cont_col}" not in qc_excluded
        ]
        donor_cols = [
            f"{st}_{cont_col}" for st in DONORS
            if f"{st}_{cont_col}" in df.columns and f"{st}_{cont_col}" not in qc_excluded
        ]

        if not treated_cols or not donor_cols:
            print(f"  [QC] {cont}: sin estaciones válidas tras el control de calidad")
            continue
            
        df["treated_mean"] = df[treated_cols].mean(axis=1)
        df["control_mean"] = df[donor_cols].mean(axis=1)
        
        # Calcular DiD diario: (T_d - C_d) para cada día d
        df["daily_diff"] = df["treated_mean"] - df["control_mean"]
        
        # Referencia: media del DiD diario en el mes -1
        ref_days = df[df["month_idx"] == -1]["daily_diff"].dropna()
        if ref_days.empty:
            continue
        ref_diff = ref_days.mean()
        
        coefficients = {}
        
        # Para cada mes, calcular beta y bootstrap CI
        for m_idx in sorted(df["month_idx"].unique()):
            m = int(m_idx)
            if m < first_month or m > last_month:
                continue
            
            month_days = df[df["month_idx"] == m]["daily_diff"].dropna()
            if month_days.empty:
                continue
            
            # El mes de referencia siempre es 0 por construcción
            if m == -1:
                coefficients[str(m)] = {
                    "beta": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0
                }
                continue
            
            beta = month_days.mean() - ref_diff
            
            # Bootstrap: resamplear los DiD diarios del mes y del mes ref
            boot_betas = []
            month_vals = month_days.values
            ref_vals = ref_days.values
            for _ in range(N_BOOTSTRAP):
                boot_month = rng.choice(month_vals, size=len(month_vals), replace=True)
                boot_ref = rng.choice(ref_vals, size=len(ref_vals), replace=True)
                boot_betas.append(boot_month.mean() - boot_ref.mean())
            
            boot_betas = np.array(boot_betas)
            ci_low = float(np.percentile(boot_betas, 2.5))
            ci_high = float(np.percentile(boot_betas, 97.5))
            
            coefficients[str(m)] = {
                "beta": float(beta),
                "ci_low": ci_low,
                "ci_high": ci_high
            }
            
        es_results[cont] = {
            "coefficients": coefficients,
            "reference_month": -1,
            "first_month": first_month,
            "last_month": last_month,
            "stations_used": {
                "treated": [c.split("_")[0] for c in treated_cols],
                "control": [c.split("_")[0] for c in donor_cols],
            },
            "qc_excluded": sorted(
                c for c in qc_excluded if c.endswith(f"_{cont_col}")
            ),
        }

    with open(OUT_ES_FILE, "w", encoding="utf-8") as f:
        json.dump(es_results, f, ensure_ascii=False)
    print(f"  OK Guardado: {OUT_ES_FILE.name}")



# --- GRAFICAR -----------------------------------------------------------------
def trigger_plots():
    print("\n-- 3. Generando Graficos con plot_causal_v9.py -----------------")
    from subprocess import call
    
    script_path = str(ROOT_DIR / "src" / "ml" / "plot_causal_v9.py")
    ret = call([sys.executable, script_path])
    if ret != 0:
        print(f"  [ERROR] Falló la generación de gráficos (plot_causal_v9.py) con código: {ret}")
    else:
        print("  [OK] Gráficos generados correctamente.")


# --- MAIN ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-data", required=True, help="Ruta al station_daily.csv")
    args = parser.parse_args()

    data_path = Path(args.station_data)
    if not data_path.exists():
        print(f"ERR No se encontro el dataset: {data_path}")
        sys.exit(1)

    df = load_data(data_path)
    
    run_synthetic_control(df)
    run_event_study(df)
    
    trigger_plots()
    
    print("\nOK train_model_v9.py finalizado correctamente.")


if __name__ == "__main__":
    main()
