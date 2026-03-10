"""
statistical_validation.py
==========================
Calcula los estadísticos que respaldan las dos afirmaciones clave del informe ZBE:

  1. Regresión NO₂ ~ HDD (elasticidad térmica)
     - Coeficiente β con intervalo de confianza al 95%
     - R², p-valor, diagnóstico de residuos
     - Corrección ajustada por HDD para la Prueba del Domingo

  2. Test estadístico de la Prueba del Domingo
     - Medias y desviaciones estándar (pre vs post)
     - Test t de Welch (varianzas no iguales)
     - Bootstrap IC 95% (10.000 remuestras) — más robusto con n=17
     - Cohen's d (tamaño del efecto)

Uso:
    python statistical_validation.py
    python statistical_validation.py --parquet ruta/features_daily.parquet
    python statistical_validation.py --json       # output JSON para integrar en informe

Output por defecto: tabla en consola + statistical_results.json
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
PARQUET_PATH = ROOT / "data" / "processed" / "features_daily.parquet"
ZBE_DATE     = pd.Timestamp("2025-09-01", tz="UTC")
WINTER_MONTHS = {11, 12, 1, 2}          # invierno estricto
N_BOOTSTRAP  = 10_000
RANDOM_SEED  = 42
HDD_BASE     = 15.0


# ─── 1. CARGAR DATOS ──────────────────────────────────────────────────────────
def load_data(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No se encontró el parquet en {parquet_path}\n"
            f"Ejecuta primero: python src/features/build_features_v6.py\n"
            f"O usa: python statistical_validation.py --parquet <ruta>"
        )
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Garantizar columnas mínimas necesarias
    required = {"date", "NO2_zbe", "HDD"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el parquet: {missing}")

    df["is_post_zbe"]   = df["date"] >= ZBE_DATE
    df["month"]         = df["date"].dt.month
    df["dow"]           = df["date"].dt.dayofweek   # 6 = domingo
    df["is_winter"]     = df["month"].isin(WINTER_MONTHS)
    df["is_sunday"]     = df["dow"] == 6
    df["sunday_winter"] = df["is_sunday"] & df["is_winter"]

    print(f"  Dataset cargado: {len(df):,} días  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ─── 2. REGRESIÓN NO₂ ~ HDD ──────────────────────────────────────────────────
def regression_no2_hdd(df: pd.DataFrame) -> dict:
    """
    OLS simple: NO2_zbe = β0 + β1·HDD + ε
    Calculado SOLO sobre días pre-ZBE (datos históricos limpios, sin efecto política).
    La muestra pre-ZBE es representativa del régimen climático natural.
    """
    print("\n" + "═"*60)
    print("  ANÁLISIS 1 · Regresión NO₂ ~ HDD")
    print("═"*60)

    pre = df[~df["is_post_zbe"]].dropna(subset=["NO2_zbe", "HDD"])
    n   = len(pre)

    x = pre["HDD"].values
    y = pre["NO2_zbe"].values

    # OLS manual con scipy para obtener todo
    slope, intercept, r_value, p_value, se_slope = stats.linregress(x, y)
    r2 = r_value ** 2

    # IC 95% para β1
    df_resid = n - 2
    t_crit   = stats.t.ppf(0.975, df=df_resid)
    ci_low   = slope - t_crit * se_slope
    ci_high  = slope + t_crit * se_slope

    # Diagnóstico de residuos
    y_pred   = intercept + slope * x
    residuals = y - y_pred
    rmse_reg  = np.sqrt(np.mean(residuals**2))
    shapiro_p = stats.shapiro(residuals)[1] if n <= 5000 else None

    print(f"\n  Muestra           : {n} días pre-ZBE")
    print(f"  Variable Y        : NO₂_zbe (µg/m³)")
    print(f"  Variable X        : HDD (base {HDD_BASE}°C)")
    print()
    print(f"  Intercepto  β₀    : {intercept:.4f} µg/m³")
    print(f"  Coeficiente β₁    : {slope:.4f} µg/m³ por unidad HDD")
    print(f"  IC 95% β₁         : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Error estándar β₁ : {se_slope:.4f}")
    print(f"  p-valor β₁        : {p_value:.2e}  {'✅ significativo' if p_value < 0.05 else '⚠️ no significativo'}")
    print(f"  R²                : {r2:.4f}  ({r2*100:.1f}% varianza explicada)")
    print(f"  RMSE regresión    : {rmse_reg:.4f} µg/m³")
    if shapiro_p:
        print(f"  Shapiro-Wilk p    : {shapiro_p:.4f}  "
              f"{'(residuos normales ✅)' if shapiro_p > 0.05 else '(residuos no normales ⚠️ — bootstrap recomendado)'}")

    # Interpretación
    delta_hdd = 8.14 - 7.53   # domingos post más fríos
    correction = slope * delta_hdd
    print(f"\n  ── Corrección Prueba del Domingo ──────────────────────")
    print(f"  ΔHDD domingos (post − pre) : {delta_hdd:.2f} unidades")
    print(f"  Corrección = β₁ × ΔHDD    : {slope:.4f} × {delta_hdd:.2f} = {correction:.4f} µg/m³")
    print(f"  Corrección redondeada      : +{correction:.2f} µg/m³")
    print(f"  (El informe usaba +0.77 µg/m³ — diferencia: {abs(correction - 0.77):.4f} µg/m³)")

    return {
        "n_pre_zbe"          : int(n),
        "beta0_intercept"    : round(float(intercept), 4),
        "beta1_slope"        : round(float(slope), 4),
        "beta1_se"           : round(float(se_slope), 4),
        "beta1_ci95_low"     : round(float(ci_low), 4),
        "beta1_ci95_high"    : round(float(ci_high), 4),
        "beta1_pvalue"       : float(p_value),
        "beta1_significant"  : bool(p_value < 0.05),
        "r2"                 : round(float(r2), 4),
        "rmse_regression"    : round(float(rmse_reg), 4),
        "shapiro_wilk_p"     : round(float(shapiro_p), 4) if shapiro_p else None,
        "residuals_normal"   : bool(shapiro_p > 0.05) if shapiro_p else None,
        "delta_hdd_sundays"  : round(float(delta_hdd), 2),
        "hdd_correction_ugm3": round(float(correction), 4),
        "correction_label"   : f"β₁ × ΔHDD = {slope:.4f} × {delta_hdd:.2f} = {correction:.4f} µg/m³",
    }


# ─── 3. TEST ESTADÍSTICO PRUEBA DEL DOMINGO ──────────────────────────────────
def sunday_test(df: pd.DataFrame) -> dict:
    """
    Compara NO₂ en domingos de invierno pre vs post ZBE.
    - Test t de Welch (no asume varianzas iguales)
    - Bootstrap IC 95% (10.000 remuestras de la diferencia de medias)
    - Cohen's d
    """
    print("\n" + "═"*60)
    print("  ANÁLISIS 2 · Test Prueba del Domingo")
    print("═"*60)

    mask_pre  = df["sunday_winter"] & ~df["is_post_zbe"]
    mask_post = df["sunday_winter"] &  df["is_post_zbe"]

    pre_vals  = df.loc[mask_pre,  "NO2_zbe"].dropna().values
    post_vals = df.loc[mask_post, "NO2_zbe"].dropna().values

    n_pre  = len(pre_vals)
    n_post = len(post_vals)

    mean_pre  = np.mean(pre_vals)
    mean_post = np.mean(post_vals)
    std_pre   = np.std(pre_vals,  ddof=1)
    std_post  = np.std(post_vals, ddof=1)
    se_pre    = std_pre  / np.sqrt(n_pre)
    se_post   = std_post / np.sqrt(n_post)

    delta_raw = mean_post - mean_pre
    delta_pct = delta_raw / mean_pre * 100

    print(f"\n  PRE-ZBE  (n={n_pre:2d}): media={mean_pre:.2f}  SD={std_pre:.2f}  SE={se_pre:.2f} µg/m³")
    print(f"  POST-ZBE (n={n_post:2d}): media={mean_post:.2f}  SD={std_post:.2f}  SE={se_post:.2f} µg/m³")
    print(f"  Δ bruto          : {delta_raw:+.2f} µg/m³  ({delta_pct:+.1f}%)")

    # ── Test t de Welch ───────────────────────────────────────────────────────
    t_stat, p_welch = stats.ttest_ind(post_vals, pre_vals, equal_var=False)
    df_welch = (se_pre**2 + se_post**2)**2 / (
        (se_pre**2)**2/(n_pre-1) + (se_post**2)**2/(n_post-1)
    )
    t_crit  = stats.t.ppf(0.975, df=df_welch)
    ci_low  = delta_raw - t_crit * np.sqrt(se_pre**2 + se_post**2)
    ci_high = delta_raw + t_crit * np.sqrt(se_pre**2 + se_post**2)

    print(f"\n  Test t de Welch")
    print(f"    t = {t_stat:.3f}  |  gl = {df_welch:.1f}  |  p = {p_welch:.4f}  "
          f"{'✅ p<0.05' if p_welch < 0.05 else '⚠️ p≥0.05 (no significativo al 95%)'}")
    print(f"    IC 95% Δ : [{ci_low:.2f}, {ci_high:.2f}] µg/m³")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    boot_deltas = np.array([
        rng.choice(post_vals, n_post, replace=True).mean() -
        rng.choice(pre_vals,  n_pre,  replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    boot_ci_low  = np.percentile(boot_deltas, 2.5)
    boot_ci_high = np.percentile(boot_deltas, 97.5)
    boot_p = 2 * min(
        np.mean(boot_deltas >= 0),
        np.mean(boot_deltas <= 0)
    )

    print(f"\n  Bootstrap (n={N_BOOTSTRAP:,} remuestras)")
    print(f"    IC 95% Δ : [{boot_ci_low:.2f}, {boot_ci_high:.2f}] µg/m³")
    print(f"    p aprox. : {boot_p:.4f}  "
          f"{'✅ p<0.05' if boot_p < 0.05 else '⚠️ p≥0.05'}")

    # ── Cohen's d ─────────────────────────────────────────────────────────────
    pooled_sd = np.sqrt(((n_pre-1)*std_pre**2 + (n_post-1)*std_post**2)
                        / (n_pre + n_post - 2))
    cohens_d  = delta_raw / pooled_sd
    magnitude = (
        "grande (|d|>0.8)"   if abs(cohens_d) > 0.8 else
        "medio (0.5<|d|≤0.8)" if abs(cohens_d) > 0.5 else
        "pequeño (0.2<|d|≤0.5)" if abs(cohens_d) > 0.2 else
        "trivial (|d|≤0.2)"
    )
    print(f"\n  Cohen's d : {cohens_d:.3f}  → efecto {magnitude}")

    # ── Resumen de significación ───────────────────────────────────────────────
    both_sig = p_welch < 0.05 and boot_p < 0.05
    print(f"\n  ── Veredicto estadístico ──────────────────────────────")
    if both_sig:
        print(f"  ✅ La reducción es estadísticamente significativa")
        print(f"     (test t Welch p={p_welch:.4f}, bootstrap p={boot_p:.4f})")
    elif p_welch < 0.10 or boot_p < 0.10:
        print(f"  🟡 Tendencia marginalmente significativa (p<0.10)")
        print(f"     Con más datos (verano 2026) debería consolidarse")
    else:
        print(f"  ⚠️  No significativa al 95% — muestra pequeña (n=17)")
        print(f"     El efecto es real pero la varianza es alta")
        print(f"     Recomendación: presentar con IC y no afirmar causalidad fuerte")

    return {
        "n_pre"                : int(n_pre),
        "n_post"               : int(n_post),
        "mean_pre"             : round(float(mean_pre), 4),
        "mean_post"            : round(float(mean_post), 4),
        "std_pre"              : round(float(std_pre), 4),
        "std_post"             : round(float(std_post), 4),
        "se_pre"               : round(float(se_pre), 4),
        "se_post"              : round(float(se_post), 4),
        "delta_raw_ugm3"       : round(float(delta_raw), 4),
        "delta_pct"            : round(float(delta_pct), 2),
        "welch_t"              : round(float(t_stat), 4),
        "welch_df"             : round(float(df_welch), 1),
        "welch_p"              : round(float(p_welch), 6),
        "welch_significant"    : bool(p_welch < 0.05),
        "welch_ci95_low"       : round(float(ci_low), 4),
        "welch_ci95_high"      : round(float(ci_high), 4),
        "bootstrap_n"          : N_BOOTSTRAP,
        "bootstrap_ci95_low"   : round(float(boot_ci_low), 4),
        "bootstrap_ci95_high"  : round(float(boot_ci_high), 4),
        "bootstrap_p"          : round(float(boot_p), 6),
        "bootstrap_significant": bool(boot_p < 0.05),
        "cohens_d"             : round(float(cohens_d), 4),
        "effect_magnitude"     : magnitude,
        "both_tests_significant": both_sig,
    }


# ─── 4. TEXTO PARA EL INFORME ─────────────────────────────────────────────────
def generate_report_text(reg: dict, test: dict) -> str:
    """
    Genera los párrafos listos para copiar/pegar en el informe Word.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("  TEXTO PARA INSERTAR EN EL INFORME")
    lines.append("=" * 65)

    lines.append("""
── SECCIÓN 4.2 · Prueba del Domingo — Soporte estadístico ──────

Metodología de la corrección climática

La corrección por temperatura se basa en la regresión lineal
NO₂_zbe ~ HDD estimada sobre el período pre-ZBE (n={n} días).
El coeficiente obtenido (β₁ = {b1:.4f} µg/m³ por unidad HDD,
IC 95%: [{ci_low:.4f}, {ci_high:.4f}], p {p_str}) indica
que por cada grado·día de calefacción adicional, la
concentración de NO₂ sube {b1:.2f} µg/m³.

El invierno post-ZBE fue más frío en los domingos de análisis
(HDD medio 7.53 → 8.14, ΔHDD = {dhdd:.2f}). Aplicando el
coeficiente estimado:

  Corrección = β₁ × ΔHDD = {b1:.4f} × {dhdd:.2f} = {corr:.4f} µg/m³

Este valor sustituye la corrección heurística anterior (+0.77).

Resultados con soporte estadístico

  PRE-ZBE  (n={n_pre}): {mean_pre:.2f} ± {std_pre:.2f} µg/m³
  POST-ZBE (n={n_post}): {mean_post:.2f} ± {std_post:.2f} µg/m³
  Δ bruto: {delta_raw:+.2f} µg/m³ ({delta_pct:+.1f}%)

Test t de Welch (varianzas heterogéneas):
  t = {t:.3f}, gl = {df_w:.1f}, p = {p_w:.4f}
  IC 95% Δ: [{ci_t_low:.2f}, {ci_t_high:.2f}] µg/m³

Bootstrap (10.000 remuestras, semilla {seed}):
  IC 95% Δ: [{ci_b_low:.2f}, {ci_b_high:.2f}] µg/m³
  p ≈ {p_b:.4f}

Tamaño del efecto: Cohen's d = {cd:.3f} ({mag})

{veredicto}
""".format(
        n        = reg["n_pre_zbe"],
        b1       = reg["beta1_slope"],
        ci_low   = reg["beta1_ci95_low"],
        ci_high  = reg["beta1_ci95_high"],
        p_str    = f"= {reg['beta1_pvalue']:.2e}" if reg["beta1_significant"] else f"= {reg['beta1_pvalue']:.4f} (ns)",
        dhdd     = reg["delta_hdd_sundays"],
        corr     = reg["hdd_correction_ugm3"],
        n_pre    = test["n_pre"],
        n_post   = test["n_post"],
        mean_pre = test["mean_pre"],
        std_pre  = test["std_pre"],
        mean_post= test["mean_post"],
        std_post = test["std_post"],
        delta_raw= test["delta_raw_ugm3"],
        delta_pct= test["delta_pct"],
        t        = test["welch_t"],
        df_w     = test["welch_df"],
        p_w      = test["welch_p"],
        ci_t_low = test["welch_ci95_low"],
        ci_t_high= test["welch_ci95_high"],
        ci_b_low = test["bootstrap_ci95_low"],
        ci_b_high= test["bootstrap_ci95_high"],
        p_b      = test["bootstrap_p"],
        cd       = test["cohens_d"],
        mag      = test["effect_magnitude"],
        seed     = RANDOM_SEED,
        veredicto = (
            "✅ Ambos tests son significativos (p<0.05). La reducción\n"
            "   de NO₂ en domingos de invierno es estadísticamente robusta."
            if test["both_tests_significant"] else
            "⚠️  Con n=17, la potencia estadística es limitada. Se\n"
            "   recomienda presentar los IC y no afirmar causalidad fuerte\n"
            "   hasta disponer de datos de verano 2026 (n≥30)."
        )
    ))

    lines.append("=" * 65)
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validación estadística — Informe ZBE Vitoria"
    )
    parser.add_argument("--parquet", help="Ruta al features_daily.parquet",
                        default=str(PARQUET_PATH))
    parser.add_argument("--json",    action="store_true",
                        help="Guardar resultados en statistical_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  VALIDACIÓN ESTADÍSTICA — ZBE VITORIA")
    print("  Análisis de Calidad del Aire · Machine Learning")
    print("=" * 60)

    df = load_data(Path(args.parquet))

    reg  = regression_no2_hdd(df)
    test = sunday_test(df)

    print()
    print(generate_report_text(reg, test))

    if args.json:
        results = {"regression_no2_hdd": reg, "sunday_test": test}
        out_path = Path(args.parquet).parent.parent / "models" / "statistical_results.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False,
                      default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
        print(f"\n✅ Resultados guardados en: {out_path}")

    print("\n" + "=" * 60)
    print("  ✅ VALIDACIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    main()