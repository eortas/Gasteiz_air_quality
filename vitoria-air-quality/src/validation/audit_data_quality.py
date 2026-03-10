"""
audit_data_quality.py
======================
Auditoría completa de calidad de datos desde CSVs locales.

Lee:
  - data/raw/traffic/trafico_YYYY.csv
  - data/raw/air/kunak_YYYY.csv
  - data/raw/weather/weather_YYYY.csv

Detecta:
  - Cobertura temporal (primer y último registro)
  - Gaps (horas/días sin datos)
  - Nulos por columna y sensor/estación
  - Solapamiento entre las 3 fuentes (base para v_combined_hourly)

Uso (desde la raíz del proyecto):
    pip install pandas
    python audit_data_quality.py

    python audit_data_quality.py --traffic
    python audit_data_quality.py --air
    python audit_data_quality.py --weather

Genera: audit_report_YYYYMMDD_HHMM.txt
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
TRAFFIC_DIR = ROOT_DIR / "data" / "raw" / "traffic"
AIR_DIR     = ROOT_DIR / "data" / "raw" / "air"
WEATHER_DIR = ROOT_DIR / "data" / "raw" / "weather"

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def section(title):
    log(); log("═" * 65); log(f"  {title}"); log("═" * 65)

def subsection(title):
    log(); log(f"── {title} " + "─" * max(0, 60 - len(title)))

def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "n/a"


def load_csvs(directory: Path, pattern: str, ts_col: str) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        log(f"  ⚠️  Sin archivos '{pattern}' en {directory}")
        return pd.DataFrame()
    frames = []
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        log(f"  Leyendo {f.name} ({size_mb:.1f} MB)...")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df.dropna(subset=[ts_col]).sort_values(ts_col)


def find_gaps(ts_series: pd.Series) -> pd.DataFrame:
    ts = ts_series.dt.floor("h").drop_duplicates().sort_values().dropna()
    if len(ts) < 2:
        return pd.DataFrame()
    expected = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq="h", tz="UTC")
    missing  = expected.difference(ts)
    if len(missing) == 0:
        return pd.DataFrame()
    gaps, start, prev = [], missing[0], missing[0]
    for t in missing[1:]:
        if (t - prev).total_seconds() > 3600:
            gaps.append({"gap_start": start, "gap_end": prev,
                         "hours": int((prev - start).total_seconds() / 3600) + 1})
            start = t
        prev = t
    gaps.append({"gap_start": start, "gap_end": prev,
                 "hours": int((prev - start).total_seconds() / 3600) + 1})
    return pd.DataFrame(gaps).sort_values("hours", ascending=False)


def audit_traffic() -> pd.Series:
    section("1. TRÁFICO — data/raw/traffic/trafico_YYYY.csv")
    df = load_csvs(TRAFFIC_DIR, "trafico_[0-9]*.csv", "start_date")
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    total = len(df)

    subsection("Cobertura temporal")
    log(f"  Primer registro  : {df['start_date'].min()}")
    log(f"  Último registro  : {df['start_date'].max()}")
    log(f"  Total filas      : {total:,}")
    log(f"  Sensores únicos  : {df['code'].nunique()}")
    days = (df['start_date'].max() - df['start_date'].min()).days
    expected = days * 24 * df['code'].nunique()
    log(f"  Cobertura        : {pct(total, expected)} de filas esperadas")

    subsection("Nulos por columna")
    for col in ["volume", "occupancy", "load"]:
        if col in df.columns:
            n = df[col].isna().sum()
            log(f"  {col:<12}: {n:,} nulos ({pct(n, total)})")

    subsection("Top 10 sensores con menos registros")
    counts = df.groupby("code").size().sort_values()
    max_c  = counts.max()
    log(f"  {'Sensor':<15} {'Registros':>10}  {'% sobre máx':>12}")
    for code, cnt in counts.head(10).items():
        log(f"  {str(code):<15} {cnt:>10,}  {pct(cnt, max_c):>12}")

    subsection("Gaps temporales globales")
    gaps = find_gaps(df["start_date"])
    if gaps.empty:
        log("  ✅ Sin gaps detectados")
    else:
        log(f"  ⚠️  {len(gaps)} gaps — {gaps['hours'].sum():,} horas perdidas")
        log(f"  {'Gap start':<22} {'Gap end':<22} {'Horas':>6}")
        for _, row in gaps.head(15).iterrows():
            log(f"  {str(row['gap_start'])[:19]:<22} {str(row['gap_end'])[:19]:<22} {row['hours']:>6}")
        if len(gaps) > 15:
            log(f"  ... y {len(gaps)-15} gaps más")

    return df["start_date"]


def audit_air() -> pd.Series:
    section("2. CALIDAD DEL AIRE — data/raw/air/kunak_YYYY.csv")
    df = load_csvs(AIR_DIR, "kunak_[0-9]*.csv", "timestamp")
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    total = len(df)

    subsection("Cobertura temporal")
    log(f"  Primer registro    : {df['timestamp'].min()}")
    log(f"  Último registro    : {df['timestamp'].max()}")
    log(f"  Total filas        : {total:,}")

    is_long = "contaminante" in df.columns
    log(f"  Formato            : {'largo (contaminante/valor)' if is_long else 'ancho'}")

    if is_long:
        est_col = "estacion_id" if "estacion_id" in df.columns else "estacion"
        log(f"  Estaciones únicas  : {df[est_col].nunique()}")
        log(f"  Contaminantes      : {sorted(df['contaminante'].unique())}")

        subsection("Contaminantes × estación (nº registros)")
        name_col = "estacion" if "estacion" in df.columns else est_col
        pivot = df.groupby([name_col, "contaminante"]).size().unstack(fill_value=0)
        log(pivot.to_string())

        subsection("Nulos en valor por contaminante")
        null_by = df[df["valor"].isna()].groupby("contaminante").size()
        tot_by  = df.groupby("contaminante").size()
        for cont in sorted(df["contaminante"].unique()):
            n = null_by.get(cont, 0); t = tot_by.get(cont, 1)
            log(f"  {cont:<20}: {n:,} nulos ({pct(n, t)})")

        subsection("Gaps por estación")
        for eid, grp in df.groupby(est_col):
            ename = grp["estacion"].iloc[0] if "estacion" in df.columns else str(eid)
            gaps  = find_gaps(grp["timestamp"])
            if gaps.empty:
                log(f"  ✅ {ename} — sin gaps")
            else:
                log(f"  ⚠️  {ename} — {len(gaps)} gaps ({gaps['hours'].sum():,} h perdidas)")
                for _, row in gaps.head(5).iterrows():
                    log(f"      {str(row['gap_start'])[:19]} → {str(row['gap_end'])[:19]} ({row['hours']}h)")
    else:
        cont_cols = [c for c in df.columns if c not in ["timestamp", "estacion", "estacion_id", "timestamp_local"]]
        subsection("Nulos por columna")
        for col in cont_cols:
            n = df[col].isna().sum()
            log(f"  {col:<20}: {n:,} nulos ({pct(n, total)})")

        subsection("Gaps temporales")
        gaps = find_gaps(df["timestamp"])
        if gaps.empty:
            log("  ✅ Sin gaps detectados")
        else:
            log(f"  ⚠️  {len(gaps)} gaps — {gaps['hours'].sum():,} horas perdidas")
            for _, row in gaps.head(10).iterrows():
                log(f"  {str(row['gap_start'])[:19]} → {str(row['gap_end'])[:19]} ({row['hours']}h)")

    return df["timestamp"]


def audit_weather() -> pd.Series:
    section("3. METEOROLOGÍA — data/raw/weather/weather_YYYY.csv")
    df = load_csvs(WEATHER_DIR, "weather_[0-9]*.csv", "timestamp")
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    total = len(df)

    subsection("Cobertura temporal")
    log(f"  Primer registro  : {df['timestamp'].min()}")
    log(f"  Último registro  : {df['timestamp'].max()}")
    log(f"  Total filas      : {total:,}")

    key_cols = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "wind_speed_10m", "wind_direction_10m", "cloud_cover",
                "boundary_layer_height", "pressure_msl"]

    subsection("Nulos por columna clave")
    for col in key_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            log(f"  {col:<30}: {n:,} nulos ({pct(n, total)})")
        else:
            log(f"  {col:<30}: ⚠️  columna no encontrada")

    subsection("Gaps temporales")
    gaps = find_gaps(df["timestamp"])
    if gaps.empty:
        log("  ✅ Sin gaps detectados")
    else:
        log(f"  ⚠️  {len(gaps)} gaps — {gaps['hours'].sum():,} horas perdidas")
        for _, row in gaps.head(10).iterrows():
            log(f"  {str(row['gap_start'])[:19]} → {str(row['gap_end'])[:19]} ({row['hours']}h)")

    return df["timestamp"]


def audit_overlap(ts_traffic, ts_air, ts_weather):
    section("4. SOLAPAMIENTO — Base para v_combined_hourly y el modelo ML")

    def to_set(ts):
        if ts is None or len(ts) == 0:
            return set()
        return set(ts.dt.floor("h").dropna().unique())

    t = to_set(ts_traffic)
    a = to_set(ts_air)
    w = to_set(ts_weather)
    all3 = t & a & w

    log(f"  Horas únicas tráfico  : {len(t):,}")
    log(f"  Horas únicas aire     : {len(a):,}")
    log(f"  Horas únicas meteo    : {len(w):,}")
    log()
    log(f"  Tráfico ∩ Aire        : {len(t & a):,} horas")
    log(f"  Tráfico ∩ Meteo       : {len(t & w):,} horas")
    log(f"  Aire    ∩ Meteo       : {len(a & w):,} horas")
    log()

    if all3:
        s = sorted(all3)
        days = (s[-1] - s[0]).days
        cov  = len(all3) / max((days + 1) * 24, 1) * 100
        log(f"  ✅ LAS 3 FUENTES      : {len(all3):,} horas solapadas")
        log(f"     Desde             : {s[0]}")
        log(f"     Hasta             : {s[-1]}")
        log(f"     Período           : {days} días")
        log(f"     Cobertura real    : {cov:.1f}%")
        if cov < 70:
            log()
            log("  ⚠️  Cobertura < 70% — revisa gaps antes de entrenar.")
        else:
            log()
            log("  ✅ Cobertura suficiente para entrenar el modelo.")
    else:
        log("  ❌ Sin solapamiento entre las 3 fuentes.")
        log("     Verifica que los rangos temporales coincidan.")


def main():
    traffic_only = "--traffic" in sys.argv
    air_only     = "--air"     in sys.argv
    weather_only = "--weather" in sys.argv
    all_tables   = not any([traffic_only, air_only, weather_only])

    log("=" * 65)
    log("  AUDITORÍA DE CALIDAD DE DATOS — Vitoria Air Quality")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 65)
    log(f"  Raíz: {ROOT_DIR}")

    ts_traffic = ts_air = ts_weather = None

    if all_tables or traffic_only:
        ts_traffic = audit_traffic()
    if all_tables or air_only:
        ts_air = audit_air()
    if all_tables or weather_only:
        ts_weather = audit_weather()
    if all_tables:
        audit_overlap(ts_traffic, ts_air, ts_weather)

    log(); log("=" * 65); log("  AUDITORÍA COMPLETADA"); log("=" * 65)

    report = ROOT_DIR / "data" / "analytical" / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n  Reporte guardado en: {report.name}")


if __name__ == "__main__":
    main()