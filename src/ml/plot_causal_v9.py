import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuración de estilo
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# Rutas
MODELS_DIR = Path("models")
ES_FILE = MODELS_DIR / "event_study_v9.json"
SC_FILE = MODELS_DIR / "synthetic_control_v9.json"
PLOTS_DIR = Path("reports/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_event_study():
    if not ES_FILE.exists():
        print(f"No se encontró {ES_FILE}")
        return

    with open(ES_FILE, "r") as f:
        data = json.load(f)

    for cont, results in data.items():
        if not results:
            continue
        
        coefs = results["coefficients"]
        months = sorted([int(k) for k in coefs.keys()])
        
        betas = [coefs[str(m)]["beta"] for m in months]
        ci_low = [coefs[str(m)]["ci_low"] for m in months]
        ci_high = [coefs[str(m)]["ci_high"] for m in months]
        yerr = [
            np.array(betas) - np.array(ci_low),
            np.array(ci_high) - np.array(betas)
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Puntos y barras de error
        ax.errorbar(months, betas, yerr=yerr, fmt='o', color='#2c3e50', 
                    ecolor='#34495e', elinewidth=2, capsize=4, markersize=8)
        
        # Líneas de referencia
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(-1, color='#e74c3c', linestyle='--', linewidth=1.5, label='Implementación ZBE (Ref: -1)')
        
        # Sombreado para el periodo post-tratamiento
        ax.axvspan(-1, max(months), color='#e74c3c', alpha=0.05)

        ax.set_title(f"Event Study DiD: Impacto Dinámico de la ZBE en {cont}", pad=15, fontweight='bold')
        ax.set_xlabel("Meses relativos a la implementación de la ZBE")
        ax.set_ylabel(f"Efecto estimado sobre {cont} ($\mu g/m^3$)")
        ax.set_xticks(months)
        ax.legend()

        plt.tight_layout()
        cont_save = cont.replace("PM2.5", "PM25")
        out_path = PLOTS_DIR / f"event_study_{cont_save}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Guardado: {out_path}")
        plt.close()

def plot_synthetic_control():
    if not SC_FILE.exists():
        print(f"No se encontró {SC_FILE}")
        return

    with open(SC_FILE, "r") as f:
        data = json.load(f)

    for cont, stations in data.items():
        for station, results in stations.items():
            series = results["series"]
            
            # Reconstruir DataFrame
            df = pd.DataFrame({
                "date": pd.to_datetime(series["dates"]),
                "observed": series["observed"],
                "synthetic": series["synthetic"]
            })
            
            # Suavizado mensual para visualización clara
            df.set_index("date", inplace=True)
            df_smooth = df.resample('W').mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df_smooth.index, df_smooth["observed"], label=f"{station} (Observado)", 
                    color='#2980b9', linewidth=2, alpha=0.9)
            ax.plot(df_smooth.index, df_smooth["synthetic"], label=f"Synthetic {station}", 
                    color='#e67e22', linewidth=2, linestyle='--')
            
            # Línea de implementación
            zbe_date = pd.Timestamp("2025-09-01")
            ax.axvline(zbe_date, color='#c0392b', linestyle=':', linewidth=2, label='Implementación ZBE')
            
            # Sombreado de la brecha (Gap) post-ZBE
            df_post = df_smooth[df_smooth.index >= zbe_date]
            ax.fill_between(df_post.index, df_post["observed"], df_post["synthetic"], 
                            where=(df_post["observed"] < df_post["synthetic"]), 
                            color='#2ecc71', alpha=0.3, label='Reducción atribuible')
            ax.fill_between(df_post.index, df_post["observed"], df_post["synthetic"], 
                            where=(df_post["observed"] > df_post["synthetic"]), 
                            color='#e74c3c', alpha=0.3, label='Aumento')

            # Formateo
            ax.set_title(f"Control Sintético: {station} vs Synthetic {station} ({cont})", pad=15, fontweight='bold')
            ax.set_ylabel(f"Concentración de {cont} ($\mu g/m^3$)")
            ax.legend(loc='upper left')

            plt.tight_layout()
            cont_save = cont.replace("PM2.5", "PM25")
            out_path = PLOTS_DIR / f"synthetic_control_{cont_save}_{station}.png"
            plt.savefig(out_path, dpi=300)
            print(f"Guardado: {out_path}")
            plt.close()

if __name__ == "__main__":
    plot_event_study()
    plot_synthetic_control()