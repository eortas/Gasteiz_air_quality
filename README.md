# Vitoria-Gasteiz Air Quality Intelligence

**[Euskara](#euskara)** | **[Castellano](#castellano)** | **[English](#english)**

<a id="euskara"></a>

Biltegi honek Gasteizko Emisio Garbiagoen Eremuaren (EGE) eragina ebaluatzeko datu-kanalizazioa (pipeline) eta eredu analitikoak ditu. Jarraipen operatibo iragarlerako eta inferentzia kausal akademiko zorrotzerako tresnak eskaintzen ditu.

### Ikuspegi orokorra

Proiektuak hainbat datu-iturri (trafikoa, meteorologia, airearen kalitatea) eguneroko prozesu automatizatu batean integratzen ditu. Eredu iragarleak eraikitzen ditu agertoki kontrafaktualak zenbatesteko (EBErik gabe zein kutsadura-maila behatuko liratekeen argitzeko), eta *Difference-in-Differences* (DiD) *Event Studies* zein *Synthetic Control* metodoak exekutatzen ditu.

### Osagai nagusiak

- **Datuen xurgapena eta eraldaketa:** Trafiko, eguraldi eta airearen kalitate-metriken deskarga eta prozesatze automatizatua.
- **Ezaugarrien ingeniaritza:** Ikaskuntza automatikoko eredularitzarako denbora- eta espazio-ezaugarriak eraikitzea.
- **Eredu iragarleak (LightGBM):** NO₂, PM10 eta PM2.5 mailak doitasun handiz iragartzea egunero.
- **Errore-zuzenketarako Meta-Eredua (Ridge Regression):** Iragarpen-geruza gehigarri bat, eredu nagusiaren alborapen sistematikoetatik ikasten duena eguneroko aurreikuspenak fintzeko.
- **Azterketa Kausala:** EBE-aren eraginkortasuna ebaluatzeko maila akademikoko kausalitate-probak.
- **Aginte-taula interaktiboa:** Iragarpenak, ereduen auditoretzak eta mapa-ikuspegiak erakusten dituen irakurketa-taula operatibo automatikoa.
- **Trafiko-Mapa Berria:** Sentsore interaktiboak, eguneko bolumen-totalak eta **karga-gailurraren** ordua barne.
- **Elebitasuna (Euskara/Castellano):** Dashboard guztiak bi hizkuntzatan daude eskuragarri, Ikurrinadun hautatzaile baten bidez eta lehentasuna `localStorage` bidez gordez.

### Instalazioa

Ziurtatu Python 3.11 instalatuta duzula, ondoren mendekotasunak instalatu:

```bash
pip install -r requirements.txt
playwright install chromium --with-deps
```

### Erabilera

Prozesatze-kanalizazio osoa exekutatu dezakezu. Horrek xurgapena, modelizazioa, analisi kausalak eta HTML aginte-taularen sorkuntza automatikoki kudeatzen ditu:

```bash
python run_pipeline.py
```

#### Aukera gehigarriak

- Ereduaren trebakuntza saihestu (lehendik dauden ereduak erabiliz):
  ```bash
  python run_pipeline.py --skip-training
  ```

### Integrazio Etengabea

GitHub Actions-eko lan-fluxu automatizatu batek (`.github/workflows/daily_pipeline.yml`) kanalizazio osoa exekutatzen du egunero, eta eguneratutako `index.html` aginte-taula operatiboa argitaratzen du *GitHub Pages* bidez.

---

<a id="castellano"></a>

Este repositorio contiene la canalización de datos (pipeline) y los modelos analíticos para evaluar el impacto de la Zona de Gran Afluencia (ZBE) en Vitoria-Gasteiz. Ofrece herramientas tanto para el monitoreo operativo predictivo como para la inferencia causal académica rigurosa.

### Resumen General

El proyecto integra múltiples fuentes de datos (tráfico, meteorología, calidad del aire) en un proceso diario automatizado. Construye modelos predictivos para estimar escenarios contrafactuales —qué niveles de contaminación se habrían observado si no se hubiera implementado la ZBE— y ejecuta métodos de *Difference-in-Differences* (DiD) *Event Studies* y *Synthetic Control*.

### Componentes Principales

- **Ingesta y Transformación de Datos:** Descarga y procesamiento automatizado de métricas de tráfico, meteorología y calidad del aire.
- **Ingeniería de Características:** Construcción de variables temporales y espaciales para el modelado con aprendizaje automático.
- **Modelado Predictivo (LightGBM):** Predicción diaria de alta precisión de los niveles de NO₂, PM10 y PM2.5.
- **Meta-Modelo de Corrección de Errores (Ridge Regression):** Una capa predictiva adicional que aprende de los sesgos sistemáticos del modelo principal para refinar los pronósticos diarios.
- **Análisis Causal:** Pruebas de causalidad de nivel académico para evaluar la efectividad de la política de la ZBE.
- **Cuadro de Mando Interactivo:** Panel operativo autogenerado que muestra predicciones, auditorías de los modelos y vistas de mapa.
- **Nuevo Mapa de Tráfico:** Mapa interactivo de sensores que incluye totales de volumen diario e identificación del **pico de carga** horaria.
- **Soporte Bilingüe (Euskara/Castellano):** Todos los dashboards están totalmente traducidos, con un selector (con icono de Ikurriña) y persistencia de preferencia mediante `localStorage`.

---

<a id="english"></a>

This repository contains the pipeline and analytical models to evaluate the impact of the Low Emission Zone (ZBE) in Vitoria-Gasteiz. It provides tools for both predictive operational monitoring and rigorous academic causal inference.

### Overview

The project integrates multiple data sources (traffic, meteorology, air quality) into an automated daily pipeline. It builds predictive models to estimate counterfactual scenarios—what pollution levels would have been observed in the absence of the ZBE—and performs Difference-in-Differences (DiD) Event Studies and Synthetic Control methods.

### Key Components

- **Data Ingestion & Transformation:** Automated download and processing of traffic, weather, and air quality metrics.
- **Feature Engineering:** Construction of temporal and spatial features for machine learning models.
- **Predictive Modeling (LightGBM):** High-precision daily forecasting of NO₂, PM10, and PM2.5 levels.
- **Error-Correction Meta-Model (Ridge Regression):** An additional predictive layer that learns from the primary model's systematic biases to refine daily forecasts.
- **Causal Analysis:** Academic-grade causality tests to evaluate the effectiveness of the ZBE policy.
- **Interactive Dashboard:** Auto-generated operational dashboard displaying predictions, model audits, and map views.
- **New Traffic Map:** Interactive sensor map with daily volume totals and **peak load hour** identification.
- **Bilingual Support (Euskara/Castellano):** All dashboards fully translated, featuring a language toggle (with Ikurriña icon) and persistent settings via `localStorage`.

### Continuous Integration

An automated GitHub Actions workflow (`.github/workflows/daily_pipeline.yml`) runs the full pipeline on a daily schedule, publishing the updated `index.html` operational dashboard via GitHub Pages.

