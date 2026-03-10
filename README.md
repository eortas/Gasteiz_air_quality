# Vitoria-Gasteiz Air Quality Intelligence

This repository contains the pipeline and analytical models to evaluate the impact of the Low Emission Zone (ZBE) in Vitoria-Gasteiz. It provides tools for both predictive operational monitoring and rigorous academic causal inference.

## Overview

The project integrates multiple data sources (traffic, meteorology, air quality) into an automated daily pipeline. It builds predictive models to estimate counterfactual scenarios—what pollution levels would have been observed in the absence of the ZBE—and performs Difference-in-Differences (DiD) Event Studies and Synthetic Control methods.

### Key Components

- **Data Ingestion & Transformation:** Automated download and processing of traffic, weather, and air quality metrics.
- **Feature Engineering:** Construction of temporal and spatial features for machine learning models.
- **Predictive Modeling (LightGBM):** High-precision daily forecasting of NO₂, PM10, and PM2.5 levels.
- **Causal Analysis:** Academic-grade causality tests to evaluate the effectiveness of the ZBE policy.
- **Interactive Dashboard:** Auto-generated operational dashboard displaying predictions, model audits, and map views.

## Installation

Ensure you have Python 3.11 installed, then install the dependencies:

```bash
pip install -r requirements.txt
playwright install chromium --with-deps
```

## Usage

You can execute the entire processing pipeline, which handles ingestion, modeling, causal plots, and HTML dashboard generation automatically:

```bash
python run_pipeline.py
```

### Additional Options

- Skip model training (uses existing models):
  ```bash
  python run_pipeline.py --skip-training
  ```
- Run exclusively with local fallback data (disables Supabase integration):
  ```bash
  python run_pipeline.py --local-only
  ```

## Continuous Integration

An automated GitHub Actions workflow (`.github/workflows/daily_pipeline.yml`) runs the full pipeline on a daily schedule, publishing the updated `index.html` operational dashboard via GitHub Pages.
