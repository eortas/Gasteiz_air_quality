# Vitoria-Gasteiz Air Quality Intelligence

Análisis del impacto de la Zona de Bajas Emisiones (ZBE) sobre la calidad del aire
y sistema predictivo de episodios de contaminación.

## Estructura del proyecto
Ver `docs/architecture.md` para la descripción completa.

## Instalación
```bash
pip install -r requirements.txt
```

## Uso rápido
```bash
# Descarga de datos
python src/ingestion/download_traffic.py
python src/ingestion/download_air_quality.py

# Pipeline completo
python run_pipeline.py

# Dashboard
streamlit run app/dashboard/app.py

# API
python app/api/main.py
```
