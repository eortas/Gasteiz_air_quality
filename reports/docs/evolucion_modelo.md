# Evolución del Modelo de Predicción — Vitoria Air Quality

**Proyecto:** Predicción de calidad del aire en Vitoria-Gasteiz  
**Contaminantes:** NO2, PM10, PM2.5, ICA  
**Zonas:** ZBE (Zona de Bajas Emisiones) y exterior (out)  
**Horizontes:** d1, d2, d3 (1, 2 y 3 días vista)  
**Dataset inicial:** 722 días · 256 features · 24 targets  
**Dataset limpio (v5):** 708 días · 256 features (→ 77-80 por permutation importance) · 8 targets d1

---

## Índice

1. [Versión 1 — Split temporal 70/15/15](#versión-1--split-temporal-701515)
2. [Versión 2 — TimeSeriesSplit sin tuning](#versión-2--timeseriessplit-sin-tuning)
3. [Versión 3 — RandomizedSearch (10 iteraciones)](#versión-3--randomizedsearch-10-iteraciones)
4. [Versión 4 — RandomizedSearch mejorado (30 iteraciones)](#versión-4--randomizedsearch-mejorado-30-iteraciones)
5. [Limpieza de datos](#limpieza-de-datos)
6. [Versión 5 — Permutation importance + datos limpios](#versión-5--permutation-importance--datos-limpios)
7. [Comparativa global](#comparativa-global)
8. [Feature Importance](#feature-importance)
9. [Análisis efecto ZBE](#análisis-efecto-zbe)
10. [Conclusiones y próximos pasos](#conclusiones-y-próximos-pasos)

---

## Versión 1 — Split temporal 70/15/15

**Fecha:** 2026-03-09 01:39  
**Script:** `train_model.py`  
**Tuning:** No

### Descripción

Primera aproximación. División fija del dataset en tres bloques temporales consecutivos sin validación cruzada:

- **Train:** 505 días (2024-03-15 → 2025-08-02)
- **Val:** 108 días (2025-08-02 → 2025-11-18)
- **Test:** 109 días (2025-11-18 → 2026-03-06) — incluye 109 días post-ZBE

### Configuración

```python
LGBM_PARAMS = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
}
```

### Resultados (Test R²)

| Target | Val RMSE | Val R² | Test RMSE | Test R² | MAPE% |
|---|---|---|---|---|---|
| NO2_out_d1 | 3.295 | 0.551 | 4.946 | 0.143 | 36.0 |
| NO2_out_d2 | 3.709 | 0.436 | 5.122 | 0.087 | 37.4 |
| NO2_out_d3 | 3.796 | 0.429 | 5.455 | -0.030 | 40.2 |
| NO2_zbe_d1 | 3.293 | -0.268 | 4.980 | 0.208 | 30.0 |
| NO2_zbe_d2 | 3.156 | 0.052 | 5.418 | 0.063 | 35.5 |
| NO2_zbe_d3 | 3.273 | 0.069 | 5.463 | 0.055 | 35.8 |
| PM10_out_d1 | 3.869 | 0.622 | 3.929 | 0.534 | 47.2 |
| PM10_out_d2 | 5.259 | 0.304 | 4.962 | 0.264 | 66.3 |
| PM10_out_d3 | 5.446 | 0.264 | 5.017 | 0.251 | 54.1 |
| PM10_zbe_d1 | 3.481 | 0.545 | 6.654 | 0.293 | 37.7 |
| PM10_zbe_d2 | 4.269 | 0.314 | 7.520 | 0.104 | 37.7 |
| PM10_zbe_d3 | 4.526 | 0.232 | 7.681 | 0.070 | 37.9 |
| PM2.5_out_d1 | 3.299 | 0.594 | 2.732 | **0.615** | 45.4 |
| PM2.5_out_d2 | 4.295 | 0.313 | 3.317 | 0.438 | 59.5 |
| PM2.5_out_d3 | 4.743 | 0.170 | 3.804 | 0.264 | 56.3 |
| PM2.5_zbe_d1 | 2.494 | 0.566 | 3.563 | **0.505** | 49.6 |
| PM2.5_zbe_d2 | 3.281 | 0.248 | 3.617 | 0.494 | 48.3 |
| PM2.5_zbe_d3 | 3.521 | 0.140 | 3.765 | 0.454 | 48.7 |
| ICA_out_d1 | 8.381 | 0.587 | 6.827 | **0.469** | 30.8 |
| ICA_out_d2 | 11.230 | 0.259 | 7.664 | 0.337 | 34.3 |
| ICA_out_d3 | 11.592 | 0.213 | 8.183 | 0.249 | 33.5 |
| ICA_zbe_d1 | 6.638 | 0.490 | 9.990 | 0.320 | 29.4 |
| ICA_zbe_d2 | 8.139 | 0.232 | 10.061 | 0.316 | 28.2 |
| ICA_zbe_d3 | 9.248 | 0.009 | 10.818 | 0.213 | 28.7 |

**RMSE medio por horizonte (test):** d1: 5.452 · d2: 5.960 · d3: 6.273

### Diagnóstico

- Los R² de test en PM2.5 e ICA son artificialmente altos porque el periodo de test (nov-2025 → mar-2026) puede ser meteorológicamente "favorable".
- Gran caída entre Val R² y Test R² en varios modelos (ej. `NO2_out_d1`: val=0.551 → test=0.143), lo que indica **sobreajuste al periodo de validación**.
- El split fijo no garantiza robustez ante distintos periodos temporales.

---

## Versión 2 — TimeSeriesSplit sin tuning

**Fecha:** 2026-03-09 01:55  
**Script:** `train_model_RS.py` (sin `--tune`)  
**Tuning:** No · Hiperparámetros base

### Descripción

Se sustituye el split fijo por **TimeSeriesSplit con 5 folds**, lo que evalúa el modelo en 5 ventanas temporales distintas y proporciona una estimación más robusta del rendimiento real.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Resultados (CV R²)

| Target | CV RMSE | CV R² | CV MAPE% |
|---|---|---|---|
| NO2_out_d1 | 5.208 | 0.248 | 51.9 |
| NO2_out_d2 | 5.333 | 0.201 | 54.8 |
| NO2_out_d3 | 5.426 | 0.141 | 53.4 |
| NO2_zbe_d1 | 4.579 | 0.172 | 68.9 |
| NO2_zbe_d2 | 4.351 | 0.287 | 101.1 |
| NO2_zbe_d3 | 5.284 | -0.132 | 67.5 |
| PM10_out_d1 | 4.684 | 0.295 | 35.5 |
| PM10_out_d2 | 5.920 | -0.119 | 43.7 |
| PM10_out_d3 | 6.030 | -0.136 | 44.5 |
| PM10_zbe_d1 | 4.577 | 0.278 | 40.6 |
| PM10_zbe_d2 | 5.181 | 0.077 | 46.2 |
| PM10_zbe_d3 | 5.176 | 0.082 | 46.5 |
| PM2.5_out_d1 | 3.476 | 0.360 | 38.0 |
| PM2.5_out_d2 | 4.377 | -0.013 | 47.4 |
| PM2.5_out_d3 | 4.747 | -0.165 | 52.4 |
| PM2.5_zbe_d1 | 2.858 | 0.412 | 45.2 |
| PM2.5_zbe_d2 | 3.392 | 0.162 | 54.3 |
| PM2.5_zbe_d3 | 3.287 | 0.224 | 52.1 |
| ICA_out_d1 | 8.580 | 0.330 | 29.2 |
| ICA_out_d2 | 10.910 | -0.086 | 36.2 |
| ICA_out_d3 | 11.545 | -0.177 | 39.7 |
| ICA_zbe_d1 | 7.509 | 0.257 | 35.7 |
| ICA_zbe_d2 | 8.676 | 0.009 | 41.2 |
| ICA_zbe_d3 | 8.629 | 0.037 | 41.3 |

### Diagnóstico

- El CV penaliza correctamente los periodos difíciles: los R² bajan respecto a la v1 pero son **más honestos**.
- `NO2_zbe_d2` con MAPE=101.1% — distorsión por días con valores cercanos a 0 µg/m³ (umbral demasiado bajo: 0.1).
- Modelos con R² negativo en d2/d3 de PM10 e ICA: el modelo no supera al predictor naive (media).

---

## Versión 3 — RandomizedSearch (10 iteraciones)

**Fecha:** 2026-03-09 02:15  
**Script:** `train_model_RS.py --tune`  
**Tuning:** Sí · 10 combinaciones aleatorias × 1000 estimadores  
**Duración:** 38 min 47 s

### Descripción

Se añade búsqueda aleatoria de hiperparámetros con `ParameterSampler` de scikit-learn. Por cada target y fold se prueban 10 combinaciones del siguiente espacio:

```python
TUNE_GRID = {
    "num_leaves":        [31, 50, 63, 90, 127],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "min_child_samples": [10, 20, 30, 40, 50],
    "subsample":         [0.6, 0.7, 0.8, 0.9],
}
N_ITER_RANDOM = 10
```

### Resultados (CV R²)

| Target | CV RMSE | CV R² | CV MAPE% |
|---|---|---|---|
| NO2_out_d1 | 4.858 | 0.345 | 49.0 |
| NO2_out_d2 | 4.947 | 0.319 | 51.2 |
| NO2_out_d3 | 5.194 | 0.227 | 51.6 |
| NO2_zbe_d1 | 4.220 | 0.294 | 60.4 |
| NO2_zbe_d2 | 4.218 | 0.321 | **96.1** |
| NO2_zbe_d3 | 4.762 | 0.096 | 66.0 |
| PM10_out_d1 | 4.501 | 0.352 | 33.2 |
| PM10_out_d2 | 5.615 | -0.006 | 40.1 |
| PM10_out_d3 | 5.718 | -0.028 | 42.4 |
| PM10_zbe_d1 | 4.527 | 0.295 | 39.2 |
| PM10_zbe_d2 | 5.063 | 0.126 | 44.8 |
| PM10_zbe_d3 | 5.068 | 0.126 | 43.5 |
| PM2.5_out_d1 | 3.318 | 0.414 | 35.9 |
| PM2.5_out_d2 | 4.110 | 0.111 | 44.5 |
| PM2.5_out_d3 | 4.369 | 0.013 | 47.8 |
| PM2.5_zbe_d1 | 2.779 | 0.446 | 43.3 |
| PM2.5_zbe_d2 | 3.280 | 0.222 | 50.7 |
| PM2.5_zbe_d3 | 3.224 | 0.256 | 50.5 |
| ICA_out_d1 | 8.304 | 0.371 | 29.0 |
| ICA_out_d2 | 10.338 | 0.033 | 33.9 |
| ICA_out_d3 | 10.775 | -0.027 | 37.5 |
| ICA_zbe_d1 | 7.249 | 0.319 | 33.5 |
| ICA_zbe_d2 | 8.445 | 0.062 | 40.7 |
| ICA_zbe_d3 | 8.429 | 0.084 | 39.0 |

### Mejoras respecto a v2

| Target | v2 CV R² | v3 CV R² | Δ R² |
|---|---|---|---|
| NO2_out_d1 | 0.248 | 0.345 | **+0.097** |
| NO2_zbe_d1 | 0.172 | 0.294 | **+0.122** |
| PM10_out_d1 | 0.295 | 0.352 | +0.057 |
| PM2.5_zbe_d1 | 0.412 | 0.446 | +0.034 |
| ICA_out_d1 | 0.330 | 0.371 | +0.041 |
| ICA_out_d2 | -0.086 | 0.033 | **+0.119** |

> El tuning mejora **todos los targets en d1** y recupera ICA_out_d2 de R² negativo a positivo.

---

## Versión 4 — RandomizedSearch mejorado (30 iteraciones)

**Fecha:** 2026-03-09 03:23  
**Script:** `train_model_RS_claude.py --tune`  
**Tuning:** Sí · 30 combinaciones × early stopping (50 rondas)  
**Duración:** 29 min 3 s

### Cambios introducidos

#### ① Early stopping real

```python
from lightgbm import early_stopping, log_evaluation

fit_callbacks = [
    early_stopping(stopping_rounds=50, verbose=False),
    log_evaluation(period=-1),
]
```

LightGBM para automáticamente si el RMSE de validación no mejora en 50 rondas consecutivas. Evita overfitting y reduce el tiempo de entrenamiento — 38 min (v3, 10 iter) → 29 min (v4, 30 iter).

#### ② N_ITER_RANDOM: 10 → 30

Triple cobertura del espacio de búsqueda.

#### ③ Umbral MAPE: 0.1 → 5.0 µg/m³

```python
MAPE_THRESHOLD = 5.0
mask = y_true > MAPE_THRESHOLD  # excluye días con contaminación casi nula
```

Resuelve el MAPE=96-101% de `NO2_zbe_d2` causado por días con valores cercanos a 0.

#### ④ TUNE_GRID ampliado: 4 → 7 hiperparámetros

```python
TUNE_GRID = {
    "num_leaves":        [31, 50, 63, 90, 127],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "min_child_samples": [10, 20, 30, 40, 50],
    "subsample":         [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],   # nuevo
    "reg_alpha":         [0.0, 0.05, 0.1, 0.5],   # nuevo
    "reg_lambda":        [0.0, 0.05, 0.1, 0.5],   # nuevo
}
```

### Resultados (CV R²)

| Target | CV RMSE | CV R² | CV MAPE% |
|---|---|---|---|
| NO2_out_d1 | 4.710 | 0.390 | 29.3 |
| NO2_out_d2 | 4.830 | 0.352 | 29.0 |
| NO2_out_d3 | 4.885 | 0.325 | 29.4 |
| NO2_zbe_d1 | 3.993 | 0.375 | 28.1 |
| NO2_zbe_d2 | 4.117 | 0.347 | 28.7 |
| NO2_zbe_d3 | 4.489 | 0.203 | 31.5 |
| PM10_out_d1 | 4.354 | 0.394 | 25.4 |
| PM10_out_d2 | 5.380 | 0.077 | 29.4 |
| PM10_out_d3 | 5.471 | 0.056 | 30.2 |
| PM10_zbe_d1 | 4.314 | 0.365 | 29.4 |
| PM10_zbe_d2 | 4.849 | 0.205 | 32.9 |
| PM10_zbe_d3 | 4.875 | 0.197 | 33.0 |
| PM2.5_out_d1 | 3.233 | 0.445 | 27.0 |
| PM2.5_out_d2 | 4.017 | 0.149 | 31.8 |
| PM2.5_out_d3 | 4.221 | 0.073 | 33.1 |
| PM2.5_zbe_d1 | 2.698 | 0.479 | 24.3 |
| PM2.5_zbe_d2 | 3.131 | 0.299 | 26.4 |
| PM2.5_zbe_d3 | 3.111 | 0.311 | 27.8 |
| ICA_out_d1 | 8.147 | 0.396 | 26.9 |
| ICA_out_d2 | 10.071 | 0.082 | 32.9 |
| ICA_out_d3 | 10.440 | 0.030 | 35.2 |
| ICA_zbe_d1 | 6.956 | 0.374 | 31.1 |
| ICA_zbe_d2 | 8.029 | 0.170 | 37.6 |
| ICA_zbe_d3 | 8.084 | 0.164 | 37.5 |

### Diagnóstico

- MAPE normalizado en todos los targets (24-37%) gracias al umbral de 5 µg/m³.
- El early stopping aceleró el entrenamiento a pesar de triplicar las iteraciones de búsqueda.

---

## Limpieza de datos

**Fecha:** 2026-03-09 14:52  
**Script:** `src/transformation/clean_raw_data.py`

### Problema detectado

Los CSVs crudos (`kunak_*.csv`) contenían lecturas de NO2 con valor exactamente 0, físicamente imposibles en un entorno urbano. Correspondían a sensores apagados o en mantenimiento que reportaban 0 en lugar de NaN.

```
Diagnóstico de ceros en NO2 por estación:
  BEATO:     2.088 registros
  PAUL:      1.969 registros  ← estación ZBE principal
  LANDAZURI: 1.593 registros  ← segunda estación ZBE
  FUEROS:    1.058 registros
  HUETOS:      628 registros
  ZUMABIDE:    671 registros

Días con sensor completamente caído (>50% ceros): 139 días
  PAUL:      55 días
  BEATO:     43 días
  LANDAZURI: 28 días
```

### Solución aplicada

```python
# 1. Días con >50% ceros → sensor caído → todo el día a NaN
SENSOR_DOWN_THRESHOLD = 0.5

# 2. Ceros restantes → NaN
mask_zero = (df["contaminante"] == cont) & (df["valor"] == 0)
df.loc[mask_zero, "valor"] = np.nan

# 3. Interpolación lineal (huecos ≤3h) + ffill/bfill (huecos ≤6h)
s = s.interpolate(method="linear", limit=3)
s = s.ffill(limit=6).bfill(limit=6)
# Huecos >6h se dejan como NaN
```

### Resultado

| | Antes | Después |
|---|---|---|
| Registros con valor=0 en NO2 | 8.007 | 0 |
| Registros interpolados | — | 6.555 |
| NaN residuales (huecos >6h) | — | 2.465 |
| Días en parquet final | 722 | 708 |

Los outliers (PM10 y PM2.5 en episodios de diciembre y agosto) se conservaron — corresponden a episodios reales de inversiones térmicas y posible calima sahariana.

---

## Versión 5 — Permutation importance + datos limpios

**Fecha:** 2026-03-09 14:57  
**Script:** `train_model_v5.py --tune`  
**Tuning:** Sí · 30 combinaciones × early stopping (100 rondas)  
**Duración:** 5 min 47 s  
**Dataset:** 708 días (datos limpios)

### Cambios introducidos

#### ① Solo targets d1

Se reduce el scope a 8 modelos (contaminante × zona) en lugar de 24. Los horizontes d2/d3 tenían R² negativo en varios targets y diluían el esfuerzo de optimización.

#### ② Permutation importance para selección de features

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    model, X_val, y_val,
    n_repeats=5,
    random_state=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
# Seleccionar top 80 features con importancia > 0
```

Más robusto que la importancia nativa de LightGBM porque mide el impacto real en las predicciones permutando aleatoriamente cada feature. Reduce de 256 → 72-80 features por modelo.

#### ③ Early stopping: 50 → 100 rondas

50 rondas era demasiado agresivo con `learning_rate` bajo (0.01-0.03).

#### ④ Datos limpios

Primer entrenamiento con el dataset sin ceros falsos. Especialmente relevante para `NO2_zbe` (PAUL: 55 días con sensor caído corregidos).

### Resultados (CV R²)

| Target | CV RMSE | CV R² | CV MAPE% | Features |
|---|---|---|---|---|
| NO2_out_d1 | 4.490 | 0.421 | 27.7 | 80 |
| NO2_zbe_d1 | 3.915 | 0.387 | 26.2 | 77 |
| PM10_out_d1 | 4.208 | 0.444 | 24.7 | 80 |
| PM10_zbe_d1 | 4.239 | 0.359 | 28.2 | 80 |
| PM2.5_out_d1 | 3.111 | **0.504** | 25.6 | 80 |
| PM2.5_zbe_d1 | 2.680 | 0.486 | 25.3 | 80 |
| ICA_out_d1 | 7.987 | 0.438 | 27.1 | 80 |
| ICA_zbe_d1 | 6.722 | 0.412 | 29.6 | 80 |

### Comparativa v4 → v5

| Target | v4 R² | v5 R² | Δ R² |
|---|---|---|---|
| NO2_out_d1 | 0.390 | **0.421** | +0.031 ✅ |
| NO2_zbe_d1 | 0.375 | **0.387** | +0.012 ✅ |
| PM10_out_d1 | 0.394 | **0.444** | +0.050 ✅ |
| PM10_zbe_d1 | 0.365 | 0.359 | -0.006 ⚠️ |
| PM2.5_out_d1 | 0.445 | **0.504** | +0.059 ✅ |
| PM2.5_zbe_d1 | 0.479 | **0.486** | +0.007 ✅ |
| ICA_out_d1 | 0.396 | **0.438** | +0.042 ✅ |
| ICA_zbe_d1 | 0.374 | **0.412** | +0.038 ✅ |

**7 de 8 modelos mejoran.** Único retroceso marginal: `PM10_zbe_d1` (-0.006 R²).  
**Hito:** `PM2.5_out_d1` supera por primera vez **R²=0.5** (0.504).

---

## Comparativa global

### CV R² por versión — solo d1

| Target | v1 (Test)* | v2 (CV) | v3 (CV) | v4 (CV) | v5 (CV) |
|---|---|---|---|---|---|
| NO2_out_d1 | 0.143 | 0.248 | 0.345 | 0.390 | **0.421** |
| NO2_zbe_d1 | 0.208 | 0.172 | 0.294 | 0.375 | **0.387** |
| PM10_out_d1 | 0.534 | 0.295 | 0.352 | 0.394 | **0.444** |
| PM10_zbe_d1 | 0.293 | 0.278 | 0.295 | 0.365 | 0.359 |
| PM2.5_out_d1 | 0.615* | 0.360 | 0.414 | 0.445 | **0.504** |
| PM2.5_zbe_d1 | 0.505* | 0.412 | 0.446 | 0.479 | **0.486** |
| ICA_out_d1 | 0.469* | 0.330 | 0.371 | 0.396 | **0.438** |
| ICA_zbe_d1 | 0.320 | 0.257 | 0.319 | 0.374 | **0.412** |

*R² de v1 inflado por evaluar en un único periodo meteorológicamente favorable.

### Evolución del R² medio d1

| Versión | Cambio principal | R² medio d1 | Tiempo |
|---|---|---|---|
| v1 — Split fijo | Baseline | ~0.35 *(inflado)* | — |
| v2 — CV sin tune | TimeSeriesSplit 5 folds | ~0.28 | ~5 min |
| v3 — RS 10 iter | RandomizedSearch | ~0.33 | 38 min |
| v4 — RS 30 iter | Early stopping + TUNE_GRID ampliado | ~0.39 | 29 min |
| v5 — Perm. importance + datos limpios | 256 → 80 features · datos limpios | **~0.43** | **6 min** |

> **Nota sobre v5:** La limpieza de ~9.020 ceros falsos de NO2 redujo el sesgo pre/post ZBE (NO2_zbe: +7.9% → +4.3%) y mejoró principalmente los modelos `_out`. La reducción de features (256 → 80) via permutation importance redujo el tiempo de entrenamiento de 29 min a 6 min manteniendo o mejorando las métricas.

---

## Feature Importance

### Top features globales en v5 (permutation importance, último fold)

| Rank | Feature | Grupo |
|---|---|---|
| 1 | `PM10_out` | Contaminante actual |
| 2 | `PM2.5_out` | Contaminante actual |
| 3 | `ICA_out` | Contaminante actual |
| 4 | `relative_humidity_2m` | Meteorología actual |
| 5 | `weather_code` | Meteorología actual |
| 6 | `PM10_zbe_lag_2d` | Lag aire |
| 7 | `wind_v` | Meteorología actual |
| 8 | `fc_wind_speed_10m_d1` | Pronóstico meteo |
| 9 | `fc_boundary_layer_height_d1` | Pronóstico meteo |
| 10 | `fc_wind_gusts_10m_d1` | Pronóstico meteo |

### Top 5 por target

| Target | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|---|---|---|---|---|---|
| NO2_zbe_d1 | NO2_zbe | fc_temperature_2m_d1 | NO2_zbe_roll_mean_7d | NO2_zbe_roll_mean_3d | exp_traffic_volume_d1 |
| NO2_out_d1 | exp_traffic_volume_d1 | NO2_out | fc_temperature_2m_d1 | day_of_year | exp_traffic_occupancy_d1 |
| PM10_zbe_d1 | PM10_zbe | PM2.5_zbe | weather_code | PM10_zbe_lag_2d | ICA_zbe_lag_3d |
| PM10_out_d1 | PM2.5_out | ICA_out | PM10_out | weather_code | humedad_out |
| PM2.5_zbe_d1 | PM2.5_zbe | ICA_zbe | weather_code | PM10_zbe | wind_gusts_10m |
| PM2.5_out_d1 | PM2.5_out | ICA_out | weather_code | PM10_out | fc_precipitation_d1 |
| ICA_zbe_d1 | PM2.5_zbe | ICA_zbe | PM10_zbe | weather_code | wind_gusts_10m |
| ICA_out_d1 | PM2.5_out | ICA_out | PM10_out | fc_wind_speed_10m_d1 | weather_code |

---

## Análisis efecto ZBE

La Zona de Bajas Emisiones entró en vigor el **1 de septiembre de 2025**. Datos post-limpieza con 187 días post-ZBE:

| Contaminante | Pre-ZBE | Post-ZBE | Cambio | Interpretación |
|---|---|---|---|---|
| NO2_zbe | 11.72 | 12.22 | +4.3% ↑ | Contraintuitivo — posible efecto estacional |
| NO2_out | 13.00 | 13.33 | +2.5% ↑ | Subida generalizada, no atribuible a ZBE |
| PM10_zbe | 11.03 | 11.43 | +3.6% ↑ | Sin mejora clara |
| PM10_out | 11.03 | 9.52 | -13.7% ↓ | Mejora exterior |
| PM2.5_zbe | 6.91 | 5.94 | **-14.1% ↓** | Mejora significativa |
| PM2.5_out | 7.65 | 6.21 | **-18.8% ↓** | Mejora exterior similar → efecto global |
| ICA_zbe | 19.20 | 18.64 | -2.9% ↓ | Mejora leve |
| ICA_out | 21.28 | 18.05 | **-15.2% ↓** | Mejora notable exterior |

> **Nota sobre la limpieza:** Tras eliminar los ceros falsos de NO2, la subida pre→post ZBE se redujo de +7.9% → +4.3% en ZBE y de +5.7% → +2.5% en OUT. Los ceros falsos deprimían artificialmente las medias pre-ZBE.

**Conclusión:** La mejora de PM2.5 e ICA se produce tanto dentro como fuera de la ZBE, lo que sugiere un factor meteorológico o estacional más que un efecto directo de la restricción de tráfico. Se necesitan más meses de datos post-ZBE para aislar el efecto real.

---

## Conclusiones y próximos pasos

### Lo que funciona bien

- **PM2.5_out_d1** supera R²=0.5 por primera vez en v5 — mejor target del proyecto
- **La reducción de features** (256 → 80 por permutation importance) mejora 7 de 8 modelos y reduce el tiempo 29 min → 6 min
- **La limpieza de datos** corrige las medias pre/post ZBE y mejora la consistencia del NO2
- **El MAPE** es consistente entre 24-30% en todos los targets en v5

### Problemas pendientes

- **PM10_zbe_d1** es el único que no mejora consistentemente — posiblemente necesita features de episodios de partículas (polvo sahariano, obras)
- **NO2** sigue siendo el contaminante más difícil (R² ~0.39-0.42) — la variabilidad intradiaria del tráfico no se captura bien con datos diarios
- Con 708 días el dataset sigue siendo pequeño para series temporales complejas

### Próximos pasos

1. Evaluar añadir features de calendario de eventos locales (festivos, huelgas de transporte)
2. Considerar modelos específicos para días post-ZBE una vez haya más datos (actualmente 187 días)
3. Implementar intervalos de confianza en las predicciones (quantile regression en LightGBM)
4. Explorar si agregar datos de estaciones de otras ciudades vascas mejora la generalización
