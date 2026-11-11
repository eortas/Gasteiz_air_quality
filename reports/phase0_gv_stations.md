# Fase 0 — Estaciones del Gobierno Vasco en Vitoria-Gasteiz

Fecha: 2026-11-11
Objetivo: decidir si las 4 estaciones del GV aportan valor real antes de escribir la ingesta.

## Veredicto

**Sí, pero no las 4 ni para lo que se pensaba.**

- **Farmacia no mide PM10, PM2.5 ni NO2.** Solo O3 y meteorología. No puede ser la referencia de partículas del proyecto.
- **3 de Marzo, Los Herrán y Av. Gasteiz sí miden NO2, PM10 y PM2.5**, tienen serie desde 2019 y están fuera de la ZBE. Son las candidatas reales.
- Las tres son de tráfico/perímetro, no de fondo regional. Valen como features exógenas, donantes del control sintético y testigos de spillover, no como “fondo limpio”.
- La correlación con la red municipal actual es moderada (0,50-0,67): no son redundantes, pero tampoco revolucionan la señal. Hay que probarlas con A/B.

## Metadatos oficiales (API Open Data Euskadi)

API: `https://api.euskadi.eus/air-quality`
Sin autenticación. Especificación: `https://opendata.euskadi.eus/contenidos/recurso_tecnico/data_apirest/es_def/adjuntos/air-quality.json`

| Estación | ID | Dirección | Lat | Lon | Contaminantes | Perfil | ZBE |
|---|---:|---|---:|---:|---|---|---|
| FARMACIA | 85 | Paseo Universidad, 7. Fac. Farmacia UPV | 42,84010 | -2,67256 | O3, O3-8h, meteo | Fondo urbano / campus | Fuera (716 m del centro) |
| 3 DE MARZO | 75 | Plaza Tres de Marzo, s/n | 42,85605 | -2,66778 | CO, NO, NO2, NOX, PM10, PM2.5, SO2 | Urbana/tráfico | Fuera (1.106 m) |
| LOS HERRÁN | 77 | Parque Judizmendi | 42,84370 | -2,66123 | NO, NO2, NOX, PM10, PM2.5 | Tráfico | Fuera (901 m) |
| AV. GASTEIZ | 78 | Avda. Gasteiz, 97 | 42,85478 | -2,68071 | CO, NO, NO2, NOX, PM10, PM2.5, benceno, tolueno | Tráfico | Fuera (1.183 m) |

La ZBE de fase 1 tiene 0,64 km² (radio equivalente ~450 m). Las cuatro están a más de 700 m del centro, por lo que quedan fuera del perímetro. Confirmar con el polígono oficial antes de usarlas como donantes.

## Cobertura histórica

Consulta al endpoint diario, 2019-2026:

| Estación | Rango | Días | Cobertura |
|---|---|---:|---|
| FARMACIA | 2019-01-01 → 2026-08-24 (diario) | 2.833 | O3 99,9 %; sin PM/NO2 |
| 3 DE MARZO | 2019-01-01 → 2026-11-10 | 2.850 | Todos los parámetros ~100 % |
| LOS HERRÁN | 2019-01-01 → 2026-11-10 | 2.847 | PM ~100 %, NO2 97 % |
| AV. GASTEIZ | 2019-01-01 → 2026-11-10 | 2.851 | PM/NO2/CO ~100 % |

Solapamiento con el dataset actual (desde 2024-08): completo para las tres de tráfico.

Aviso: el endpoint **diario** de Farmacia tiene huecos después de agosto de 2026, aunque el **horario** sí llega al 11-11. Para features, agregar desde horario, no desde diario.

## Correlación con la red municipal

Días comunes 2024-2026:

| Estación GV | NO2 vs ZBE | NO2 vs OUT | PM10 vs ZBE | PM2.5 vs ZBE |
|---|---:|---:|---:|---:|
| 3 DE MARZO | 0,61 | 0,63 | 0,54 | 0,67 |
| LOS HERRÁN | 0,59 | 0,64 | 0,58 | 0,56 |
| AV. GASTEIZ | 0,55 | 0,58 | 0,60 | 0,53 |

Entre las tres GV: NO2 0,83-0,88; PM10 0,75-0,88; PM2.5 0,74-0,77. Hay colinealidad, así que no hace falta meter las tres a la vez. Empezar por una o dos.

## API: hallazgos de ingeniería

- **Actualización:** cada hora; última actualización del catálogo 11/11/2026.
- **Latencia real:** a las 09:38 UTC el último dato válido era de 07:00 UTC → **~2,5 h**.
- **Trampa importante:** las horas futuras del día actual devuelven `0`, no `null`. Hay que filtrar por `fecha <= ahora - 2 h`; si no, la media diaria sale artificialmente baja.
- **Zona horaria:** la API devuelve UTC (`Z`). Agregar a día local `Europe/Madrid` para que coincida con el pipeline.
- **Unidades:** NO2/PM10/PM2.5/SO2 en µg/m³; CO en mg/m³; O3 en µg/m³. Confirmar condiciones de referencia (normalizado vs ambiente) con el GV; la API no lo documenta.
- **Formatos:** API REST, CSV, XLSX y JSON. Hay históricos por año en el catálogo.
- **Licencia:** datos abiertos del GV; la ficha remite a “Información legal”. Open Data Euskadi usa habitualmente CC BY. Verificar atribución antes de publicar.

## Qué haría con cada una

| Estación | Uso recomendado | No usar para |
|---|---|---|
| FARMACIA | O3 y meteorología; futuro si se modela O3 | PM10/PM2.5/NO2 (no los mide) |
| 3 DE MARZO | Feature exógena de NO2/PM y donante norte | Fondo regional limpio |
| LOS HERRÁN | Spillover este + feature de tráfico | Control limpio de fondo |
| AV. GASTEIZ | Spillover oeste + feature de tráfico | Control limpio de fondo |

## Decisión Fase 0

**Go parcial:** implementar ingesta solo para **3 de Marzo, Los Herrán y Av. Gasteiz**. Farmacia se deja fuera del alcance PM/NO2; como mucho, O3 en una fase futura.

**Go/no-go para Fase 1:** descargar 2024-2026, agregar a día local, crear `gv_*` con lags 1/2/3/7 y medias móviles, y hacer A/B contra el mismo corte jul-nov. Solo se integra si mejora de forma consistente; no se tocan los targets ni las medias de zona.

## Siguiente paso propuesto

1. `src/ingestion/download_gv_air.py` con `requests`, checkpoint y reintentos.
2. Guardar `data/raw/gv/{station}_{year}.csv.gz`.
3. Agregar a diario local con filtro de horas futuras.
4. Añadir features `gv_*` al parquet.
5. A/B con `--output-dir models/experiments/gv` y comparar por target.

## Resultado Fase 1 (A/B, 2026-11-11)

Implementado `download_gv_air.py` y features `gv_*` (lags 1/2/3/7, medias 3/7, diff 1d).
Comparación sobre 816 días y el mismo test final:

| Variante | Skill medio | D+1 | D+2 |
|---|---:|---:|---:|
| baseline sin GV | **0.1742** | 0.1573 | **0.1911** |
| GV lag 1d | 0.1723 | **0.1600** | 0.1845 |
| GV PM (lag1 + media 3d) | 0.1704 | 0.1575 | 0.1834 |
| GV completo (63 features) | 0.1600 | 0.1472 | 0.1728 |

**Decisión:** no se activan las features GV en producción. La ingesta y el modo
experimental quedan disponibles, pero desactivados por defecto. El modelo
actual ya captura la señal vía estaciones propias y meteorología; añadir las GV
como features introduce más ruido que señal.
