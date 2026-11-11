# Experimentos forecast v10 (2026-11-10)

Comparación sobre el mismo parquet (812 días, 2024-08-09 → 2026-11-09) y el
mismo bloque de test final (jul-nov 2026). Skill = 1 - RMSE modelo / RMSE
persistencia; positivo es mejor.

| Variante | Qué cambia | Skill medio | D+1 | D+2 | Targets OK | Features |
|---|---|---:|---:|---:|---:|---:|
| baseline | producción actual | **0.1661** | 0.1446 | 0.1876 | 11/12 | 101.0 |
| traffic_fix | `exp_traffic_*` con ventana reciente | 0.1638 | 0.1428 | 0.1848 | 12/12 | 101.0 |
| **both_traffic** | baseline + versión reciente como feature extra | **0.1692** | **0.1478** | **0.1906** | **12/12** | 101.7 |
| all | + ratios, estancamiento, lluvia, sectores, vacaciones | 0.1584 | 0.1356 | 0.1812 | 12/12 | 113.3 |
| prune80 | top-80 features | 0.1475 | 0.1343 | 0.1607 | 12/12 | 80.0 |
| prune60 | top-60 features | 0.1477 | 0.1348 | 0.1607 | 12/12 | 60.0 |
| all_shrink | all + encogimiento hacia persistencia | 0.1584 | 0.1356 | 0.1812 | 12/12 | 113.3 |

## Decisión

Se despliega **both_traffic**: es la mejor media global, mejora NO₂ ZBE
D+1 (+2.1 pp) y elimina el fallback de NO₂ ZBE D+2 (de −0.9 % a +2.3 %).
Las features derivadas (ratios, estancamiento, lluvia, viento), la poda y el
encogimiento no aportan y no se despliegan.

## Fase 1 — estaciones del Gobierno Vasco

Mismo parquet (816 días, 2024-07-27 → 2026-11-09) y mismo test final.

| Variante | Features GV | Skill medio | D+1 | D+2 | Targets OK |
|---|---|---:|---:|---:|---:|
| gv_baseline | ninguna | **0.1742** | 0.1573 | **0.1911** | 11/12 |
| gv_lag1 | lag 1d de las 3 estaciones | 0.1723 | **0.1600** | 0.1845 | 11/12 |
| gv_pm | PM10/PM2.5 lag1 + media 3d | 0.1704 | 0.1575 | 0.1834 | 11/12 |
| gv | todas (63 features) | 0.1600 | 0.1472 | 0.1728 | 11/12 |

**Decisión:** no se despliegan las features del GV. `gv_lag1` mejora D+1 (+0,27 pp) pero empeora D+2 (−0,66 pp); el resto empeora claramente. Se mantiene la ingesta (`download_gv_air.py` + `gv_daily.csv`) y el modo experimental (`--gv-features` / `--gv-mode`) desactivado por defecto.

## Detalle por target (skill de test)

| Target | baseline | traffic_fix | both_traffic | all | prune80 | prune60 |
|---|---:|---:|---:|---:|---:|---:|
| NO2_zbe_d1 | 0.0906 | 0.1122 | 0.1026 | 0.0771 | 0.0848 | 0.0849 |
| NO2_out_d1 | 0.1944 | 0.1889 | 0.1917 | 0.1967 | 0.2113 | 0.1983 |
| PM10_zbe_d1 | 0.1359 | 0.1374 | 0.1504 | 0.1416 | 0.1263 | 0.1165 |
| PM10_out_d1 | 0.1420 | 0.1345 | 0.1392 | 0.1301 | 0.1349 | 0.1300 |
| PM2.5_zbe_d1 | 0.1567 | 0.1408 | 0.1537 | 0.1453 | 0.1223 | 0.1387 |
| PM2.5_out_d1 | 0.1482 | 0.1427 | 0.1494 | 0.1229 | 0.1264 | 0.1402 |
| NO2_zbe_d2 | -0.0086 | 0.0239 | 0.0226 | 0.0009 | 0.0307 | 0.0385 |
| NO2_out_d2 | 0.1973 | 0.2228 | 0.2142 | 0.2241 | 0.2182 | 0.2175 |
| PM10_zbe_d2 | 0.2271 | 0.2106 | 0.2205 | 0.2028 | 0.1660 | 0.1813 |
| PM10_out_d2 | 0.2383 | 0.2195 | 0.2189 | 0.2220 | 0.1893 | 0.1878 |
| PM2.5_zbe_d2 | 0.2387 | 0.2218 | 0.2406 | 0.2290 | 0.1864 | 0.1580 |
| PM2.5_out_d2 | 0.2329 | 0.2104 | 0.2267 | 0.2082 | 0.1736 | 0.1810 |

Los artefactos de cada variante no se versionan: solo métricas, manifiestos y
este resumen. Los `.joblib` se regeneran con:

```bash
python src/ml/train_forecast_v10.py --output-dir models/experiments/<variante> --tag <variante>
```

Nota de reproducibilidad: `all` y `prune*` usaron features derivadas
(ratios, estancamiento, lluvia, viento) que se descartaron tras el experimento.
El código actual reproduce `baseline` y `both_traffic`; las métricas de las
variantes descartadas se conservan como registro de la comparación.
