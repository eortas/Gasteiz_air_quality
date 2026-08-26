# Ficha técnica — forecast v10

## Estado

**Aprobado para un piloto institucional controlado.** Los doce objetivos D+1 y D+2 superan los controles internos de rendimiento y cobertura. Esta aprobación se limita al modelo técnico: aún requiere validación externa, monitorización en producción y los controles operativos, jurídicos y de seguridad de la institución que lo despliegue.

Versión: `forecast_v10`  
Entrenamiento: 21 de agosto de 2026  
Datos: 708 días, del 9 de agosto de 2024 al 28 de julio de 2026  
Ámbito: Vitoria-Gasteiz, grupos de estaciones ZBE y exterior

## Uso previsto

El sistema predice la concentración media de los días D+1 y D+2 de NO₂, PM10 y PM2.5 en µg/m³. El ICA se calcula después mediante subíndices CAQI, sin un modelo adicional. D+1 es el horizonte operativo principal y D+2 se presenta como apoyo a la planificación, con intervalos más amplios.

La predicción se ejecuta después de las 22:00, hora local de Vitoria-Gasteiz. Las variables de calidad del aire del día D solo incluyen observaciones disponibles hasta ese corte; los objetivos D+1 y D+2 representan días locales completos. La meteorología histórica procede de pronósticos emitidos uno y dos días antes, respectivamente. El contrato queda registrado en `data/processed/feature_contract.json`.

No debe utilizarse para atribuir causalidad, sancionar, certificar cumplimiento normativo ni sustituir una medición oficial. El análisis causal de la ZBE pertenece a un pipeline separado.

## Modelo

Cada objetivo usa un ensemble de corrección sobre persistencia:

1. La concentración conocida de D actúa como predicción base.
2. LightGBM, Extra Trees y Ridge estiman el cambio esperado para D+1.
3. Los pesos no negativos se seleccionan con ventanas temporales y su suma no supera uno.
4. Si un objetivo no supera el control de calidad, producción usa persistencia automáticamente.

Se emplean 105 variables operativas por objetivo: concentraciones recientes, rezagos y medias móviles, tráfico, meteorología observada y prevista, cobertura de estaciones y calendario conocido de D+1. No se usan columnas objetivo como entrada.

Las explicaciones locales suman las contribuciones ponderadas de los tres componentes. Son explicaciones predictivas del modelo, no efectos causales.

## Validación

La validación respeta el orden temporal:

- Cinco ventanas móviles de 45 días, con una separación igual al horizonte: un día para D+1 y dos para D+2.
- Las dos primeras ventanas calibran los pesos; las tres últimas estiman el rendimiento de backtesting.
- Un bloque final de 105 días queda fuera de la selección y se usa como test final.
- La referencia mínima es persistencia: usar el valor de D como pronóstico de D+1.
- Los intervalos se calibran con el cuantil conformal conservador del 95 % y se presentan como intervalos nominales del 90 %.

| Objetivo | RMSE backtest | Mejora vs. persistencia | RMSE test | MAE test | R² test | Mejora test | Cobertura IC90 test |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO₂ ZBE | 2,097 | 24,6 % | 1,564 | 1,210 | 0,566 | 8,5 % | 98,1 % |
| NO₂ exterior | 2,204 | 30,4 % | 2,799 | 2,033 | 0,509 | 5,8 % | 92,4 % |
| PM10 ZBE | 4,380 | 14,5 % | 4,533 | 3,609 | 0,472 | 16,1 % | 97,1 % |
| PM10 exterior | 3,326 | 12,5 % | 3,132 | 2,559 | 0,562 | 16,6 % | 98,1 % |
| PM2.5 ZBE | 2,739 | 17,4 % | 2,672 | 2,119 | 0,539 | 17,8 % | 98,1 % |
| PM2.5 exterior | 2,486 | 12,6 % | 2,486 | 2,009 | 0,541 | 16,4 % | 98,1 % |

### Horizonte D+2

| Objetivo | RMSE backtest | Mejora vs. persistencia | RMSE test | MAE test | R² test | Mejora test | Cobertura IC90 test |
|---|---:|---:|---:|---:|---:|---:|---:|
| NO₂ ZBE | 2,359 | 23,4 % | 1,769 | 1,403 | 0,444 | 12,4 % | 99,0 % |
| NO₂ exterior | 2,364 | 35,1 % | 3,552 | 2,469 | 0,253 | 13,1 % | 93,3 % |
| PM10 ZBE | 5,629 | 19,3 % | 5,778 | 4,615 | 0,172 | 21,7 % | 96,2 % |
| PM10 exterior | 4,221 | 14,4 % | 4,103 | 3,412 | 0,280 | 22,6 % | 96,2 % |
| PM2.5 ZBE | 3,788 | 18,7 % | 3,492 | 2,980 | 0,219 | 23,7 % | 99,0 % |
| PM2.5 exterior | 3,041 | 18,6 % | 3,253 | 2,731 | 0,220 | 22,9 % | 96,2 % |

El test cubre aproximadamente del 10/11 de abril al 28 de julio de 2026. Las métricas completas, pesos, periodos y variables principales se conservan en `models/forecast_v10_metrics.json`.

## Criterios de aceptación

Un objetivo solo utiliza el ensemble si cumple simultáneamente:

- mejora de RMSE en backtesting de al menos 3 % frente a persistencia;
- mejora no negativa en el test final;
- cobertura del intervalo nominal del 90 % de al menos 85 %.

Resultado actual: **12 de 12 objetivos aprobados**.

## Operación y monitorización

Para un despliegue institucional se recomienda:

- comprobar en cada ejecución el corte horario, la fecha de los datos, las variables ausentes y la versión del modelo;
- ejecutar `src/ml/validate_forecast_v10.py` antes de publicar; el pipeline ya aborta si falla;
- guardar predicción, intervalo, datos de origen y versión en un registro inmutable;
- calcular mensualmente RMSE, MAE, mejora frente a persistencia y cobertura con ventanas de 30 y 90 días;
- activar persistencia y una alerta si la mejora reciente es nula o negativa, faltan datos críticos o la cobertura cae por debajo del 85 %;
- reentrenar después de cambios en estaciones, sensores, fuentes meteorológicas o patrones de movilidad, y repetir todo el test temporal antes de promover una versión;
- someter el servicio a una validación externa durante el piloto antes de usarlo en decisiones públicas.

## Limitaciones y riesgo residual

- El histórico cubre menos de dos años y una sola ciudad; no demuestra generalización geográfica ni ante cambios estructurales futuros.
- Episodios poco frecuentes como polvo sahariano, incendios, obras o averías de sensores pueden quedar fuera del patrón aprendido.
- Los intervalos expresan incertidumbre empírica reciente; no garantizan que todos los valores futuros queden cubiertos.
- La agregación por grupos de estaciones puede ocultar diferencias locales dentro de cada zona.
- El modelo produce medias diarias D+1 y D+2, no alertas horarias ni una medición reglamentaria.
- D+2 tiene errores absolutos e intervalos mayores; debe comunicarse como perspectiva de planificación y no con la misma confianza que D+1.

La propuesta comercial defendible es un servicio de apoyo a la decisión con trazabilidad, intervalos y fallback, inicialmente bajo piloto con supervisión humana; no una promesa de exactitud absoluta.
