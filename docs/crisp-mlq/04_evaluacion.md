# Fase 4 · Evaluación

**Código:** `src/traveler_insights/ml/evaluate.py` · **Evidencia:** `reports/metrics.json`, `reports/evaluation.md`

## Qué se evalúa
| Pregunta | Método |
|---|---|
| ¿Qué tan bueno es? | Macro-F1, exactitud, reporte por clase, matriz de confusión |
| ¿Qué tan seguro es el número? | IC 95 % por bootstrap del macro-F1 |
| ¿La mejora sobre el baseline es real? | Prueba exacta de McNemar frente a Naive Bayes (Dietterich, 1998) |
| ¿La confianza significa algo? | Calibración: ECE y curva de fiabilidad (Guo et al., 2017) |
| ¿Falla en algún grupo? | Macro-F1 por longitud de reseña |
| ¿Cuánto se puede automatizar? | Cobertura y exactitud por umbral τ |

## Compuerta de calidad (`params.yaml › quality_gate`)
- `min_macro_f1`: mínimo aceptable para el negocio.
- `must_beat_dummy_by`: mejora mínima sobre la clase mayoritaria.
- `max_regression_vs_production`: el nuevo modelo no puede ser peor que el desplegado.
- `max_parity_error`: el modelo JSON debe reproducir a scikit-learn.

Si alguna condición falla, el modelo no se exporta y el workflow termina en error.

## Pendiente sugerido
Pruebas de comportamiento al estilo CheckList (Ribeiro et al., 2020): invariancia ante nombres de lugares, negación, errores tipográficos y reseñas mixtas.
