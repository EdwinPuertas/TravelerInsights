# Fase 1 · Entendimiento del negocio y de los datos

## Objetivo
Reducir el tiempo que un operador turístico dedica a leer, clasificar y responder reseñas, y saber de qué aspectos se quejan los viajeros. Las respuestas deben seguir siendo correctas y responsables.

## Decisión que cambia
Qué reseñas se atienden automáticamente y cuáles revisa una persona, y en qué aspectos (limpieza, servicio, precio…) invertir primero.

## Métricas de éxito (se definen antes de modelar)
| Tipo | Métrica | Umbral inicial |
|---|---|---|
| Técnica | Macro-F1 en prueba | `quality_gate.min_macro_f1` en `configs/params.yaml` |
| Técnica | Mejora sobre la clase mayoritaria | `quality_gate.must_beat_dummy_by` |
| Negocio | % de reseñas automatizadas con exactitud aceptable | Curva τ en `reports/metrics.json` |
| Negocio | Horas de trabajo ahorradas por mes | Pestaña *Impacto* (supuestos de la organización) |

## Datos
Competencia de Kaggle *Traveler Insights UPTC 2026*: reseñas de viajeros con etiqueta. Se usa solo `train.csv`, porque `test.csv` no trae etiquetas.

## Riesgos y mitigación
| Riesgo | Mitigación |
|---|---|
| Automatizar respuestas incorrectas o con promesas indebidas | Umbral τ, guardarraíl, plantillas sin compromisos, revisión humana |
| Desempeño desigual entre tipos de reseña | Evaluación por subgrupos |
| Datos con licencia restringida | Datos fuera del repositorio; modelo en repositorio privado si las reglas lo exigen |

## Criterios de aceptación de la fase
- [ ] Métricas técnicas y de negocio acordadas y registradas en `params.yaml`.
- [ ] Riesgos documentados.
