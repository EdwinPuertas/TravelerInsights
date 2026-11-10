# Fase 2 · Preparación de datos

**Código:** `src/traveler_insights/ml/data.py` · **Evidencia:** `reports/data_quality.json`

## Qué se hace
1. Detección o configuración de las columnas de texto y etiqueta.
2. Eliminación de vacíos.
3. Eliminación de duplicados, comparando el texto normalizado.
4. Eliminación de textos idénticos con etiquetas contradictorias.
5. Huella SHA-256 del archivo, para trazabilidad.

## Requisitos de calidad (`params.yaml › data.quality`)
| Requisito | Por qué |
|---|---|
| `min_rows` | Con pocos datos la evaluación no es confiable |
| `min_per_class` | Sin ejemplos suficientes, el F1 de una clase es ruido |
| `max_duplicate_rate` | Los duplicados inflan las métricas (fuga de información) |
| `max_conflict_rate` | Las etiquetas contradictorias indican problemas de anotación |

Si alguno falla, el pipeline se detiene y no se entrena.

## Criterios de aceptación
- [ ] Todos los controles en ✔.
- [ ] Distribución de clases y razón de desbalance revisadas.
