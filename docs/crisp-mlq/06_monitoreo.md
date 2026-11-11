# Fase 6 · Monitoreo y mantenimiento

**Código:** `.github/workflows/monitor.yml`, `.github/workflows/train.yml`

## Implementado
- Prueba de humo diaria: salud de la API y una predicción de control.
- Reentrenamiento manual u opcionalmente programado, con la misma compuerta de calidad y revisión por PR.
- Reversión: revertir el commit del modelo en `main` (o *Instant Rollback* en Vercel).

## Siguiente paso recomendado
Medir **deriva** sin guardar textos personales. Por ejemplo, registrar de forma anónima la distribución de clases, de confianza y la tasa de envío a revisión humana, y comparar semanalmente con `reports/metrics.json`. Requiere una base de datos externa, porque las funciones de Vercel no tienen disco persistente.

## Criterios para reentrenar
- Caída de la exactitud en auditorías por muestreo.
- Cambio sostenido en la distribución de clases o de confianza.
- Nuevos datos etiquetados disponibles.

## v1.4 · Retroalimentación humana
- Cada corrección (clase correcta, respuesta editada) se guarda en `ti:feedback`.
- Métricas de producción: micro-F1 (= exactitud) y macro-F1 frente a las correcciones; BLEU y ROUGE-1/2/L entre respuesta sugerida y editada.
- `train.yml` exporta el feedback a `data/feedback/feedback.csv`; el pipeline lo agrega **solo al entrenamiento**, excluye textos que estén en la prueba y el modelo vuelve a pasar la compuerta y la revisión del PR.
- Riesgos: pocas correcciones, correcciones de una sola clase o malintencionadas. Revise `reports/` antes de fusionar.
