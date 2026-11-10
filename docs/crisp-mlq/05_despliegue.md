# Fase 5 · Despliegue

**Código:** `src/traveler_insights/ml/export.py`, `src/traveler_insights/core/`, `app.py`, `web/`

## Decisiones
- **Modelo como JSON + inferencia en Python puro:** la función de Vercel solo instala `fastapi`. La paridad con scikit-learn se verifica en cada ejecución del pipeline y en CI.
- **Humano en el bucle:** se envían a revisión las predicciones con confianza menor que τ y las respuestas con compromisos.
- **Revisión humana del despliegue:** el modelo llega a producción solo al fusionar un Pull Request.
- **Ficha del modelo:** `model/MODEL_CARD.md` documenta el uso previsto, las limitaciones y los datos.

## Criterios de aceptación
- [ ] CI en verde (pruebas de paridad, API y contrato del modelo).
- [ ] Vista previa de Vercel revisada en el PR.
- [ ] `MODEL_CARD.md` actualizado.
