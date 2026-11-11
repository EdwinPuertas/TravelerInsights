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

## v1.4 · Historial, emociones y spaCy
- `/api/analyze` agrega el perfil de emociones (léxico SEL + NRC) y guarda la reseña en el historial (Upstash Redis).
- La app muestra NER y árbol de dependencias con displaCy a través de un servicio en Hugging Face Spaces (`services/spacy-displacy`), porque spaCy no cabe en una función de Vercel.
