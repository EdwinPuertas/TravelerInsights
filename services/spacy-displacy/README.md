---
title: Traveler Insights spaCy
emoji: 🌴
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Traveler Insights · spaCy (NER y dependencias con displaCy)

Servicio auxiliar de la app [Traveler Insights](https://traveler-insights.vercel.app). Recibe una reseña y devuelve:

- `ents_html`: entidades nombradas, con `displacy.render(doc, style="ent")`.
- `dep_html`: árbol de dependencias de la primera oración, con `displacy.render(doc, style="dep")`.
- `ents` y `tokens` en JSON: lema, categoría gramatical y dependencia.

```bash
curl -X POST https://<usuario>-traveler-insights-spacy.hf.space/analyze \
  -H "Content-Type: application/json" -d '{"text": "El hotel en Coveñas tenía una vista increíble"}'
```

Modelo: `es_core_news_sm` (spaCy 3.8). Variable `ALLOWED_ORIGINS` para CORS.
