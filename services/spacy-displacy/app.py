"""Servicio spaCy para Traveler Insights: NER y árbol de dependencias con displaCy.

Se despliega en Hugging Face Spaces (Docker). La app de Vercel lo llama desde el navegador.
Local:  uvicorn app:app --port 7860
"""
import os

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy import displacy

MODELO = os.environ.get("SPACY_MODEL", "es_core_news_sm")
nlp = spacy.load(MODELO)

app = FastAPI(title="Traveler Insights · spaCy displaCy")
origenes = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "https://traveler-insights.vercel.app").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origenes, allow_origin_regex=r"https://traveler-insights.*\.vercel\.app",
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class Texto(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)


@app.get("/")
def salud():
    return {"status": "ok", "model": MODELO, "pipeline": nlp.pipe_names}


@app.post("/analyze")
def analizar(t: Texto):
    doc = nlp(t.text)
    # Para el árbol se usa la primera oración: una reseña larga produce un SVG enorme
    oracion = next(iter(doc.sents), doc)
    return {
        "model": MODELO,
        "ents": [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents],
        "ents_html": displacy.render(doc, style="ent", page=False, jupyter=False),
        "dep_html": displacy.render(oracion.as_doc(), style="dep", page=False, jupyter=False,
                                    options={"compact": True, "distance": 110, "color": "#222222", "font": "system-ui"}),
        "tokens": [{"text": w.text, "lemma": w.lemma_, "pos": w.pos_, "dep": w.dep_, "head": w.head.i} for w in doc],
    }
