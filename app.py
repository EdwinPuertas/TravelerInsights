"""API y app web de Traveler Insights (entrypoint de Vercel).

Local:   uvicorn app:app --reload        → http://127.0.0.1:8000
Vercel:  detecta automáticamente la instancia FastAPI `app` en este archivo.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from traveler_insights.core import feedback as fb
from traveler_insights.core import store, textmetrics
from traveler_insights.core.business import analizar
from traveler_insights.core.emotions import perfil
from traveler_insights.core.model import LinearTextModel

MODEL_PATH = ROOT / "model" / "model.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
MAX_TEXTS = 200

app = FastAPI(title="Traveler Insights API", version="1.4.0",
              description="Sentimiento, emociones, aspectos, respuesta sugerida, humano en el bucle y retroalimentación.")


@lru_cache(maxsize=1)
def modelo() -> LinearTextModel:
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No hay modelo desplegado. Ejecute el pipeline de entrenamiento.")
    return LinearTextModel.load(MODEL_PATH)


def _ahora() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class Peticion(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS, description="Reseñas a analizar")
    tau: float | None = Field(None, ge=0.0, le=1.0, description="Umbral de confianza; por defecto el del modelo")
    save: bool | None = Field(None, description="Guardar en el historial; por defecto solo análisis individuales")


class Feedback(BaseModel):
    id: str | None = None
    texto: str = Field(..., min_length=1, max_length=5000)
    clase_predicha: str
    clase_correcta: str
    confianza: float | None = None
    respuesta_sugerida: str | None = None
    respuesta_editada: str | None = Field(None, max_length=3000)
    comentario: str | None = Field(None, max_length=1000)


class ItemsFeedback(BaseModel):
    items: list[dict] = Field(..., max_length=5000)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def inicio():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def salud():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "store": store.configurado()}


@app.get("/api/config")
def configuracion():
    return {"store": store.configurado(), "spacy_url": os.environ.get("SPACY_SERVICE_URL", "").rstrip("/")}


@app.get("/api/model")
def info_modelo():
    m = modelo()
    out = {"classes": m.classes, "metadata": {k: v for k, v in m.metadata.items() if k != "columns"}}
    if METRICS_PATH.exists():
        met = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c = met["candidate"]
        t = c["test"]
        out["evaluation"] = {
            "test_macro_f1": t["macro_f1"],
            # En clasificación de una sola etiqueta, micro-F1 = exactitud
            "test_micro_f1": t.get("micro_f1", t["accuracy"]),
            "test_accuracy": t["accuracy"],
            "per_class": {k: v for k, v in t["per_class"].items() if k in m.classes},
            "bootstrap_ci": c["bootstrap_ci_macro_f1"], "ece": c["calibration"]["ece"],
            "tau_curve": c["tau_curve"], "quality_gate": met["quality_gate"]}
    return out


@app.post("/api/analyze")
def analizar_resenas(p: Peticion):
    m = modelo()
    tau = p.tau if p.tau is not None else m.metadata.get("tau", 0.7)
    textos = [t[:5000] for t in p.texts]
    resultados = analizar(m, textos, tau)
    for r in resultados:
        r["id"] = uuid.uuid4().hex[:12]
        r["fecha"] = _ahora()
        r["emociones"] = perfil(r["texto"])
    guardar = p.save if p.save is not None else len(textos) == 1
    guardado = False
    if guardar and store.configurado():
        try:
            for r in resultados:
                store.guardar_historial({k: r[k] for k in ("id", "fecha", "texto", "clase", "confianza", "decision",
                                                          "aspectos", "respuesta_sugerida")} |
                                        {"emocion": r["emociones"]["dominante"]})
            guardado = True
        except Exception:   # el análisis no debe fallar si el almacenamiento falla
            guardado = False
    return {"tau": tau, "stored": guardado, "results": resultados}


@app.get("/api/history")
def ver_historial(limit: int = 50):
    if not store.configurado():
        return {"configured": False, "items": []}
    return {"configured": True, "items": store.historial(min(max(limit, 1), 500))}


@app.post("/api/feedback")
def recibir_feedback(f: Feedback):
    m = modelo()
    if f.clase_correcta not in m.classes:
        raise HTTPException(422, f"clase_correcta debe ser una de {m.classes}")
    reg = f.model_dump() | {"fecha": _ahora(), "modelo": m.metadata.get("created_utc")}
    if f.respuesta_sugerida and f.respuesta_editada:
        reg["metricas_respuesta"] = textmetrics.todas(f.respuesta_sugerida, f.respuesta_editada)
    if not store.configurado():
        return {"stored": False, "registro": reg}
    store.guardar_feedback(reg)
    return {"stored": True, "registro": reg}


@app.get("/api/feedback/stats")
def stats_feedback():
    if not store.configurado():
        return {"configured": False}
    return {"configured": True, **fb.resumen(store.feedback(), modelo().classes)}


@app.post("/api/feedback/stats")
def stats_feedback_local(p: ItemsFeedback):
    """Calcula las mismas métricas sobre la retroalimentación guardada en el navegador."""
    return {"configured": False, **fb.resumen(p.items, modelo().classes)}
