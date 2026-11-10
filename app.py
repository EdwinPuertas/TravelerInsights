"""API y app web de Traveler Insights (entrypoint de Vercel).

Local:   uvicorn app:app --reload        → http://127.0.0.1:8000
Vercel:  detecta automáticamente la instancia FastAPI `app` en este archivo.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from traveler_insights.core.business import analizar
from traveler_insights.core.model import LinearTextModel

MODEL_PATH = ROOT / "model" / "model.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
MAX_TEXTS = 200

app = FastAPI(title="Traveler Insights API", version="1.0.0",
              description="Clasificación de reseñas, aspectos, respuesta sugerida y humano en el bucle.")


@lru_cache(maxsize=1)
def modelo() -> LinearTextModel:
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No hay modelo desplegado. Ejecute el pipeline de entrenamiento.")
    return LinearTextModel.load(MODEL_PATH)


class Peticion(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS, description="Reseñas a analizar")
    tau: float | None = Field(None, ge=0.0, le=1.0, description="Umbral de confianza; por defecto el del modelo")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def inicio():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def salud():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/api/model")
def info_modelo():
    m = modelo()
    out = {"classes": m.classes, "metadata": {k: v for k, v in m.metadata.items() if k != "columns"}}
    if METRICS_PATH.exists():
        met = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c = met["candidate"]
        out["evaluation"] = {"test_macro_f1": c["test"]["macro_f1"], "test_accuracy": c["test"]["accuracy"],
                             "bootstrap_ci": c["bootstrap_ci_macro_f1"], "ece": c["calibration"]["ece"],
                             "tau_curve": c["tau_curve"], "quality_gate": met["quality_gate"]}
    return out


@app.post("/api/analyze")
def analizar_resenas(p: Peticion):
    m = modelo()
    tau = p.tau if p.tau is not None else m.metadata.get("tau", 0.7)
    textos = [t[:5000] for t in p.texts]          # límite de longitud por reseña
    return {"tau": tau, "results": analizar(m, textos, tau)}
