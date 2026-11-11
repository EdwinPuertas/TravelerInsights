"""Análisis de emociones con el léxico afectivo de los modelos previos del curso.

Léxico: ``lexicon_afectivo_categorias.csv`` del catálogo ``open_corpus_co_es``
(categorías SEL y NRC en español; 6.625 entradas). Se normaliza igual que el texto
(minúsculas, sin tildes) y se agrupa en 8 emociones: las 6 de SEL (Repulsión → disgusto)
más confianza y expectativa de NRC.

Incluye un respaldo morfológico simple (plural y femenino → forma del léxico).
Es un método por conteo léxico: no entiende negación ni ironía ("no estoy feliz" suma alegría).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .text import normalizar

_DATA = Path(__file__).parent / "data" / "emociones.json"
_PAL = re.compile(r"\w+")


@lru_cache(maxsize=1)
def _lexico():
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    simples, frases = {}, []
    for w, emos in data["words"].items():
        if " " in w:
            frases.append((re.compile(r"\b" + re.escape(w) + r"\b"), w, emos))
        else:
            simples[w] = emos
    return data["emotions"], simples, frases


EMOCIONES = json.loads(_DATA.read_text(encoding="utf-8"))["emotions"]


def _variantes(w: str) -> list[str]:
    """Respaldo morfológico simple (el léxico está mayormente en masculino singular): sucias → sucia → sucio."""
    out = [w]
    if len(w) > 4 and w.endswith("es"):
        out.append(w[:-2])
    if len(w) > 3 and w.endswith("s"):
        out.append(w[:-1])
    for v in list(out):
        if len(v) > 3 and v.endswith("a"):
            out.append(v[:-1] + "o")
    return out


def _emos(w: str, simples: dict):
    for v in _variantes(w):
        if v in simples:
            return simples[v]
    return ()


def perfil(texto: str) -> dict:
    """Conteo, proporción y palabras por emoción, más la emoción dominante."""
    emociones, simples, frases = _lexico()
    t = normalizar(texto)
    conteos = dict.fromkeys(emociones, 0)
    palabras = {e: [] for e in emociones}
    for w in _PAL.findall(t):
        for e in _emos(w, simples):
            conteos[e] += 1
            palabras[e].append(w)
    for pat, w, emos in frases:
        n = len(pat.findall(t))
        for e in emos:
            conteos[e] += n
            palabras[e].extend([w] * n)
    total = sum(conteos.values())
    proporciones = {e: (round(c / total, 4) if total else 0.0) for e, c in conteos.items()}
    dominante = max(conteos, key=conteos.get) if total else None
    return {"conteos": conteos, "proporciones": proporciones, "dominante": dominante,
            "palabras": {e: sorted(set(v)) for e, v in palabras.items() if v}}
