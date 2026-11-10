"""Modelo lineal de texto servido desde JSON (sin scikit-learn).

Formato de ``model.json`` (lo genera ``traveler_insights.ml.export``):

{
  "format": "traveler-insights/linear-tfidf@1",
  "classes": ["neg", "neu", "pos"],
  "intercept": [...],                 # uno por clase (o uno si es binario)
  "binary": false,
  "blocks": [                          # un bloque por TfidfVectorizer de la FeatureUnion
     {"analyzer": "word", "ngram_range": [1, 2], "sublinear_tf": true,
      "features": {"termino": [idf, w_1, ..., w_k], ...}},
     ...
  ],
  "metadata": {...}
}
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .lexical import NOMBRES as NOMBRES_LEX
from .lexical import rasgos
from .text import char_wb_ngrams, normalizar, word_ngrams

FORMAT = "traveler-insights/linear-tfidf@1"


class LinearTextModel:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT:
            raise ValueError(f"Formato de modelo no soportado: {spec.get('format')}")
        self.spec = spec
        self.classes: list[str] = spec["classes"]
        self.intercept: list[float] = spec["intercept"]
        self.binary: bool = spec["binary"]
        self.blocks = spec["blocks"]
        self.metadata: dict = spec.get("metadata", {})

    @classmethod
    def load(cls, path: str | Path) -> LinearTextModel:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # -------------------------------------------------------------- inferencia
    def _block_scores(self, texto_norm: str, block: dict, n_out: int) -> list[float]:
        lo, hi = block["ngram_range"]
        grams = (word_ngrams if block["analyzer"] == "word" else char_wb_ngrams)(texto_norm, lo, hi)
        feats = block["features"]
        counts = Counter(g for g in grams if g in feats)
        if not counts:
            return [0.0] * n_out
        vals = {}
        for g, tf in counts.items():
            tf = 1.0 + math.log(tf) if block["sublinear_tf"] else float(tf)
            vals[g] = tf * feats[g][0]
        norm = math.sqrt(sum(v * v for v in vals.values())) or 1.0
        scores = [0.0] * n_out
        for g, v in vals.items():
            w = feats[g]
            x = v / norm
            for k in range(n_out):
                scores[k] += x * w[k + 1]
        return scores

    @staticmethod
    def _lexical_scores(texto: str, block: dict, n_out: int) -> list[float]:
        if block["names"] != NOMBRES_LEX:
            raise ValueError("Los rasgos léxicos del modelo no coinciden con core/lexical.py")
        scores = [0.0] * n_out
        for x, m, sc, w in zip(rasgos(texto), block["mean"], block["scale"], block["weights"]):
            v = (x - m) / sc
            for k in range(n_out):
                scores[k] += v * w[k]
        return scores

    def decision_function(self, texto: str) -> list[float]:
        t = normalizar(texto)
        n_out = len(self.intercept)
        z = list(self.intercept)
        for b in self.blocks:
            if b["analyzer"] == "lexical":
                s = self._lexical_scores(texto, b, n_out)
            else:
                s = self._block_scores(t, b, n_out)
            z = [a + c for a, c in zip(z, s)]
        return z

    def predict_proba_one(self, texto: str) -> list[float]:
        z = self.decision_function(texto)
        if self.binary:
            p1 = 1.0 / (1.0 + math.exp(-z[0]))
            return [1.0 - p1, p1]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def predict_proba(self, textos: list[str]) -> list[list[float]]:
        return [self.predict_proba_one(t) for t in textos]

    def predict(self, textos: list[str]) -> list[str]:
        out = []
        for p in self.predict_proba(textos):
            out.append(self.classes[max(range(len(p)), key=p.__getitem__)])
        return out
