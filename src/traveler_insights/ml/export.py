"""Fase 5 de CRISP-ML(Q): exportar el modelo a un formato liviano para producción."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.model import FORMAT, LinearTextModel


def a_json(pipeline, metadata: dict) -> dict:
    feats = pipeline.named_steps["feats"]
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_                      # (k, n) o (1, n) si es binario
    blocks, offset = [], 0
    for name, vec in feats.transformer_list:
        vocab = vec.vocabulary_
        idf = vec.idf_
        n = len(vocab)
        features = {}
        for term, j in vocab.items():
            w = coef[:, offset + j]
            features[term] = [round(float(idf[j]), 6)] + [round(float(x), 6) for x in w]
        blocks.append({"name": name, "analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range),
                       "sublinear_tf": bool(vec.sublinear_tf), "features": features})
        offset += n
    return {
        "format": FORMAT,
        "classes": [str(c) for c in clf.classes_],
        "intercept": [round(float(x), 6) for x in clf.intercept_],
        "binary": coef.shape[0] == 1,
        "blocks": blocks,
        "metadata": metadata,
    }


def paridad(pipeline, spec: dict, textos) -> float:
    """Máxima diferencia absoluta de probabilidades entre scikit-learn y el modelo JSON."""
    ref = pipeline.predict_proba(list(textos))
    js = np.array(LinearTextModel(spec).predict_proba(list(textos)))
    return float(np.abs(ref - js).max())


def guardar(spec: dict, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    path.write_text(data, encoding="utf-8")
    return len(data.encode("utf-8"))
