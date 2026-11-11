"""Métricas de la retroalimentación humana: el desempeño del modelo *en producción*.

Cada retroalimentación trae la clase predicha y la clase que confirmó o corrigió una persona.
Con eso se calculan micro-F1 (= exactitud en clasificación de una etiqueta) y macro-F1 sobre
casos reales, y BLEU/ROUGE entre la respuesta sugerida y la editada por la persona.
"""
from __future__ import annotations

from collections import Counter

from . import textmetrics


def _f1_por_clase(pares, clases):
    out = {}
    for c in clases:
        tp = sum(1 for p, r in pares if p == c and r == c)
        fp = sum(1 for p, r in pares if p == c and r != c)
        fn = sum(1 for p, r in pares if p != c and r == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0, "soporte": tp + fn}
    return out


def resumen(items: list[dict], clases: list[str]) -> dict:
    pares = [(i["clase_predicha"], i["clase_correcta"]) for i in items if i.get("clase_correcta")]
    n = len(pares)
    res = {"n_feedback": len(items), "n_etiquetados": n}
    if n:
        por_clase = _f1_por_clase(pares, clases)
        presentes = [c for c in clases if por_clase[c]["soporte"] > 0]
        res.update({
            "micro_f1": round(sum(p == r for p, r in pares) / n, 4),
            "macro_f1": round(sum(por_clase[c]["f1"] for c in presentes) / len(presentes), 4) if presentes else None,
            "por_clase": por_clase,
            "correcciones": dict(Counter(f"{p} → {r}" for p, r in pares if p != r).most_common(10)),
        })
    editadas = [i for i in items if i.get("respuesta_editada") and i.get("respuesta_sugerida")
                and i["respuesta_editada"].strip() != i["respuesta_sugerida"].strip()]
    aprobadas = [i for i in items if i.get("respuesta_sugerida") and i.get("respuesta_editada")
                 and i["respuesta_editada"].strip() == i["respuesta_sugerida"].strip()]
    con_ref = editadas + aprobadas
    res["respuestas"] = {"evaluadas": len(con_ref), "editadas": len(editadas)}
    if con_ref:
        m = [textmetrics.todas(i["respuesta_sugerida"], i["respuesta_editada"]) for i in con_ref]
        res["respuestas"].update({k: round(sum(x[k] for x in m) / len(m), 4) for k in m[0]})
    return res
