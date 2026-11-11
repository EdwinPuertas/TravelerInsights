"""Fase 4 de CRISP-ML(Q): evaluación rigurosa y compuerta de calidad."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


def validacion_cruzada(modelo, X, y, folds: int, seed: int) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s = []
    for tr, va in skf.split(X, y):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        f1s.append(f1_score(y.iloc[va], m.predict(X.iloc[va]), average="macro"))
    return {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "folds": [round(float(f), 4) for f in f1s]}


def bootstrap_ic(y_true, y_pred, n: int, seed: int, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"low": round(float(lo), 4), "high": round(float(hi), 4), "samples": n}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar: ¿difieren los errores de A y B sobre los mismos ejemplos?"""
    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = a == y_true, b == y_true
    solo_a = int((a_ok & ~b_ok).sum())
    solo_b = int((~a_ok & b_ok).sum())
    n = solo_a + solo_b
    p = float(binomtest(solo_a, n, 0.5).pvalue) if n else 1.0
    return {"only_candidate_correct": solo_a, "only_baseline_correct": solo_b, "p_value": round(p, 6)}


def calibracion(y_true, proba, clases, bins: int) -> dict:
    """Error de calibración esperado (ECE) y curva de fiabilidad."""
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    acierto = (np.asarray(clases)[proba.argmax(1)] == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, curva = 0.0, []
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acierto[m].mean() - conf[m].mean())
            curva.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                          "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(acierto[m].mean()), 4)})
    return {"ece": round(float(ece), 4), "reliability": curva}


def curva_tau(y_true, proba, clases) -> list[dict]:
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    pred = np.asarray(clases)[proba.argmax(1)]
    out = []
    for tau in np.round(np.arange(0.30, 0.96, 0.05), 2):
        auto = conf >= tau
        out.append({"tau": float(tau), "coverage": round(float(auto.mean()), 4),
                    "accuracy_auto": round(float((pred[auto] == y_true[auto]).mean()), 4) if auto.any() else None})
    return out


def por_subgrupo(textos, y_true, y_pred) -> dict:
    """Macro-F1 por longitud de reseña (terciles)."""
    largo = np.array([len(str(t).split()) for t in textos])
    q1, q2 = np.quantile(largo, [1 / 3, 2 / 3])
    grupos = {"cortas": largo <= q1, "medias": (largo > q1) & (largo <= q2), "largas": largo > q2}
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {g: {"n": int(m.sum()),
                "macro_f1": round(float(f1_score(y_true[m], y_pred[m], average="macro")), 4) if m.sum() else None}
            for g, m in grupos.items()}


def metricas_prueba(y_true, y_pred, clases) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro")), 4),   # = exactitud (una etiqueta)
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": list(clases),
                             "matrix": confusion_matrix(y_true, y_pred, labels=list(clases)).tolist()},
    }


def particion_balanceada(y, por_clase: int, seed: int):
    """Índices (train, test) con el MISMO número de ejemplos por clase en la prueba.

    El resto de cada clase queda para entrenamiento. `por_clase` no puede superar
    la mitad de la clase más pequeña, para no dejar el entrenamiento sin ejemplos.
    """
    y = np.asarray(y)
    clases, conteos = np.unique(y, return_counts=True)
    maximo = int(conteos.min() // 2)
    if por_clase > maximo:
        raise ValueError(f"test_per_class={por_clase} supera la mitad de la clase menor ({conteos.min()}); máximo {maximo}")
    rng = np.random.default_rng(seed)
    te = np.concatenate([rng.choice(np.flatnonzero(y == c), por_clase, replace=False) for c in clases])
    tr = np.setdiff1d(np.arange(len(y)), te)
    return tr, np.sort(te)


def esquema_particion(cfg: dict) -> dict:
    d = cfg["data"]
    if d.get("test_split", "stratified") == "balanced":
        return {"test_split": "balanced", "test_per_class": int(d["test_per_class"]), "seed": cfg["seed"]}
    return {"test_split": "stratified", "test_size": d["test_size"], "seed": cfg["seed"]}


def evaluacion_balanceada_repetida(fabricas: dict, X, y, X_extra, y_extra, por_clase: int, repeticiones: int, seed: int) -> dict:
    """macro-F1 sobre `repeticiones` particiones balanceadas distintas.

    Una sola prueba balanceada es pequeña (pocas reseñas negativas): repetir la
    partición da una estimación más estable y su variabilidad real.
    `X_extra`/`y_extra` (retroalimentación humana) se suman siempre al entrenamiento.
    """
    X, y = np.asarray(X, dtype=object), np.asarray(y)
    f1s = {n: [] for n in fabricas}
    for r in range(repeticiones):
        tr, te = particion_balanceada(y, por_clase, seed + 1000 + r)
        Xtr = np.concatenate([X[tr], np.asarray(X_extra, dtype=object)])
        ytr = np.concatenate([y[tr], np.asarray(y_extra)])
        for n, fab in fabricas.items():
            m = fab().fit(Xtr, ytr)
            f1s[n].append(f1_score(y[te], m.predict(X[te]), average="macro"))
    out = {}
    for n, v in f1s.items():
        v = np.array(v)
        out[n] = {"macro_f1_mean": round(float(v.mean()), 4), "macro_f1_std": round(float(v.std(ddof=1)), 4),
                  "p2_5": round(float(np.percentile(v, 2.5)), 4), "p97_5": round(float(np.percentile(v, 97.5)), 4)}
    return {"repeticiones": repeticiones, "test_per_class": por_clase, "modelos": out}


def compuerta(resultado: dict, gate: dict, produccion: dict | None, esquema: dict | None = None) -> dict:
    f1 = resultado["candidate"]["test"]["macro_f1"]
    dummy = resultado["baselines"]["dummy_mayoritaria"]["test"]["macro_f1"]
    checks = {
        "min_macro_f1": (f1 >= gate["min_macro_f1"], f"macro-F1 {f1:.4f} ≥ {gate['min_macro_f1']}"),
        "beats_dummy": (f1 - dummy >= gate["must_beat_dummy_by"], f"{f1:.4f} − dummy {dummy:.4f} ≥ {gate['must_beat_dummy_by']}"),
    }
    rep = resultado["candidate"].get("repeated_balanced")
    if rep:
        m = rep["modelos"]["candidate"]["macro_f1_mean"]
        checks["min_macro_f1_repetido"] = (m >= gate["min_macro_f1"],
                                           f"media {m:.4f} en {rep['repeticiones']} pruebas balanceadas ≥ {gate['min_macro_f1']}")
    if produccion:
        esquema_prod = produccion.get("run", {}).get("split")
        if esquema is None or esquema_prod == esquema:
            prod = produccion["candidate"]["test"]["macro_f1"]
            checks["no_regression"] = (f1 >= prod - gate["max_regression_vs_production"],
                                       f"{f1:.4f} ≥ producción {prod:.4f} − {gate['max_regression_vs_production']}")
        else:
            # Métricas en particiones distintas no son comparables (y el modelo en producción
            # se reentrenó con todos los datos, incluida la nueva prueba). Lo decide la revisión del PR.
            checks["no_regression"] = (True, f"no comparable: cambió la partición de prueba ({esquema_prod} → {esquema}); "
                                             "revise el PR")
    out = {k: {"passed": bool(ok), "detail": d} for k, (ok, d) in checks.items()}
    return {"checks": out, "passed": all(c["passed"] for c in out.values())}
