"""Balanceo de clases para el entrenamiento (CRISP-ML(Q) fase 3).

- Remuestreo "promedio de las dos clases menores": el objetivo por clase es el promedio
  de los conteos de las dos clases más pequeñas del conjunto de ENTRENAMIENTO recibido.
  Las clases por encima se submuestrean (RandomUnderSampler) y las que quedan por debajo
  se completan con SMOTE en el espacio de rasgos (TF-IDF + léxico).
  Va dentro del pipeline (imbalanced-learn), así que solo se aplica al ajustar y en cada
  pliegue de validación: nunca toca los datos de evaluación ni la predicción.
- Datos externos: reseñas reales de otros corpus que se suman SOLO al entrenamiento.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def objetivo_dos_menores(y) -> int:
    conteos = sorted(Counter(y).values())
    return int(round((conteos[0] + conteos[1]) / 2)) if len(conteos) > 1 else conteos[0]


def _remuestrear(X, y, seed: int = 42, k_neighbors: int = 5):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    y = np.asarray(y)
    obj = objetivo_dos_menores(y)
    conteos = Counter(y)
    bajar = {c: obj for c, n in conteos.items() if n > obj}
    if bajar:
        X, y = RandomUnderSampler(sampling_strategy=bajar, random_state=seed).fit_resample(X, y)
    subir = {c: obj for c, n in Counter(y).items() if n < obj}
    if subir:
        k = max(1, min(k_neighbors, min(Counter(y)[c] for c in subir) - 1))
        X, y = SMOTE(sampling_strategy=subir, k_neighbors=k, random_state=seed).fit_resample(X, y)
    return X, y


def muestreador(seed: int = 42, k_neighbors: int = 5):
    from imblearn import FunctionSampler

    return FunctionSampler(func=_remuestrear, accept_sparse=True, validate=False,
                           kw_args={"seed": seed, "k_neighbors": k_neighbors})


class ConDatosExternos(ClassifierMixin, BaseEstimator):
    """Envuelve un clasificador y le suma reseñas externas al entrenar (nunca al evaluar)."""

    def __init__(self, estimator, X_ext=(), y_ext=()):
        self.estimator = estimator
        self.X_ext = X_ext
        self.y_ext = y_ext

    def fit(self, X, y):
        X = np.concatenate([np.asarray(X, dtype=object), np.asarray(self.X_ext, dtype=object)])
        y = np.concatenate([np.asarray(y, dtype=object), np.asarray(self.y_ext, dtype=object)])
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
