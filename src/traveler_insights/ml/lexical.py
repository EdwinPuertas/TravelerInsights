"""Adaptador scikit-learn para los rasgos léxicos de ``core/lexical.py``."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..core.lexical import NOMBRES, rasgos


class RasgosLexicos(BaseEstimator, TransformerMixin):
    """Convierte cada texto en el vector de rasgos léxicos (sin estado: fit no aprende nada)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([rasgos(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(NOMBRES)
