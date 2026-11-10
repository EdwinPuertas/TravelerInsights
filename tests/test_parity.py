"""El modelo exportado a JSON debe dar las mismas probabilidades que scikit-learn."""
import numpy as np
from conftest import TEXTOS_RAROS

from traveler_insights.core.model import LinearTextModel
from traveler_insights.ml.export import a_json, paridad
from traveler_insights.ml.train import modelo_candidato


def _check(df, params, lexical=False):
    m = modelo_candidato(params, lexical=lexical).fit(df.review, df.label)
    spec = a_json(m, {"tau": 0.7})
    textos = list(df.review.head(50)) + TEXTOS_RAROS
    assert paridad(m, spec, textos) < 1e-4
    js = LinearTextModel(spec)
    assert js.predict(textos) == list(m.predict(textos))
    assert np.allclose([sum(p) for p in js.predict_proba(textos)], 1.0)


def test_paridad_multiclase(df_multi, params):
    _check(df_multi, params)


def test_paridad_binaria(df_bin, params):
    _check(df_bin, params)


def test_paridad_con_rasgos_lexicos(df_multi, params):
    _check(df_multi, params, lexical=True)


def test_paridad_binaria_con_rasgos_lexicos(df_bin, params):
    _check(df_bin, params, lexical=True)
