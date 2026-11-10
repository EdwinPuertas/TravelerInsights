import numpy as np
from scipy.stats import kurtosis, skew

from traveler_insights.core.lexical import NOMBRES, rasgos


def _d(t):
    return dict(zip(NOMBRES, rasgos(t), strict=True))


def test_tildes_alcanzables_y_frases():
    d = _d("Él dijo que aún no, en absoluto, nunca más")
    assert d["third_person_singular"] >= 1          # "él" con tilde
    assert d["adverb_neg"] == 3                     # no, en absoluto, nunca
    assert d["adverb_cant"] >= 1                    # "más" con tilde


def test_momentos_igual_a_scipy_y_con_signo():
    t = "a bb ccc dddddddddd ee f"
    largos = [len(w) for w in t.split()]
    d = _d(t)
    assert np.isclose(d["kur_word"], kurtosis(largos))
    assert np.isclose(d["skew_word"], skew(largos))
    assert d["kur_word"] < 0 or d["skew_word"] > 0   # el signo se conserva


def test_diversidad_por_palabras_y_texto_vacio():
    assert _d("hola hola hola mundo")["lexical_diversity"] == 0.5
    assert all(v == 0.0 for v in rasgos(""))
