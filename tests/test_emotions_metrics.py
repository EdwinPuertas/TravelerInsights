import pytest

from traveler_insights.core import feedback, textmetrics
from traveler_insights.core.emotions import EMOCIONES, perfil


def test_perfil_emociones():
    p = perfil("Qué alegría, el lugar es hermoso y la gente muy amable")
    assert set(p["conteos"]) == set(EMOCIONES)
    assert sum(p["conteos"].values()) > 0 and p["dominante"] in EMOCIONES
    assert perfil("")["dominante"] is None


def test_bleu_rouge_identicos_y_distintos():
    s = "Lamentamos tu experiencia. Un integrante del equipo te contactará."
    assert textmetrics.bleu(s, s) == pytest.approx(1.0)
    assert textmetrics.rouge_l(s, s) == 1.0 and textmetrics.rouge_n(s, s, 2) == 1.0
    assert textmetrics.bleu(s, "gracias por visitarnos") < 0.2
    assert 0 < textmetrics.rouge_n(s, "Lamentamos mucho tu experiencia en la playa", 1) < 1


def test_resumen_feedback_micro_macro():
    items = [{"clase_predicha": "neutral", "clase_correcta": "negativo",
              "respuesta_sugerida": "Gracias por tus comentarios", "respuesta_editada": "Lamentamos lo ocurrido"},
             {"clase_predicha": "positivo", "clase_correcta": "positivo"},
             {"clase_predicha": "neutral", "clase_correcta": "neutral"}]
    r = feedback.resumen(items, ["negativo", "neutral", "positivo"])
    assert r["micro_f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["correcciones"] == {"neutral → negativo": 1}
    assert r["respuestas"]["editadas"] == 1 and "bleu" in r["respuestas"]
