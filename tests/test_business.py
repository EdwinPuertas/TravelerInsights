from traveler_insights.core.business import detectar_aspectos, mapa_polaridad, requiere_revision, respuesta_plantilla


def test_aspectos():
    assert detectar_aspectos("El baño sucio y el personal grosero") == ["limpieza", "servicio"]
    assert detectar_aspectos("nada que decir") == []


def test_polaridad():
    assert mapa_polaridad(["negativo", "neutro", "positivo"]) == {"negativo": "negativa", "neutro": "neutra", "positivo": "positiva"}
    assert mapa_polaridad(["1", "3", "5"]) == {"1": "negativa", "3": "neutra", "5": "positiva"}
    assert mapa_polaridad(["0", "1"]) == {"0": "negativa", "1": "positiva"}


def test_plantillas_no_prometen():
    for pol in ["positiva", "negativa", "neutra"]:
        assert not requiere_revision(respuesta_plantilla("el precio muy caro", pol))


def test_guardarrail():
    assert requiere_revision("Te ofrecemos un descuento del 20 %")
    assert requiere_revision("Te haremos el reembolso en 5 días")
    assert not requiere_revision("Un integrante del equipo se comunicará contigo")
