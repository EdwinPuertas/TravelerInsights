import app as api
from fastapi.testclient import TestClient

from traveler_insights.core.model import LinearTextModel
from traveler_insights.ml.export import a_json, guardar
from traveler_insights.ml.train import modelo_candidato


def _cliente(tmp_path, monkeypatch, df, params):
    m = modelo_candidato(params).fit(df.review, df.label)
    p = tmp_path / "model.json"
    guardar(a_json(m, {"tau": 0.7}), p)
    monkeypatch.setattr(api, "MODEL_PATH", p)
    monkeypatch.setattr(api, "METRICS_PATH", tmp_path / "no_existe.json")
    api.modelo.cache_clear()
    return TestClient(api.app)


def test_flujo(tmp_path, monkeypatch, df_multi, params):
    c = _cliente(tmp_path, monkeypatch, df_multi, params)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/").status_code == 200
    info = c.get("/api/model").json()
    assert set(info["classes"]) == {"negativo", "neutro", "positivo"}
    r = c.post("/api/analyze", json={"texts": ["la habitación sucia y el personal grosero"], "tau": 0.5}).json()
    res = r["results"][0]
    assert res["clase"] == "negativo"
    assert "limpieza" in res["aspectos"] and "servicio" in res["aspectos"]
    assert res["decision"] in {"automatizar", "revision_humana"}


def test_validacion(tmp_path, monkeypatch, df_multi, params):
    c = _cliente(tmp_path, monkeypatch, df_multi, params)
    assert c.post("/api/analyze", json={"texts": []}).status_code == 422
    assert c.post("/api/analyze", json={"texts": ["x"] * 201}).status_code == 422
    assert c.post("/api/analyze", json={"texts": ["x"], "tau": 2}).status_code == 422


def test_sin_modelo(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nada.json")
    api.modelo.cache_clear()
    assert TestClient(api.app).get("/api/model").status_code == 503
    api.modelo.cache_clear()
    assert isinstance(LinearTextModel, type)


def test_historial_feedback_con_store_simulado(tmp_path, monkeypatch, df_multi, params):
    from traveler_insights.core import store
    c = _cliente(tmp_path, monkeypatch, df_multi, params)
    mem = {"h": [], "f": []}
    monkeypatch.setattr(store, "configurado", lambda: True)
    monkeypatch.setattr(store, "guardar_historial", lambda r: mem["h"].insert(0, r))
    monkeypatch.setattr(store, "historial", lambda limite=50: mem["h"][:limite])
    monkeypatch.setattr(store, "guardar_feedback", lambda r: mem["f"].insert(0, r))
    monkeypatch.setattr(store, "feedback", lambda: mem["f"])
    r = c.post("/api/analyze", json={"texts": ["la habitación sucia y el personal grosero"]}).json()
    assert r["stored"] and "emociones" in r["results"][0]
    assert c.get("/api/history").json()["items"][0]["texto"].startswith("la habitación")
    res = r["results"][0]
    ok = c.post("/api/feedback", json={"texto": res["texto"], "clase_predicha": res["clase"], "clase_correcta": "negativo",
                                      "respuesta_sugerida": res["respuesta_sugerida"],
                                      "respuesta_editada": "Lamentamos lo ocurrido, te contactaremos."}).json()
    assert ok["stored"] and "bleu" in ok["registro"]["metricas_respuesta"]
    st = c.get("/api/feedback/stats").json()
    assert st["n_etiquetados"] == 1 and "micro_f1" in st
    assert c.post("/api/feedback", json={"texto": "x", "clase_predicha": "a", "clase_correcta": "inventada"}).status_code == 422


def test_stats_locales_sin_store(tmp_path, monkeypatch, df_multi, params):
    c = _cliente(tmp_path, monkeypatch, df_multi, params)
    st = c.post("/api/feedback/stats", json={"items": [{"clase_predicha": "neutro", "clase_correcta": "neutro"}]}).json()
    assert st["micro_f1"] == 1.0
