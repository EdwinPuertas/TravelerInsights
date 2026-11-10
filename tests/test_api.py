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
