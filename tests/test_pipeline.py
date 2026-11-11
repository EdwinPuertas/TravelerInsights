"""Ejecuta el pipeline completo con datos de prueba y umbrales relajados."""
import json

import yaml

from traveler_insights.pipeline import run


def _cfg(tmp_path, csv, **gate):
    with open("configs/params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["train_path"] = str(csv)
    cfg["data"].update({"text_col": "review", "label_col": "label"})
    cfg["data"]["quality"].update({"min_rows": 50, "min_per_class": 5})
    cfg["model"].update({"word_min_df": 1, "char_min_df": 1})
    cfg["evaluation"].update({"cv_folds": 3, "bootstrap_samples": 50})
    cfg["quality_gate"].update(gate)
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_pipeline_aprueba_y_exporta(tmp_path, df_multi):
    csv = tmp_path / "train.csv"
    df_multi.to_csv(csv, index=False)
    code = run(str(_cfg(tmp_path, csv)), str(tmp_path / "model"), str(tmp_path / "reports"))
    assert code == 0
    assert (tmp_path / "model" / "model.json").exists()
    m = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert m["quality_gate"]["passed"] and "json_parity" in m["quality_gate"]["checks"]


def test_compuerta_rechaza(tmp_path, df_multi):
    csv = tmp_path / "train.csv"
    df_multi.to_csv(csv, index=False)
    code = run(str(_cfg(tmp_path, csv, min_macro_f1=1.01)), str(tmp_path / "model"), str(tmp_path / "reports"))
    assert code == 1
    assert not (tmp_path / "model" / "model.json").exists()


def test_feedback_se_suma_solo_al_entrenamiento(tmp_path, df_multi):
    csv = tmp_path / "train.csv"
    df_multi.to_csv(csv, index=False)
    fbp = tmp_path / "feedback.csv"
    import pandas as pd
    pd.DataFrame({"texto": ["reseña nueva corregida por humano uno", "reseña nueva corregida por humano dos"],
                  "clase_correcta": ["negativo", "positivo"]}).to_csv(fbp, index=False)
    cfgp = _cfg(tmp_path, csv)
    with open(cfgp, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["feedback_path"] = str(fbp)
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run(str(cfgp), str(tmp_path / "model"), str(tmp_path / "reports")) == 0
    dq = json.loads((tmp_path / "reports" / "data_quality.json").read_text(encoding="utf-8"))
    assert dq["feedback"]["rows_added"] == 2


def test_particion_balanceada_soporte_igual():
    import numpy as np

    from traveler_insights.ml.evaluate import particion_balanceada
    y = np.array(["neutral"] * 60 + ["positivo"] * 20 + ["negativo"] * 8)
    tr, te = particion_balanceada(y, 4, 0)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(y)
    vals, cnt = np.unique(y[te], return_counts=True)
    assert list(cnt) == [4, 4, 4]
    import pytest
    with pytest.raises(ValueError):
        particion_balanceada(y, 5, 0)          # más de la mitad de la clase menor
