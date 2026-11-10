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
