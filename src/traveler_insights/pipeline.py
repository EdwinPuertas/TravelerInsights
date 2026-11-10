"""Pipeline de entrenamiento reproducible siguiendo CRISP-ML(Q).

Uso:
    python -m traveler_insights.pipeline --config configs/params.yaml

Etapas (cada una deja evidencia en reports/):
    1. data      → validación de calidad de datos           (reports/data_quality.json)
    2. train     → baselines + candidato, validación cruzada
    3. evaluate  → prueba, bootstrap, McNemar, calibración, subgrupos, curva τ
    4. gate      → compuerta de calidad                      (reports/metrics.json, reports/evaluation.md)
    5. export    → model/model.json + paridad + MODEL_CARD.md
Código de salida 0 = modelo aprobado y exportado; 1 = rechazado (no se despliega).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

from .core.business import mapa_polaridad
from .ml import evaluate as ev
from .ml.data import preparar
from .ml.export import a_json, guardar, paridad
from .ml.train import baselines, modelo_candidato


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(config: str, out_model: str = "model", out_reports: str = "reports") -> int:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    rep_dir, mod_dir = Path(out_reports), Path(out_model)

    # ---------------------------------------------------------------- 1. datos
    data_path = Path(cfg["data"]["train_path"])
    print(f"[1/5] Datos: {data_path}")
    df_raw = pd.read_csv(data_path)
    d, calidad = preparar(df_raw, cfg["data"])
    calidad["data_sha256"] = _sha256(data_path)
    _dump(calidad, rep_dir / "data_quality.json")
    for k, c in calidad["checks"].items():
        print(f"      {'✔' if c['passed'] else '✘'} {k}: {c['detail']}")
    if not calidad["passed"]:
        print("✘ Los datos no cumplen los requisitos de calidad. Pipeline detenido.")
        return 1

    X_tr, X_te, y_tr, y_te = train_test_split(d["texto"], d["etiqueta"], test_size=cfg["data"]["test_size"],
                                              stratify=d["etiqueta"], random_state=seed)

    # ---------------------------------------------------------------- 2. entrenamiento
    print("[2/5] Entrenamiento con validación cruzada")
    p = cfg["model"]
    folds = cfg["evaluation"]["cv_folds"]
    modelos = {**baselines(p), "candidate": modelo_candidato(p)}
    res = {"baselines": {}, "candidate": {}}
    preds = {}
    for nombre, m in modelos.items():
        cv = ev.validacion_cruzada(m, X_tr, y_tr, folds, seed)
        m.fit(X_tr, y_tr)
        preds[nombre] = m.predict(X_te)
        bloque = {"cv": cv, "test": ev.metricas_prueba(y_te, preds[nombre], m.classes_)}
        (res["candidate"] if nombre == "candidate" else res["baselines"].setdefault(nombre, {})).update(bloque)
        print(f"      {nombre:20s} CV macro-F1 {cv['macro_f1_mean']:.4f} ± {cv['macro_f1_std']:.4f}"
              f" | prueba {bloque['test']['macro_f1']:.4f}")

    # ---------------------------------------------------------------- 3. evaluación
    print("[3/5] Evaluación del candidato")
    cand = modelos["candidate"]
    proba = cand.predict_proba(X_te)
    e = cfg["evaluation"]
    res["candidate"]["bootstrap_ci_macro_f1"] = ev.bootstrap_ic(y_te, preds["candidate"], e["bootstrap_samples"], seed)
    res["candidate"]["mcnemar_vs_naive_bayes"] = ev.mcnemar(y_te, preds["candidate"], preds["tfidf_naive_bayes"])
    res["candidate"]["calibration"] = ev.calibracion(y_te, proba, cand.classes_, e["calibration_bins"])
    res["candidate"]["tau_curve"] = ev.curva_tau(y_te, proba, cand.classes_)
    res["candidate"]["by_length"] = ev.por_subgrupo(X_te, y_te, preds["candidate"])

    # ---------------------------------------------------------------- 4. compuerta
    print("[4/5] Compuerta de calidad")
    prod_path = mod_dir / "metrics.json"
    produccion = json.loads(prod_path.read_text(encoding="utf-8")) if prod_path.exists() else None
    gate = ev.compuerta(res, cfg["quality_gate"], produccion)
    res["quality_gate"] = gate
    res["run"] = {"timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                  "python": platform.python_version(), "sklearn": sklearn.__version__,
                  "config": cfg, "data_sha256": calidad["data_sha256"],
                  "n_train": len(X_tr), "n_test": len(X_te)}
    for k, c in gate["checks"].items():
        print(f"      {'✔' if c['passed'] else '✘'} {k}: {c['detail']}")

    # ---------------------------------------------------------------- 5. exportación
    print("[5/5] Exportación")
    final = modelo_candidato(p).fit(d["texto"], d["etiqueta"])   # reentrenar con todos los datos
    meta = {"created_utc": res["run"]["timestamp_utc"], "data_sha256": calidad["data_sha256"],
            "columns": calidad["columns"], "test_macro_f1": res["candidate"]["test"]["macro_f1"],
            "polarity": mapa_polaridad([str(c) for c in final.classes_]), "tau": cfg["serving"]["tau"]}
    spec = a_json(final, meta)
    err = paridad(final, spec, X_te.head(300))
    ok_par = err <= cfg["quality_gate"]["max_parity_error"]
    gate["checks"]["json_parity"] = {"passed": ok_par, "detail": f"error máx. {err:.2e} ≤ {cfg['quality_gate']['max_parity_error']}"}
    gate["passed"] = gate["passed"] and ok_par
    print(f"      {'✔' if ok_par else '✘'} json_parity: error máx. {err:.2e}")

    _dump(res, rep_dir / "metrics.json")
    (rep_dir / "evaluation.md").write_text(informe(res, calidad), encoding="utf-8")
    if not gate["passed"]:
        print("✘ El candidato NO pasó la compuerta de calidad. No se exporta el modelo.")
        return 1
    size = guardar(spec, mod_dir / "model.json")
    _dump(res, mod_dir / "metrics.json")
    (mod_dir / "MODEL_CARD.md").write_text(model_card(res, calidad, spec), encoding="utf-8")
    print(f"✔ Modelo aprobado y exportado: {mod_dir / 'model.json'} ({size / 1e6:.1f} MB)")
    return 0


def informe(res: dict, calidad: dict) -> str:
    c = res["candidate"]
    filas = [f"| {n} | {b['cv']['macro_f1_mean']:.4f} ± {b['cv']['macro_f1_std']:.4f} | {b['test']['macro_f1']:.4f} |"
             for n, b in {**res["baselines"], "candidate (TF-IDF + LogReg)": c}.items()]
    gate = "\n".join(f"- {'✅' if v['passed'] else '❌'} **{k}**: {v['detail']}" for k, v in res["quality_gate"]["checks"].items())
    ic = c["bootstrap_ci_macro_f1"]
    mc = c["mcnemar_vs_naive_bayes"]
    sub = "\n".join(f"| {g} | {v['n']} | {v['macro_f1']} |" for g, v in c["by_length"].items())
    return f"""# Informe de evaluación

Generado: {res['run']['timestamp_utc']} · datos `{calidad['data_sha256'][:12]}` · entrenamiento {res['run']['n_train']} · prueba {res['run']['n_test']}

## Compuerta de calidad: {'APROBADO ✅' if res['quality_gate']['passed'] else 'RECHAZADO ❌'}
{gate}

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
{chr(10).join(filas)}

- IC 95 % por bootstrap del macro-F1 del candidato: [{ic['low']}, {ic['high']}]
- McNemar frente a Naive Bayes: solo el candidato acierta {mc['only_candidate_correct']}, solo NB acierta {mc['only_baseline_correct']}, p = {mc['p_value']}
- Calibración: ECE = {c['calibration']['ece']}

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
{sub}
"""


def model_card(res: dict, calidad: dict, spec: dict) -> str:
    c = res["candidate"]
    return f"""# Model card · Traveler Insights

- **Tarea:** clasificación de reseñas de viajeros ({', '.join(spec['classes'])}).
- **Arquitectura:** TF-IDF de palabras {res['run']['config']['model']['word_ngram_range']} y caracteres {res['run']['config']['model']['char_ngram_range']} → Regresión logística (C={res['run']['config']['model']['C']}, pesos de clase balanceados). Servido como JSON sin scikit-learn.
- **Datos:** competencia Kaggle *Traveler Insights UPTC 2026*; huella `{calidad['data_sha256'][:16]}`; {calidad['rows_final']} reseñas tras limpieza.
- **Desempeño en prueba:** macro-F1 {c['test']['macro_f1']} (IC 95 % [{c['bootstrap_ci_macro_f1']['low']}, {c['bootstrap_ci_macro_f1']['high']}]), exactitud {c['test']['accuracy']}, ECE {c['calibration']['ece']}.
- **Uso previsto:** apoyo a un equipo humano que clasifica y responde reseñas. Las respuestas son sugerencias.
- **Usos no previstos:** decisiones automáticas sobre personas, reseñas en otros idiomas o dominios.
- **Limitaciones:** ironía, negación y reseñas mixtas; el léxico de aspectos no cubre todo; el desempeño puede degradarse con la deriva de los datos.
- **Responsabilidad:** las reseñas pueden contener datos personales; no se almacenan en la API.
- **Generado:** {res['run']['timestamp_utc']} con scikit-learn {res['run']['sklearn']}.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    sys.exit(run(a.config, a.model_dir, a.reports_dir))


if __name__ == "__main__":
    main()
