"""Fase 3 de CRISP-ML(Q): modelado (baselines y modelo candidato)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from ..core.text import normalizar
from .lexical import RasgosLexicos


def modelo_candidato(p: dict, lexical: bool | None = None, resampling: bool | None = None):
    """TF-IDF de palabras + caracteres (+ rasgos léxicos del curso) → Regresión logística.

    ``lexical=None`` toma el valor de ``p["lexical_features"]``. Todo es exportable a JSON.
    """
    usar_lex = p.get("lexical_features", False) if lexical is None else lexical
    bloques = [
        ("word", TfidfVectorizer(preprocessor=normalizar, analyzer="word",
                                 ngram_range=tuple(p["word_ngram_range"]), min_df=p["word_min_df"],
                                 sublinear_tf=p["sublinear_tf"])),
        ("char", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                 ngram_range=tuple(p["char_ngram_range"]), min_df=p["char_min_df"],
                                 sublinear_tf=p["sublinear_tf"])),
    ]
    if usar_lex:
        # El escalador se ajusta dentro de cada pliegue (es parte del pipeline): sin fuga de información
        bloques.append(("lex", Pipeline([("rasgos", RasgosLexicos()), ("escala", StandardScaler())])))
    usar_rs = (p.get("resampling", "none") != "none") if resampling is None else resampling
    clf = LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"])
    if usar_rs:
        # SMOTE + submuestreo al promedio de las dos clases menores, solo durante el ajuste
        from imblearn.pipeline import Pipeline as ImbPipeline

        from .balance import muestreador
        return ImbPipeline([("feats", FeatureUnion(bloques)), ("balance", muestreador(p.get("seed", 42))), ("clf", clf)])
    return Pipeline([("feats", FeatureUnion(bloques)), ("clf", clf)])


def baselines(p: dict, externos: tuple | None = None) -> dict:
    """Baselines y ablaciones. `externos` = (X_ext, y_ext) si el candidato usa datos externos."""
    from .balance import ConDatosExternos
    extra = {}
    rs = p.get("resampling", "none") != "none"
    if rs or externos:
        extra["sin_balanceo_ni_externos"] = modelo_candidato(p, resampling=False)
    if rs and externos:
        extra["solo_smote_submuestreo"] = modelo_candidato(p, resampling=True)
        extra["solo_datos_externos"] = ConDatosExternos(modelo_candidato(p, resampling=False), *externos)
    if p.get("lexical_features", False):
        # Ablación: el mismo candidato sin rasgos léxicos, para medir cuánto aportan
        sin_lex = modelo_candidato(p, lexical=False)
        extra["tfidf_logreg_sin_lexico"] = ConDatosExternos(sin_lex, *externos) if externos else sin_lex
    return {**extra,
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "tfidf_naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), min_df=p["word_min_df"])),
            ("clf", MultinomialNB()),
        ]),
    }
