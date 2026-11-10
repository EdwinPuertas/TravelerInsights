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


def modelo_candidato(p: dict, lexical: bool | None = None) -> Pipeline:
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
    return Pipeline([
        ("feats", FeatureUnion(bloques)),
        ("clf", LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"])),
    ])


def baselines(p: dict) -> dict:
    extra = {}
    if p.get("lexical_features", False):
        # Ablación: el mismo candidato sin rasgos léxicos, para medir cuánto aportan
        extra["tfidf_logreg_sin_lexico"] = modelo_candidato(p, lexical=False)
    return {**extra,
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "tfidf_naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), min_df=p["word_min_df"])),
            ("clf", MultinomialNB()),
        ]),
    }
