"""Fase 3 de CRISP-ML(Q): modelado (baselines y modelo candidato)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from ..core.text import normalizar


def modelo_candidato(p: dict) -> Pipeline:
    """TF-IDF de palabras + caracteres → Regresión logística (exportable a JSON)."""
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=normalizar, analyzer="word",
                                     ngram_range=tuple(p["word_ngram_range"]), min_df=p["word_min_df"],
                                     sublinear_tf=p["sublinear_tf"])),
            ("char", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                     ngram_range=tuple(p["char_ngram_range"]), min_df=p["char_min_df"],
                                     sublinear_tf=p["sublinear_tf"])),
        ])),
        ("clf", LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"])),
    ])


def baselines(p: dict) -> dict:
    return {
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "tfidf_naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), min_df=p["word_min_df"])),
            ("clf", MultinomialNB()),
        ]),
    }
