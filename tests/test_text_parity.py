"""Los n-gramas propios deben ser idénticos a los de scikit-learn."""
from conftest import NEG, NEU, POS, TEXTOS_RAROS
from sklearn.feature_extraction.text import TfidfVectorizer

from traveler_insights.core.text import char_wb_ngrams, normalizar, word_ngrams

DOCS = POS + NEG + NEU + TEXTOS_RAROS


def test_word_ngrams_igual_a_sklearn():
    an = TfidfVectorizer(preprocessor=normalizar, analyzer="word", ngram_range=(1, 2)).build_analyzer()
    for d in DOCS:
        assert word_ngrams(normalizar(d), 1, 2) == an(d)


def test_char_wb_igual_a_sklearn():
    an = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb", ngram_range=(3, 5)).build_analyzer()
    for d in DOCS:
        assert char_wb_ngrams(normalizar(d), 3, 5) == an(d)


def test_normalizar():
    assert normalizar("  Árbol   ÑANDÚ ") == "arbol nandu"
