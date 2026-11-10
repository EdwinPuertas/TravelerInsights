import pandas as pd
import pytest

from traveler_insights.ml.data import detectar_columnas


def test_prefiere_columna_de_sentimiento():
    df = pd.DataFrame({"ID": range(6), "Sitio": ["Tripadvisor", "Foursquare"] * 3,
                       "Comentario": [f"texto largo de reseña número {i}" for i in range(6)],
                       "Sentimiento": ["positive", "neutral", "negative"] * 2})
    assert detectar_columnas(df) == ("Comentario", "Sentimiento")


def test_columna_inexistente():
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    with pytest.raises(ValueError):
        detectar_columnas(df, "texto", "etiqueta")
