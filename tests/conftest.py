"""Datos de prueba sintéticos SOLO para pruebas automáticas del código.

No representan el dataset real ni se usan para reportar desempeño.
"""
import itertools
import random

import pandas as pd
import pytest

POS = ["excelente servicio y habitación muy limpia", "la comida deliciosa y el personal amable",
       "hermoso paisaje, el tour fue una gran experiencia", "buena ubicación cerca del centro"]
NEG = ["la habitación sucia y el baño con mal olor", "el personal grosero en la recepción",
       "muy caro para lo que ofrecen", "el wifi no funciona y mucho ruido"]
NEU = ["el hotel está bien, nada especial", "la visita fue normal", "ubicación regular, precio normal"]
RELLENO = ["tunja", "villa de leyva", "boyacá", "fin de semana", "con la familia", "en vacaciones"]


def _resenas(n=300, binaria=False, seed=0):
    rnd = random.Random(seed)
    filas = []
    grupos = [("positivo", POS), ("negativo", NEG)] + ([] if binaria else [("neutro", NEU)])
    for i in range(n):
        lab, base = rnd.choice(grupos)
        txt = f"{rnd.choice(base)} {' '.join(rnd.sample(RELLENO, 2))} visita {i}"
        filas.append({"id": i, "review": txt, "label": lab})
    return pd.DataFrame(filas)


@pytest.fixture
def df_multi():
    return _resenas()


@pytest.fixture
def df_bin():
    return _resenas(binaria=True)


@pytest.fixture
def params():
    return {"word_ngram_range": [1, 2], "word_min_df": 1, "char_ngram_range": [3, 5], "char_min_df": 1,
            "sublinear_tf": True, "C": 4.0, "class_weight": "balanced", "max_iter": 2000}


TEXTOS_RAROS = ["", "  ", "a", "¡¡Excelente!!", "Ño ñandú  muy   bueno", "abc\tdef\nghi", "x" * 40,
                "El baño estaba SUCIO 😡", "precio: 100.000 COP", "it was great, gracias"] + \
               [" ".join(p) for p in itertools.islice(itertools.permutations(["muy", "buen", "hotel"]), 3)]
