"""Normalización y extracción de n-gramas.

Replica exactamente el comportamiento de ``TfidfVectorizer`` de scikit-learn para
``analyzer="word"`` y ``analyzer="char_wb"``. Así el modelo se entrena con
scikit-learn y se sirve en producción sin depender de scikit-learn.
La equivalencia se verifica en ``tests/test_parity.py``.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_WHITE_SPACES = re.compile(r"\s\s+")          # igual que sklearn
_TOKEN = re.compile(r"(?u)\b\w\w+\b")          # token_pattern por defecto de sklearn


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y espacios colapsados."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return _WS.sub(" ", t).strip()


def word_ngrams(texto_normalizado: str, min_n: int, max_n: int) -> list[str]:
    tokens = _TOKEN.findall(texto_normalizado)
    out: list[str] = []
    if min_n == 1:
        out.extend(tokens)
        min_n = 2
    n_tok = len(tokens)
    for n in range(min_n, min(max_n + 1, n_tok + 1)):
        for i in range(n_tok - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


def char_wb_ngrams(texto_normalizado: str, min_n: int, max_n: int) -> list[str]:
    texto = _WHITE_SPACES.sub(" ", texto_normalizado)
    out: list[str] = []
    for w in texto.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:   # palabra más corta que n
                break
    return out
