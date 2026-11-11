"""BLEU y ROUGE en Python puro, para comparar la respuesta sugerida con la editada por una persona.

- BLEU (Papineni et al., 2002): precisión de n-gramas (1–4) con penalización por brevedad;
  a nivel de oración se usa suavizado +1 (Lin y Och, 2004).
- ROUGE (Lin, 2004): ROUGE-1 y ROUGE-2 (n-gramas) y ROUGE-L (subsecuencia común más larga), en F1.
Miden coincidencia superficial con la referencia, no si la respuesta es adecuada.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .text import normalizar

_PAL = re.compile(r"\w+")


def tokens(texto: str) -> list[str]:
    return _PAL.findall(normalizar(texto))


def _ngramas(tok, n):
    return Counter(tuple(tok[i:i + n]) for i in range(len(tok) - n + 1))


def bleu(candidato: str, referencia: str, max_n: int = 4) -> float:
    c, r = tokens(candidato), tokens(referencia)
    if not c or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cn, rn = _ngramas(c, n), _ngramas(r, n)
        coincide = sum(min(v, rn[g]) for g, v in cn.items())
        total = max(sum(cn.values()), 0)
        if n == 1:
            if coincide == 0:
                return 0.0
            p = coincide / total
        else:
            p = (coincide + 1) / (total + 1)       # suavizado +1 para n > 1
        log_p += math.log(p) / max_n
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return round(bp * math.exp(log_p), 4)


def _f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def rouge_n(candidato: str, referencia: str, n: int) -> float:
    cn, rn = _ngramas(tokens(candidato), n), _ngramas(tokens(referencia), n)
    if not cn or not rn:
        return 0.0
    coincide = sum((cn & rn).values())
    return round(_f1(coincide / sum(cn.values()), coincide / sum(rn.values())), 4)


def rouge_l(candidato: str, referencia: str) -> float:
    a, b = tokens(candidato), tokens(referencia)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    return round(_f1(lcs / len(a), lcs / len(b)), 4)


def todas(candidato: str, referencia: str) -> dict:
    return {"bleu": bleu(candidato, referencia), "rouge1": rouge_n(candidato, referencia, 1),
            "rouge2": rouge_n(candidato, referencia, 2), "rougeL": rouge_l(candidato, referencia)}
