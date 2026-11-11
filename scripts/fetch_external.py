"""Descarga reseñas REALES de otros corpus en español para complementar el entrenamiento.

Fuente incluida (descarga pública desde GitHub):
  OpeNER-es (Agerri et al., 2013), reseñas de hoteles anotadas a nivel de oración y
  redistribuidas en el repositorio de SemEval-2022 Task 10 (Barnes et al., 2022).
  Se etiqueta una oración como "negativo" si TODAS sus expresiones de opinión son
  negativas, y "positivo" si todas son positivas. Las mixtas y sin opinión se descartan.

Por defecto solo se toman negativas (la clase con 22 ejemplos en Kaggle).
Las salidas van a data/external/ (no se versionan; ver .gitignore).

Uso:  python scripts/fetch_external.py [--clases negativo positivo] [--max-por-clase 400]
Verifique la licencia de cada corpus antes de un uso comercial.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/jerbarnes/semeval22_structured_sentiment/master/data/opener_es/{}.json"


def opener_es(clases: set[str]) -> pd.DataFrame:
    filas = []
    for sp in ("train", "dev", "test"):
        with urllib.request.urlopen(BASE.format(sp), timeout=60) as r:
            datos = json.loads(r.read().decode("utf-8"))
        for x in datos:
            pols = {o["Polarity"] for o in x["opinions"]}
            et = "negativo" if pols == {"Negative"} else "positivo" if pols == {"Positive"} else None
            texto = " ".join(x["text"].split())
            if et in clases and len(texto.split()) >= 4:
                filas.append({"texto": texto, "etiqueta": et, "fuente": "opener_es"})
    return pd.DataFrame(filas).drop_duplicates("texto")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clases", nargs="+", default=["negativo"])
    ap.add_argument("--max-por-clase", type=int, default=0, help="0 = sin límite")
    ap.add_argument("--out", default="data/external/opener_es.csv")
    a = ap.parse_args()
    df = opener_es(set(a.clases))
    if a.max_por_clase:
        df = df.groupby("etiqueta", group_keys=False).apply(lambda g: g.sample(min(len(g), a.max_por_clase), random_state=42))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} oraciones externas → {a.out} {df['etiqueta'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
