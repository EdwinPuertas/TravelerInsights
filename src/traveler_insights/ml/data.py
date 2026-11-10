"""Fase 2 de CRISP-ML(Q): preparación y validación de datos."""
from __future__ import annotations

import re

import pandas as pd

from ..core.text import normalizar

NOMBRES_ETIQUETA = re.compile(r"sentim|label|etiqueta|target|clase|polari")


def detectar_columnas(df: pd.DataFrame, text_col=None, label_col=None) -> tuple[str, str]:
    obj = [c for c in df.columns
           if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if text_col is None:
        if not obj:
            raise ValueError("No hay columnas de texto: defina data.text_col en params.yaml")
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    if label_col is None:
        cand = [c for c in df.columns if c != text_col and 2 <= df[c].nunique() <= 20
                and not str(c).lower().startswith("id")]
        if not cand:
            raise ValueError("No hay columna de etiqueta: defina data.label_col en params.yaml")
        # Preferir nombres típicos de etiqueta; si no hay, la columna con menos categorías
        preferidas = [c for c in cand if NOMBRES_ETIQUETA.search(str(c).lower())]
        label_col = preferidas[0] if preferidas else min(cand, key=lambda c: df[c].nunique())
    for c in (text_col, label_col):
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe. Columnas disponibles: {list(df.columns)}")
    return text_col, label_col


def preparar(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Limpia los datos y devuelve (datos, reporte de calidad). Lanza error si no pasan los requisitos."""
    tcol, lcol = detectar_columnas(df_raw, cfg.get("text_col"), cfg.get("label_col"))
    d = df_raw[[tcol, lcol]].rename(columns={tcol: "texto", lcol: "etiqueta"})
    n0 = len(d)
    d = d.dropna()
    d["texto"] = d["texto"].astype(str)
    d = d[d["texto"].str.strip() != ""]
    n_validas = len(d)
    norm = d["texto"].map(normalizar)
    dup_rate = float(norm.duplicated().mean()) if n_validas else 0.0
    conflictos = d.assign(_n=norm).groupby("_n")["etiqueta"].nunique()
    conflict_rate = float((conflictos > 1).sum() / max(len(conflictos), 1))
    # Se eliminan duplicados y textos con etiquetas contradictorias
    d = d.assign(_n=norm)
    d = d[~d["_n"].isin(conflictos[conflictos > 1].index)]
    d = d.drop_duplicates(subset="_n").drop(columns="_n")
    d["etiqueta"] = d["etiqueta"].astype(str)
    d = d.reset_index(drop=True)

    dist = d["etiqueta"].value_counts().to_dict()
    q = cfg["quality"]
    checks = {
        "min_rows": (len(d) >= q["min_rows"], f"{len(d)} filas (mínimo {q['min_rows']})"),
        "min_per_class": (min(dist.values()) >= q["min_per_class"] if dist else False,
                          f"clase menor con {min(dist.values()) if dist else 0} ejemplos (mínimo {q['min_per_class']})"),
        "max_duplicate_rate": (dup_rate <= q["max_duplicate_rate"], f"duplicados {dup_rate:.1%} (máximo {q['max_duplicate_rate']:.0%})"),
        "max_conflict_rate": (conflict_rate <= q["max_conflict_rate"], f"conflictos {conflict_rate:.1%} (máximo {q['max_conflict_rate']:.0%})"),
        "n_classes": (len(dist) >= 2, f"{len(dist)} clases"),
    }
    reporte = {
        "columns": {"text": tcol, "label": lcol},
        "rows_raw": n0, "rows_valid": n_validas, "rows_final": len(d),
        "duplicate_rate": round(dup_rate, 4), "conflict_rate": round(conflict_rate, 4),
        "class_distribution": dist,
        "imbalance_ratio": round(max(dist.values()) / min(dist.values()), 2) if dist else None,
        "checks": {k: {"passed": bool(ok), "detail": msg} for k, (ok, msg) in checks.items()},
    }
    reporte["passed"] = all(c["passed"] for c in reporte["checks"].values())
    return d, reporte
