"""Lógica de negocio: aspectos, polaridad, respuestas, guardarraíl y enrutamiento.

Python puro: la usan tanto el pipeline de entrenamiento como la API desplegada.
"""
from __future__ import annotations

import re

from .text import normalizar

ASPECTOS = {
    "limpieza": r"limpi|suci|aseo|polvo|olor|higien|bano",
    "servicio": r"servicio|atenci|personal|amabl|grosero|recepci|mesero|guia",
    "precio": r"precio|caro|barat|costo|cobr|tarifa|valor|dinero|plata",
    "ubicacion": r"ubicaci|ubicad|cerca|lejos|acceso|centro|transporte|vista",
    "comida": r"comida|desayun|almuerz|cena|restaurante|plato|sabor|delicios",
    "instalaciones": r"habitaci|cama|piscina|wifi|internet|instalacion|ruido|ducha|aire",
    "experiencia": r"paisaje|tour|recorrido|experiencia|visita|actividad|cultura|historia",
}
_PAT = {a: re.compile(p) for a, p in ASPECTOS.items()}
NOMBRE_ASPECTO = {
    "limpieza": "la limpieza", "servicio": "la atención de nuestro equipo",
    "precio": "la relación calidad-precio", "ubicacion": "la ubicación", "comida": "la comida",
    "instalaciones": "las instalaciones", "experiencia": "la experiencia y las actividades",
}
COMPROMISOS = re.compile(
    r"reembols|devoluci|descuento|compensa|gratis|garantiz|te devolvemos|en \d+ (dias|horas)")


def detectar_aspectos(texto: str) -> list[str]:
    t = normalizar(texto)
    return [a for a, p in _PAT.items() if p.search(t)]


def mapa_polaridad(clases: list[str]) -> dict[str, str]:
    """Deduce la polaridad de cada clase a partir de su nombre. Revísala siempre."""
    clases = [str(c) for c in clases]
    numericas = all(re.fullmatch(r"\d+(\.\d+)?", c) for c in clases)
    binaria = numericas and set(map(float, clases)) <= {0.0, 1.0}
    out = {}
    for c in clases:
        n = normalizar(c)
        if binaria:
            out[c] = "positiva" if float(n) == 1 else "negativa"
        elif numericas:
            v = float(n)
            out[c] = "negativa" if v <= 2 else ("neutra" if v == 3 else "positiva")
        elif re.search(r"neg|malo|mala|queja", n):
            out[c] = "negativa"
        elif re.search(r"pos|bueno|buena", n):
            out[c] = "positiva"
        else:
            out[c] = "neutra"
    return out


def _unir(items: list[str]) -> str:
    nombres = [NOMBRE_ASPECTO[a] for a in items]
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def respuesta_plantilla(texto: str, polaridad: str) -> str:
    asp = detectar_aspectos(texto)
    menc = f" sobre {_unir(asp)}" if asp else ""
    if polaridad == "positiva":
        return f"¡Gracias por tu reseña! Nos alegra que hayas valorado{menc or ' tu visita'}. Te esperamos de nuevo."
    if polaridad == "negativa":
        return (f"Lamentamos tu experiencia{menc}. Compartimos tu comentario con el área responsable "
                "y un integrante de nuestro equipo se comunicará contigo.")
    return f"Gracias por tus comentarios{menc}. Tus observaciones nos ayudan a mejorar."


def requiere_revision(respuesta: str) -> bool:
    return bool(COMPROMISOS.search(normalizar(respuesta)))


def analizar(modelo, textos: list[str], tau: float = 0.7, polaridad: dict | None = None) -> list[dict]:
    """Clasifica, detecta aspectos, sugiere respuesta y decide el enrutamiento."""
    polaridad = polaridad or modelo.metadata.get("polarity") or mapa_polaridad(modelo.classes)
    resultados = []
    for texto, p in zip(textos, modelo.predict_proba(textos)):
        k = max(range(len(p)), key=p.__getitem__)
        clase, conf = modelo.classes[k], float(p[k])
        pol = polaridad.get(clase, "neutra")
        resp = respuesta_plantilla(texto, pol)
        motivos = []
        if conf < tau:
            motivos.append("confianza baja")
        if requiere_revision(resp):
            motivos.append("respuesta con compromisos")
        resultados.append({
            "texto": texto, "clase": clase, "confianza": round(conf, 4), "polaridad": pol,
            "aspectos": detectar_aspectos(texto), "respuesta_sugerida": resp,
            "decision": "revision_humana" if motivos else "automatizar", "motivos": motivos,
        })
    return resultados
