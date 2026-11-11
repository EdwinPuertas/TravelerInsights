"""Almacenamiento de historial y retroalimentación humana en Upstash Redis (REST, sin dependencias).

Se activa con las variables que crea la integración de Vercel → Storage → Upstash for Redis:
``KV_REST_API_URL`` y ``KV_REST_API_TOKEN`` (o ``UPSTASH_REDIS_REST_URL``/``UPSTASH_REDIS_REST_TOKEN``).
Sin ellas, ``configurado()`` es False y la app guarda el historial en el navegador.
"""
from __future__ import annotations

import json
import os
import urllib.request

HISTORIAL, FEEDBACK = "ti:historial", "ti:feedback"
MAX_HISTORIAL = 500


def _cred():
    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
    tok = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    return url, tok


def configurado() -> bool:
    return all(_cred())


def _cmd(*args):
    url, tok = _cred()
    req = urllib.request.Request(url, data=json.dumps(list(args)).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["result"]


def guardar_historial(registro: dict) -> None:
    _cmd("LPUSH", HISTORIAL, json.dumps(registro, ensure_ascii=False))
    _cmd("LTRIM", HISTORIAL, 0, MAX_HISTORIAL - 1)


def historial(limite: int = 50) -> list[dict]:
    return [json.loads(x) for x in _cmd("LRANGE", HISTORIAL, 0, limite - 1)]


def guardar_feedback(registro: dict) -> None:
    _cmd("LPUSH", FEEDBACK, json.dumps(registro, ensure_ascii=False))


def feedback() -> list[dict]:
    return [json.loads(x) for x in _cmd("LRANGE", FEEDBACK, 0, -1)]
