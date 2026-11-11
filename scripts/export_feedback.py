"""Exporta la retroalimentación humana de Upstash Redis a data/feedback/feedback.csv (la usa el workflow Train)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traveler_insights.core import store  # noqa: E402

if not store.configurado():
    print("Sin credenciales de Upstash: no se exporta retroalimentación.")
    sys.exit(0)
items = store.feedback()                    # más reciente primero
out = Path("data/feedback/feedback.csv")
out.parent.mkdir(parents=True, exist_ok=True)
campos = ["fecha", "texto", "clase_predicha", "clase_correcta", "confianza", "respuesta_sugerida", "respuesta_editada"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
    w.writeheader()
    w.writerows(items)
corr = sum(1 for i in items if i.get("clase_predicha") != i.get("clase_correcta"))
print(f"Exportadas {len(items)} retroalimentaciones ({corr} correcciones de clase) → {out}")
