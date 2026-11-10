# Fase 3 · Modelado

**Código:** `src/traveler_insights/ml/train.py`

## Modelos
| Modelo | Papel |
|---|---|
| Clase mayoritaria (`DummyClassifier`) | Piso mínimo |
| TF-IDF + Naive Bayes | Baseline clásico (Jurafsky & Martin, cap. 4) |
| **TF-IDF palabras (1–2) + caracteres (3–5) + regresión logística** | Candidato: los n-gramas de caracteres toleran errores ortográficos y variación del español; es exportable a JSON |

## Protocolo
- Partición de prueba estratificada separada **al inicio**; se usa una sola vez.
- Validación cruzada estratificada (k = 5) dentro del entrenamiento, con semilla fija.
- Mismas particiones para todos los modelos.
- Hiperparámetros solo en `configs/params.yaml`.

## Criterios de aceptación
- [ ] El candidato supera a los baselines en validación cruzada.
- [ ] Variación entre pliegues reportada (media ± desviación).
