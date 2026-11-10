# Fase 3 · Modelado

**Código:** `src/traveler_insights/ml/train.py`

## Modelos
| Modelo | Papel |
|---|---|
| Clase mayoritaria (`DummyClassifier`) | Piso mínimo |
| TF-IDF + Naive Bayes | Baseline clásico (Jurafsky & Martin, cap. 4) |
| **TF-IDF palabras (1–2) + caracteres (3–5) + regresión logística** | Candidato: los n-gramas de caracteres toleran errores ortográficos y variación del español; es exportable a JSON |

## Rasgos léxicos (v1.2)
El candidato une tres bloques: TF-IDF de palabras, TF-IDF de caracteres y **rasgos léxicos** de los modelos anteriores del curso (`core/lexical.py`). Los rasgos léxicos se escalan con `StandardScaler` dentro del pipeline, de modo que el escalador se ajusta en cada pliegue y no hay fuga de información.

Para saber si aportan, el pipeline entrena el mismo modelo **sin** rasgos léxicos (`tfidf_logreg_sin_lexico`) y los compara con la prueba de McNemar (ablación).

## Protocolo
- Partición de prueba estratificada separada **al inicio**; se usa una sola vez.
- Validación cruzada estratificada (k = 5) dentro del entrenamiento, con semilla fija.
- Mismas particiones para todos los modelos.
- Hiperparámetros solo en `configs/params.yaml`.

## Criterios de aceptación
- [ ] El candidato supera a los baselines en validación cruzada.
- [ ] Variación entre pliegues reportada (media ± desviación).
