# Informe de evaluación

Generado: 2026-11-10T17:56:02+00:00 · datos `ab6775f37637` · entrenamiento 647 · prueba 162

## Compuerta de calidad: APROBADO ✅
- ✅ **min_macro_f1**: macro-F1 0.6414 ≥ 0.5
- ✅ **beats_dummy**: 0.6414 − dummy 0.2917 ≥ 0.1
- ✅ **json_parity**: error máx. 3.59e-07 ≤ 0.0001

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
| dummy_mayoritaria | 0.2919 ± 0.0007 | 0.2917 |
| tfidf_naive_bayes | 0.3135 ± 0.0110 | 0.3329 |
| candidate (TF-IDF + LogReg) | 0.6924 ± 0.0587 | 0.6414 |

- IC 95 % por bootstrap del macro-F1 del candidato: [0.4852, 0.8157]
- McNemar frente a Naive Bayes: solo el candidato acierta 17, solo NB acierta 4, p = 0.007197
- Calibración: ECE = 0.0548

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
| cortas | 56 | 0.4678 |
| medias | 54 | 0.3205 |
| largas | 52 | 1.0 |
