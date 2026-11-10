# Informe de evaluación

Generado: 2026-11-10T17:49:38+00:00 · datos `ab6775f37637` · entrenamiento 647 · prueba 162

## Compuerta de calidad: APROBADO ✅
- ✅ **min_macro_f1**: macro-F1 0.9424 ≥ 0.5
- ✅ **beats_dummy**: 0.9424 − dummy 0.4296 ≥ 0.1
- ✅ **json_parity**: error máx. 2.58e-07 ≤ 0.0001

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
| dummy_mayoritaria | 0.4300 ± 0.0009 | 0.4296 |
| tfidf_naive_bayes | 0.4611 ± 0.0397 | 0.4555 |
| candidate (TF-IDF + LogReg) | 0.9154 ± 0.0387 | 0.9424 |

- IC 95 % por bootstrap del macro-F1 del candidato: [0.8949, 0.9828]
- McNemar frente a Naive Bayes: solo el candidato acierta 36, solo NB acierta 4, p = 0.0
- Calibración: ECE = 0.0901

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
| cortas | 58 | 0.9254 |
| medias | 50 | 0.7696 |
| largas | 54 | 1.0 |
