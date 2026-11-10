# Informe de evaluación

Generado: 2026-11-10T18:12:52+00:00 · datos `ab6775f37637` · entrenamiento 647 · prueba 162

## Compuerta de calidad: APROBADO ✅
- ✅ **min_macro_f1**: macro-F1 0.7665 ≥ 0.5
- ✅ **beats_dummy**: 0.7665 − dummy 0.2917 ≥ 0.1
- ✅ **no_regression**: 0.7665 ≥ producción 0.6414 − 0.01
- ✅ **json_parity**: error máx. 2.19e-07 ≤ 0.0001

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
| tfidf_logreg_sin_lexico | 0.6924 ± 0.0587 | 0.6414 |
| dummy_mayoritaria | 0.2919 ± 0.0007 | 0.2917 |
| tfidf_naive_bayes | 0.3135 ± 0.0110 | 0.3329 |
| candidate (TF-IDF + LogReg) | 0.7173 ± 0.0837 | 0.7665 |

- IC 95 % por bootstrap del macro-F1 del candidato: [0.5547, 0.9067]
- McNemar frente a Naive Bayes: solo el candidato acierta 26, solo NB acierta 8, p = 0.002935
- Calibración: ECE = 0.0323
- **Ablación de rasgos léxicos:** macro-F1 con léxico 0.7665 vs. sin léxico 0.6414; McNemar: solo con léxico acierta 10, solo sin léxico acierta 5, p = 0.301758

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
| cortas | 56 | 0.5273 |
| medias | 54 | 0.6533 |
| largas | 52 | 1.0 |
