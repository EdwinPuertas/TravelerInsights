# Informe de evaluación

Generado: 2026-11-11T02:18:36+00:00 · datos `ab6775f37637` · entrenamiento 788 (incluye 0 de retroalimentación humana) · prueba 21

**Partición de prueba:** balanced · soporte por clase {'negativo': 7, 'neutral': 7, 'positivo': 7}

**Evaluación repetida (30 particiones balanceadas de 7 por clase):** candidate macro-F1 0.6707 ± 0.0946 [0.5305, 0.8717] · dummy_mayoritaria macro-F1 0.1667 ± 0.0 [0.1667, 0.1667]

**Candidato en prueba:** macro-F1 0.6046 · micro-F1 0.619 (= exactitud)

## Compuerta de calidad: APROBADO ✅
- ✅ **min_macro_f1**: macro-F1 0.6046 ≥ 0.5
- ✅ **beats_dummy**: 0.6046 − dummy 0.1667 ≥ 0.1
- ✅ **min_macro_f1_repetido**: media 0.6707 en 30 pruebas balanceadas ≥ 0.5
- ✅ **no_regression**: no comparable: cambió la partición de prueba (None → {'test_split': 'balanced', 'test_per_class': 7, 'seed': 42}); revise el PR
- ✅ **json_parity**: error máx. 1.25e-07 ≤ 0.0001

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
| tfidf_logreg_sin_lexico | 0.6928 ± 0.0646 | 0.5605 |
| dummy_mayoritaria | 0.2944 ± 0.0001 | 0.1667 |
| tfidf_naive_bayes | 0.3369 ± 0.0291 | 0.1667 |
| candidate (TF-IDF + LogReg) | 0.6672 ± 0.0905 | 0.6046 |

- IC 95 % por bootstrap del macro-F1 del candidato: [0.3641, 0.7979]
- McNemar frente a Naive Bayes: solo el candidato acierta 7, solo NB acierta 1, p = 0.070312
- Calibración: ECE = 0.2015
- **Ablación de rasgos léxicos:** macro-F1 con léxico 0.6046 vs. sin léxico 0.5605; McNemar: solo con léxico acierta 2, solo sin léxico acierta 2, p = 1.0

## Desempeño por clase (prueba)
| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| negativo | 1.000 | 0.286 | 0.444 | 7 |
| neutral | 0.462 | 0.857 | 0.600 | 7 |
| positivo | 0.833 | 0.714 | 0.769 | 7 |

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
| cortas | 7 | 0.3636 |
| medias | 7 | 0.6571 |
| largas | 7 | 0.65 |
