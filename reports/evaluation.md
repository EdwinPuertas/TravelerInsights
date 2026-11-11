# Informe de evaluación

Generado: 2026-11-11T03:05:36+00:00 · datos `ab6775f37637` · entrenamiento 809 (incluye 0 de retroalimentación humana) · prueba 809

**Balanceo:** none · **datos externos (solo entrenamiento):** 0 

**Evaluación:** fuera de pliegue (validación cruzada estratificada, 5 pliegues × 5 repeticiones; cada reseña se predice con un modelo que no la vio) · macro-F1 0.7422 ± 0.0129 · micro-F1 0.89 · soporte por clase {'negativo': 22, 'neutral': 630, 'positivo': 157}

**Candidato en prueba:** macro-F1 0.7473 · micro-F1 0.8888 (= exactitud)

## Compuerta de calidad: APROBADO ✅
- ✅ **min_macro_f1**: macro-F1 0.7473 ≥ 0.5
- ✅ **beats_dummy**: 0.7473 − dummy 0.2919 ≥ 0.1
- ✅ **min_macro_f1_repetido**: media 0.7422 ± 0.0129 en 5 repeticiones de 5 pliegues ≥ 0.5
- ✅ **no_regression**: 0.7473 ≥ producción 0.7473 − 0.01
- ✅ **json_parity**: error máx. 3.98e-07 ≤ 0.0001

## Comparación con baselines (macro-F1)
| Modelo | Validación cruzada | Prueba |
|---|---|---|
| tfidf_logreg_sin_lexico | 0.7262 ± 0.0161 | 0.7037 |
| dummy_mayoritaria | 0.2919 ± 0.0000 | 0.2919 |
| tfidf_naive_bayes | 0.3242 ± 0.0037 | 0.3258 |
| candidate (TF-IDF + LogReg) | 0.7422 ± 0.0129 | 0.7473 |

- IC 95 % por bootstrap del macro-F1 del candidato: [0.6681, 0.8107]
- McNemar frente a Naive Bayes: solo el candidato acierta 130, solo NB acierta 49, p = 0.0
- Calibración: ECE = 0.0223
- **Ablación de rasgos léxicos:** macro-F1 con léxico 0.7473 vs. sin léxico 0.7037; McNemar: solo con léxico acierta 28, solo sin léxico acierta 32, p = 0.698883

## Regla de seguridad (revisión humana de posibles negativas)
Con τ = 0.7, sin regla 5 de 22 negativas reales se clasificarían como otra clase y se responderían automáticamente (revisión humana 14.3%). Con la regla (p(negativo) > 0.15 o emoción enojo, disgusto): 4 automatizadas, revisión humana 15.7%.

| Umbral p(negativo) | Negativas mal clasificadas y automatizadas | Revisión humana |
|---|---|---|
| 0.05 | 3 / 22 | 22.2% |
| 0.1 | 4 / 22 | 17.4% |
| 0.15 | 4 / 22 | 15.7% |
| 0.2 | 5 / 22 | 14.7% |
| 0.3 | 5 / 22 | 14.7% |
| 0.5 | 5 / 22 | 14.7% |

## Desempeño por clase (prueba)
| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| negativo | 0.667 | 0.455 | 0.541 | 22 |
| neutral | 0.939 | 0.922 | 0.930 | 630 |
| positivo | 0.731 | 0.815 | 0.771 | 157 |

## Desempeño por longitud de reseña
| Grupo | n | macro-F1 |
|---|---|---|
| cortas | 289 | 0.7437 |
| medias | 251 | 0.5982 |
| largas | 269 | 0.7175 |
