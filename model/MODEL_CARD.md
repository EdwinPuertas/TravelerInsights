# Model card · Traveler Insights

- **Tarea:** clasificación de reseñas de viajeros (negativo, neutral, positivo).
- **Arquitectura:** TF-IDF de palabras [1, 2] y caracteres [3, 5] + rasgos léxicos del curso (core/lexical.py, escalados) → Regresión logística (C=4.0, pesos de clase balanceados). Servido como JSON sin scikit-learn.
- **Datos:** competencia Kaggle *Traveler Insights UPTC 2026*; huella `ab6775f3763794c6`; 809 reseñas tras limpieza.
- **Desempeño en prueba:** macro-F1 0.6046 (IC 95 % [0.3641, 0.7979]), exactitud 0.619, ECE 0.2015.
- **Uso previsto:** apoyo a un equipo humano que clasifica y responde reseñas. Las respuestas son sugerencias.
- **Usos no previstos:** decisiones automáticas sobre personas, reseñas en otros idiomas o dominios.
- **Limitaciones:** ironía, negación y reseñas mixtas; el léxico de aspectos no cubre todo; el desempeño puede degradarse con la deriva de los datos.
- **Responsabilidad:** las reseñas pueden contener datos personales; no se almacenan en la API.
- **Generado:** 2026-11-11T02:18:36+00:00 con scikit-learn 1.9.1.
