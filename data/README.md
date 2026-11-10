# Datos

Los datos vienen de la competencia de Kaggle **Traveler Insights UPTC 2026** y **no se versionan** en este repositorio.

```bash
# Requiere aceptar las reglas de la competencia en Kaggle y tener ~/.kaggle/kaggle.json
kaggle competitions download -c traveler-insights-uptc-2026 -p data/raw
cd data/raw && unzip -o *.zip
```

El pipeline espera `data/raw/train.csv` (ruta configurable en `configs/params.yaml`).
