# Prueba_emociones

Pequeño paquete para experimentar con clasificación de intensidad emocional en
texto. Incluye una CLI y utilidades para entrenar rápidamente usando TF-IDF y
regresión logística sobre datasets locales o de Hugging Face.

## Instalación

```bash
pip install -e .[datasets]
```

- ``[datasets]`` habilita la descarga automática de conjuntos de datos desde
  Hugging Face. Si ya cuentas con tus propios CSV puedes omitirlo.

## Uso de la CLI

### Entrenamiento

Descargando el dataset ``stepp1/tweet_emotion_intensity``:

```bash
prueba-emociones train \
  --dataset stepp1/tweet_emotion_intensity \
  --split train \
  --text-column tweet \
  --label-column intensity \
  --limit 500 \
  --model-out models/emotions.joblib
```

Entrenamiento con un CSV propio:

```bash
prueba-emociones train --data-file datos.csv --text-column texto --label-column intensidad
```

### Predicción

```bash
prueba-emociones predict --model models/emotions.joblib --text "I absolutely love this!"
```

Usando un archivo de texto (una línea = un ejemplo):

```bash
prueba-emociones predict --model models/emotions.joblib --file ejemplos.txt --probabilities
```

## Uso en Python

```python
from prueba_emociones import EmotionModel, load_hf_dataset

# Cargar un subset pequeño para iteraciones rápidas

df, text_col, label_col = load_hf_dataset(
    "stepp1/tweet_emotion_intensity",
    split="train",
    text_column="tweet",
    label_column="intensity",
    limit=500,
)

model = EmotionModel()
result = model.fit(df[text_col], df[label_col])
print(f"Accuracy: {result.accuracy:.3f}")
print(result.report)

print(model.predict(["Amazing work!", "Not great at all"]))
```

## Ejemplo interactivo

Consulta ``examples/analysis.ipynb`` para ver un flujo completo de carga de
datos, entrenamiento y predicción usando este paquete.

## Documentación

La carpeta ``docs/`` contiene una guía rápida y descripción de la API. Puedes
levantar la documentación local con:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Licencia

MIT
