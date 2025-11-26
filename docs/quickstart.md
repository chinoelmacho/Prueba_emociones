# Guía rápida

## Instalación

```bash
pip install -e .[datasets]
```

Si no necesitas descargar datasets de Hugging Face puedes omitir ``[datasets]``.

## Entrenamiento desde la línea de comandos

```bash
prueba-emociones train \
  --dataset stepp1/tweet_emotion_intensity \
  --split train \
  --text-column tweet \
  --label-column intensity \
  --limit 500 \
  --model-out models/emotions.joblib
```

Para datos locales:

```bash
prueba-emociones train --data-file datos.csv --text-column texto --label-column intensidad
```

## Predicción desde la línea de comandos

```bash
prueba-emociones predict --model models/emotions.joblib --text "I absolutely love this!"
```

## Uso en Python

```python
from prueba_emociones.data import load_hf_dataset
from prueba_emociones.model import EmotionModel

# Cargar datos
df, text_col, label_col = load_hf_dataset(
    "stepp1/tweet_emotion_intensity", split="train", text_column="tweet", label_column="intensity", limit=500
)

model = EmotionModel()
result = model.fit(df[text_col], df[label_col])
print("Accuracy:", result.accuracy)

predicciones = model.predict(["Amazing work!", "Not great at all"])
print(predicciones)
```
