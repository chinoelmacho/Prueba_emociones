# API

## ``prueba_emociones.data``

### ``load_hf_dataset(name="stepp1/tweet_emotion_intensity", split="train", text_column=None, label_column=None, limit=None)``
Descarga un dataset de Hugging Face y devuelve un ``DataFrame`` junto a los nombres
de las columnas usadas.

## ``prueba_emociones.model``

### ``EmotionModel``
- ``fit(texts, labels, test_size=0.2, random_state=42)``: entrena y devuelve ``TrainingResult``.
- ``predict(texts)``: devuelve etiquetas predichas.
- ``predict_proba(texts)``: probabilidades por clase (si el clasificador lo soporta).
- ``save(path)`` / ``load(path)``: serialización del pipeline.

### ``train_model(data, text_column, label_column, test_size=0.2, random_state=42)``
Atajo para crear un ``EmotionModel`` y entrenarlo desde un ``DataFrame``.
