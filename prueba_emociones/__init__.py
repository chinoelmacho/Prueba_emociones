"""Utilidades básicas para entrenar y probar modelos de intensidad emocional."""
from .data import load_hf_dataset
from .model import EmotionModel, TrainingResult, train_model

__all__ = [
    "EmotionModel",
    "TrainingResult",
    "load_hf_dataset",
    "train_model",
]
