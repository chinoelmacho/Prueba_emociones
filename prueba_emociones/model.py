"""Modelos clásicos para clasificación de intensidad emocional."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainingResult:
    """Resumen de métricas obtenidas al entrenar."""

    accuracy: float
    report: str
    classes: Sequence[str]


class EmotionModel:
    """Envoltura ligera sobre un ``Pipeline`` de scikit-learn.

    Por defecto se usa ``TfidfVectorizer`` seguido de ``LogisticRegression``.
    """

    def __init__(self, pipeline: Pipeline | None = None):
        if pipeline is None:
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
                ]
            )
        self.pipeline = pipeline

    def fit(
        self,
        texts: Iterable[str],
        labels: Iterable[str],
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingResult:
        """Entrena el modelo y devuelve las métricas de validación."""

        x_train, x_test, y_train, y_test = train_test_split(
            list(texts), list(labels), test_size=test_size, random_state=random_state, stratify=list(labels)
        )
        self.pipeline.fit(x_train, y_train)

        predictions = self.pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        return TrainingResult(accuracy=accuracy, report=report, classes=sorted(set(y_train) | set(y_test)))

    def predict(self, texts: Iterable[str]) -> List[str]:
        """Devuelve una predicción por cada texto."""

        return list(self.pipeline.predict(list(texts)))

    def predict_proba(self, texts: Iterable[str]):
        """Devuelve probabilidades si el clasificador lo permite."""

        clf = self.pipeline.named_steps.get("clf")
        if not hasattr(clf, "predict_proba"):
            raise AttributeError("El clasificador actual no implementa predict_proba")
        return self.pipeline.predict_proba(list(texts))

    def save(self, path: str) -> None:
        """Serializa el pipeline en disco."""

        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "EmotionModel":
        """Carga un pipeline previamente guardado."""

        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)


def train_model(
    data: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[EmotionModel, TrainingResult]:
    """Atajo para entrenar un ``EmotionModel`` a partir de un ``DataFrame``."""

    model = EmotionModel()
    result = model.fit(data[text_column], data[label_column], test_size=test_size, random_state=random_state)
    return model, result
