"""Random Forest model utilities for training, evaluation and persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}


def train_model(X, y, **kwargs) -> RandomForestClassifier:
    """Train a RandomForestClassifier with reproducible defaults.

    Args:
        X: Feature matrix.
        y: Target labels.
        **kwargs: Extra hyperparameters to override defaults.

    Returns:
        Trained RandomForestClassifier instance.
    """

    params = {**DEFAULT_PARAMS, **kwargs}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestClassifier, X_test, y_test) -> Dict[str, float]:
    """Evaluate a trained model on test data.

    Args:
        model: Trained RandomForestClassifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with accuracy, precision, recall and f1 metrics.
    """

    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }


def save_model(model: RandomForestClassifier, path: Path | str) -> Path:
    """Persist a model to disk using joblib.

    Args:
        model: Trained RandomForestClassifier to save.
        path: Destination filepath.

    Returns:
        Path to the saved model file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: Path | str) -> RandomForestClassifier:
    """Load a model from disk.

    Args:
        path: Filepath to the serialized model.

    Returns:
        Deserialized RandomForestClassifier instance.
    """

    return joblib.load(path)
