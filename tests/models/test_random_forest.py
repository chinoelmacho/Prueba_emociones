import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prueba_emociones.models.random_forest import (  # noqa: E402
    evaluate_model,
    load_model,
    save_model,
    train_model,
)


def test_train_and_evaluate_random_forest():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    model = train_model(X_train, y_train, n_estimators=20)
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_model_serialization_roundtrip(tmp_path):
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=99,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=99
    )

    model = train_model(X_train, y_train, n_estimators=15, random_state=99)
    model_path = tmp_path / "random_forest.joblib"

    saved_path = save_model(model, model_path)
    assert saved_path.exists()

    loaded_model = load_model(saved_path)

    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)
    np.testing.assert_array_equal(original_predictions, loaded_predictions)

    metrics = evaluate_model(loaded_model, X_test, y_test)
    assert metrics["accuracy"] >= 0.5
