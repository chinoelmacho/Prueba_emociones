"""Transformers for counting intensifiers and negations in text."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_INTENSIFIERS: Sequence[str] = (
    "muy",
    "extremadamente",
    "sumamente",
    "increíblemente",
    "bastante",
    "totalmente",
    "absolutamente",
    "super",
    "re",
    "hiper",
    "demasiado",
)
DEFAULT_NEGATORS: Sequence[str] = (
    "no",
    "nunca",
    "jamás",
    "ninguno",
    "nadie",
    "nada",
    "tampoco",
    "sin",
)


class IntensityFlagTransformer(BaseEstimator, TransformerMixin):
    """Counts intensifiers and negations inside text samples."""

    def __init__(
        self,
        intensifiers: Iterable[str] | None = None,
        negators: Iterable[str] | None = None,
    ) -> None:
        self.intensifiers = tuple(intensifiers) if intensifiers is not None else DEFAULT_INTENSIFIERS
        self.negators = tuple(negators) if negators is not None else DEFAULT_NEGATORS

    def fit(self, X: Sequence[str], y: Sequence[str] | None = None) -> "IntensityFlagTransformer":
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        values: List[List[float]] = []
        for text in X:
            tokens = self._tokenize(text)
            intensifier_count = float(sum(token in self.intensifiers for token in tokens))
            negation_count = float(sum(token in self.negators for token in tokens))
            values.append([intensifier_count, negation_count])
        return np.asarray(values, dtype=np.float64)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:  # type: ignore[override]
        return np.asarray(["intensifier_count", "negation_count"], dtype=str)

    @staticmethod
    def _tokenize(text: str | None) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[\w']+\b", text.lower())
