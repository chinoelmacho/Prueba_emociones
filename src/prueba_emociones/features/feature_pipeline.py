"""Pipeline helpers for converting raw text into feature matrices."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse

from .intensity_flags import IntensityFlagTransformer
from .vectorizers import build_tfidf_vectorizer


class FeaturePipeline:
    """Combine sparse lexical features with simple rule-based indicators."""

    def __init__(
        self,
        *,
        vectorizer=None,
        intensity_transformer=None,
    ) -> None:
        self.vectorizer = vectorizer if vectorizer is not None else build_tfidf_vectorizer()
        self.intensity_transformer = (
            intensity_transformer if intensity_transformer is not None else IntensityFlagTransformer()
        )

    def fit(self, X: Sequence[str], y=None) -> "FeaturePipeline":
        self.vectorizer.fit(X)
        self.intensity_transformer.fit(X)
        return self

    def transform(self, X: Sequence[str]):
        lexical_features = self.vectorizer.transform(X)
        rule_features = sparse.csr_matrix(self.intensity_transformer.transform(X))
        return sparse.hstack([lexical_features, rule_features], format="csr")

    def fit_transform(self, X: Sequence[str], y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        names = list(self.vectorizer.get_feature_names_out())
        names.extend(self.intensity_transformer.get_feature_names_out())
        return np.asarray(names)


def build_feature_pipeline() -> FeaturePipeline:
    """Return a ready-to-use :class:`FeaturePipeline` instance."""

    return FeaturePipeline()
