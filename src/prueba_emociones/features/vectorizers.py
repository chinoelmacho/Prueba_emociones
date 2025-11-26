"""Helpers for text vectorization."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_VECTOR_PARAMS: Dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "max_features": 5000,
    "min_df": 1,
    "norm": "l2",
    "dtype": float,
}


def build_tfidf_vectorizer(
    *,
    max_features: int | None = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int | float = 1,
    analyzer: str = "word",
) -> TfidfVectorizer:
    """Create a deterministic ``TfidfVectorizer`` with sensible defaults."""

    params = DEFAULT_VECTOR_PARAMS | {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "analyzer": analyzer,
    }
    return TfidfVectorizer(**params)
