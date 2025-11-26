"""Feature extraction utilities for emotional text analysis."""
from .feature_pipeline import FeaturePipeline, build_feature_pipeline
from .intensity_flags import IntensityFlagTransformer
from .vectorizers import build_tfidf_vectorizer

__all__ = [
    "FeaturePipeline",
    "build_feature_pipeline",
    "IntensityFlagTransformer",
    "build_tfidf_vectorizer",
]
