"""Utilities for loading local or remote datasets used in the project."""
from __future__ import annotations

import os
from typing import Optional

from datasets import DownloadMode, Dataset, load_dataset


DATASET_NAME = "stepp1/tweet_emotion_intensity"
ENV_DATASET_PATH = "PRUEBA_EMOCIONES_DATASET_PATH"
ENV_CACHE_DIR = "PRUEBA_EMOCIONES_CACHE_DIR"
ENV_DATA_DIR = "PRUEBA_EMOCIONES_DATA_DIR"
ENV_DOWNLOAD_MODE = "PRUEBA_EMOCIONES_DOWNLOAD_MODE"
ENV_LOCAL_FILES_ONLY = "PRUEBA_EMOCIONES_LOCAL_FILES_ONLY"

VALID_SPLITS = {"train", "validation", "test"}


class InvalidSplitError(ValueError):
    """Raised when an unsupported split is requested."""


def _parse_download_mode(mode: str) -> DownloadMode:
    normalized = mode.strip().lower()
    mapping = {
        "reuse_cache_if_exists": DownloadMode.REUSE_CACHE_IF_EXISTS,
        "reuse_dataset_if_exists": DownloadMode.REUSE_DATASET_IF_EXISTS,
        "force_redownload": DownloadMode.FORCE_REDOWNLOAD,
    }

    if normalized not in mapping:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown download mode '{mode}'. Valid options: {valid}.")

    return mapping[normalized]


def _resolve_bool(value: Optional[str | bool]) -> Optional[bool]:
    if isinstance(value, bool) or value is None:
        return value

    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_tweet_emotion_intensity(
    split: str = "train",
    *,
    dataset_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    download_mode: Optional[str | DownloadMode] = None,
    local_files_only: Optional[bool] = None,
) -> Dataset:
    """Load the Tweet Emotion Intensity dataset with flexible storage options.

    Parameters
    ----------
    split:
        Which dataset split to load. Must be one of ``train``, ``validation`` or
        ``test``.
    dataset_path:
        Optional dataset identifier or path. Overrides the default
        ``stepp1/tweet_emotion_intensity`` and the ``PRUEBA_EMOCIONES_DATASET_PATH``
        environment variable.
    cache_dir:
        Directory to use for the Hugging Face cache. Can also be provided via the
        ``PRUEBA_EMOCIONES_CACHE_DIR`` environment variable.
    data_dir:
        Local directory containing the dataset files. Can also be provided via the
        ``PRUEBA_EMOCIONES_DATA_DIR`` environment variable.
    download_mode:
        Optional ``datasets.DownloadMode`` or string value. String values are
        case-insensitive and may be ``reuse_cache_if_exists``,
        ``reuse_dataset_if_exists`` or ``force_redownload``. Can also be set through
        the ``PRUEBA_EMOCIONES_DOWNLOAD_MODE`` environment variable.
    local_files_only:
        If ``True``, disable remote downloads and only use locally available data
        (including cached data). Can also be set via the
        ``PRUEBA_EMOCIONES_LOCAL_FILES_ONLY`` environment variable.

    Returns
    -------
    datasets.Dataset
        The requested dataset split.

    Raises
    ------
    InvalidSplitError
        If an unsupported split is requested.
    ValueError
        When an invalid download mode is provided.
    """

    if split not in VALID_SPLITS:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise InvalidSplitError(f"Unsupported split '{split}'. Valid options: {valid}.")

    resolved_dataset_path = dataset_path or os.getenv(ENV_DATASET_PATH, DATASET_NAME)
    resolved_cache_dir = cache_dir or os.getenv(ENV_CACHE_DIR)
    resolved_data_dir = data_dir or os.getenv(ENV_DATA_DIR)

    resolved_download_mode = download_mode or os.getenv(ENV_DOWNLOAD_MODE)
    if isinstance(resolved_download_mode, str):
        resolved_download_mode = _parse_download_mode(resolved_download_mode)

    resolved_local_only = _resolve_bool(local_files_only)
    if resolved_local_only is None:
        resolved_local_only = _resolve_bool(os.getenv(ENV_LOCAL_FILES_ONLY))

    return load_dataset(
        resolved_dataset_path,
        split=split,
        cache_dir=resolved_cache_dir,
        data_dir=resolved_data_dir,
        download_mode=resolved_download_mode,
        local_files_only=resolved_local_only,
    )
