import sys
import types
from enum import Enum
from unittest.mock import Mock

import pytest

# Provide a lightweight stub for the `datasets` package when it is not installed.
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    class DummyDownloadMode(Enum):
        REUSE_CACHE_IF_EXISTS = "reuse_cache_if_exists"
        REUSE_DATASET_IF_EXISTS = "reuse_dataset_if_exists"
        FORCE_REDOWNLOAD = "force_redownload"

    datasets_stub.DownloadMode = DummyDownloadMode
    datasets_stub.Dataset = object
    datasets_stub.load_dataset = Mock(name="load_dataset_stub")
    sys.modules["datasets"] = datasets_stub

from datasets import DownloadMode

from prueba_emociones.data.loading import (
    ENV_CACHE_DIR,
    ENV_DATASET_PATH,
    ENV_DATA_DIR,
    ENV_DOWNLOAD_MODE,
    ENV_LOCAL_FILES_ONLY,
    DATASET_NAME,
    InvalidSplitError,
    load_tweet_emotion_intensity,
)


def test_load_default_split(monkeypatch):
    fake_dataset = Mock(name="dataset")
    mocked_loader = Mock(return_value=fake_dataset)
    monkeypatch.setenv(ENV_CACHE_DIR, "/tmp/cache")
    monkeypatch.setattr("prueba_emociones.data.loading.load_dataset", mocked_loader)

    result = load_tweet_emotion_intensity()

    assert result is fake_dataset
    mocked_loader.assert_called_once_with(
        DATASET_NAME,
        split="train",
        cache_dir="/tmp/cache",
        data_dir=None,
        download_mode=None,
        local_files_only=None,
    )


def test_invalid_split_raises():
    with pytest.raises(InvalidSplitError):
        load_tweet_emotion_intensity(split="train-dev")


def test_env_overrides_and_download_mode(monkeypatch):
    fake_dataset = Mock(name="dataset")
    mocked_loader = Mock(return_value=fake_dataset)
    monkeypatch.setattr("prueba_emociones.data.loading.load_dataset", mocked_loader)

    monkeypatch.setenv(ENV_DATASET_PATH, "./local_dataset")
    monkeypatch.setenv(ENV_DATA_DIR, "/data")
    monkeypatch.setenv(ENV_CACHE_DIR, "/cache")
    monkeypatch.setenv(ENV_DOWNLOAD_MODE, "force_redownload")
    monkeypatch.setenv(ENV_LOCAL_FILES_ONLY, "true")

    result = load_tweet_emotion_intensity(split="validation")

    assert result is fake_dataset
    mocked_loader.assert_called_once_with(
        "./local_dataset",
        split="validation",
        cache_dir="/cache",
        data_dir="/data",
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        local_files_only=True,
    )


def test_download_mode_string_argument(monkeypatch):
    fake_dataset = Mock(name="dataset")
    mocked_loader = Mock(return_value=fake_dataset)
    monkeypatch.setattr("prueba_emociones.data.loading.load_dataset", mocked_loader)

    result = load_tweet_emotion_intensity(
        split="test", download_mode="reuse_cache_if_exists", local_files_only=False
    )

    assert result is fake_dataset
    mocked_loader.assert_called_once_with(
        DATASET_NAME,
        split="test",
        cache_dir=None,
        data_dir=None,
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        local_files_only=False,
    )


def test_invalid_download_mode_string(monkeypatch):
    with pytest.raises(ValueError):
        load_tweet_emotion_intensity(download_mode="not-a-mode")
