"""Carga y preparación de datos para los ejemplos de emociones."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd


class DatasetMissingDependency(ImportError):
    """Error lanzado cuando falta una dependencia opcional."""


def _ensure_datasets_installed():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependerá del entorno de usuario
        raise DatasetMissingDependency(
            "El paquete 'datasets' es necesario para descargar conjuntos de datos "
            "de Hugging Face. Instálalo con `pip install datasets`."
        ) from exc


def load_hf_dataset(
    name: str = "stepp1/tweet_emotion_intensity",
    *,
    split: str = "train",
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Descarga un *Dataset* de Hugging Face y devuelve un ``DataFrame``.

    Parameters
    ----------
    name:
        Nombre del dataset (por defecto ``"stepp1/tweet_emotion_intensity"``).
    split:
        División a cargar (``"train"``, ``"test"``...).
    text_column / label_column:
        Nombres de las columnas de texto y etiqueta. Si se dejan en ``None`` se
        intentan inferir buscando opciones comunes como ``text``, ``tweet`` o
        ``content`` para el texto y ``intensity`` o ``label`` para la etiqueta.
    limit:
        Si se indica, se recorta el número de filas para ejecuciones rápidas.

    Returns
    -------
    tuple
        ``(df, text_column, label_column)`` con los nombres finalmente usados.
    """

    _ensure_datasets_installed()
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(name, split=split)
    df = dataset.to_pandas()

    if limit is not None:
        df = df.iloc[:limit]

    text_column = text_column or _infer_column(df.columns, ["text", "tweet", "content", "sentence"])
    label_column = label_column or _infer_column(df.columns, ["intensity", "label", "target", "emotion"])

    if text_column is None or label_column is None:
        raise ValueError(
            "No se pudieron inferir las columnas de texto y etiqueta. "
            "Pásalas explícitamente mediante text_column/label_column."
        )

    return df[[text_column, label_column]].copy(), text_column, label_column


def _infer_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    columns_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None
