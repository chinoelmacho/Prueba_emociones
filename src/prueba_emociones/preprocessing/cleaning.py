"""Funciones de limpieza y tokenización básicas."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_SYMBOLS_PATTERN = re.compile(r"[^a-z0-9ñü\s]", re.IGNORECASE)
_MULTISPACE_PATTERN = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Normaliza un texto a minúsculas eliminando tildes y espacios extra."""
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    texto = _SYMBOLS_PATTERN.sub(" ", texto)
    texto = _MULTISPACE_PATTERN.sub(" ", texto).strip()
    return texto


def limpiar_texto(texto: str) -> str:
    """Limpia el texto eliminando símbolos y normalizando espacios."""
    return normalizar_texto(texto)


def tokenizar_texto(texto: str) -> List[str]:
    """Realiza una tokenización simple basada en espacios tras la limpieza."""
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return []
    return texto_limpio.split(" ")


def tokenizar_lote(textos: Iterable[str]) -> List[List[str]]:
    """Aplica la tokenización simple sobre una colección de textos."""
    return [tokenizar_texto(texto) for texto in textos]


__all__ = ["limpiar_texto", "tokenizar_texto", "tokenizar_lote", "normalizar_texto"]
