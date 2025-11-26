"""Funciones de corrección ortográfica simples."""
from __future__ import annotations

from typing import Iterable, Optional
import unicodedata
import re


def _jaro_distance(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    match_flags_1 = [False] * len1
    match_flags_2 = [False] * len2

    matches = 0
    transpositions = 0

    for i, ch1 in enumerate(s1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match_flags_2[j]:
                continue
            if ch1 == s2[j]:
                match_flags_1[i] = True
                match_flags_2[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    for i, flag in enumerate(match_flags_1):
        if not flag:
            continue
        while not match_flags_2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions / 2) / matches)
    ) / 3.0


def _jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
    jaro = _jaro_distance(s1, s2)
    prefix_len = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break
        prefix_len += 1
        if prefix_len == 4:
            break
    return jaro + prefix_len * prefix_scale * (1 - jaro)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text


def corregir_palabra(palabra: str, objetivos: Iterable[str], umbral: float = 0.88) -> str:
    """Devuelve la palabra de objetivos más cercana si supera el umbral.

    Se realiza una normalización sencilla (minúsculas, eliminación de tildes y
    espacios sobrantes) antes de calcular la similitud Jaro-Winkler.
    """

    palabra_norm = _normalize(palabra)
    objetivos = list(objetivos)
    if not objetivos:
        return palabra_norm

    mejor_objetivo: Optional[str] = None
    mejor_puntaje = umbral

    for objetivo in objetivos:
        objetivo_norm = _normalize(objetivo)
        puntaje = _jaro_winkler_similarity(palabra_norm, objetivo_norm)
        if puntaje > mejor_puntaje or (
            mejor_objetivo is None and puntaje >= mejor_puntaje
        ) or (
            puntaje == mejor_puntaje and mejor_objetivo is not None and len(objetivo_norm) < len(mejor_objetivo)
        ):
            mejor_puntaje = puntaje
            mejor_objetivo = objetivo_norm

    return mejor_objetivo if mejor_objetivo is not None else palabra_norm


__all__ = ["corregir_palabra"]
