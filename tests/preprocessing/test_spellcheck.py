import pytest

from prueba_emociones.preprocessing.spellcheck import corregir_palabra


def test_corrige_con_similitud_jaro_winkler(intensificadores):
    palabra = "muuuy"
    corregida = corregir_palabra(palabra, intensificadores, umbral=0.75)
    assert corregida == "muy"


def test_no_corrige_bajo_umbral(objetivos_emocionales):
    palabra = "emocion"
    corregida = corregir_palabra(palabra, objetivos_emocionales, umbral=0.95)
    assert corregida == "emocion"


def test_normaliza_mayusculas_y_tildes(negadores):
    palabra = "NEGACIÓN"
    corregida = corregir_palabra(palabra, negadores, umbral=0.8)
    assert corregida == "negacion"
