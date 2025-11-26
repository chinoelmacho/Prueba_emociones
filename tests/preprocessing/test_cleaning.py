import pytest

from prueba_emociones.preprocessing.cleaning import limpiar_texto, tokenizar_texto, tokenizar_lote


def test_limpia_simbolos_y_normaliza(intensificadores):
    texto = "¡¡¡MUY!!!, contento;;;"  # contiene símbolos y mayúsculas
    limpio = limpiar_texto(texto)
    assert limpio == "muy contento"


def test_tokeniza_texto_simple(negadores):
    texto = "No   quiero   hacerlo"
    tokens = tokenizar_texto(texto)
    assert tokens == ["no", "quiero", "hacerlo"]


def test_tokeniza_lote(atenuadores):
    textos = ["Apenas hay ruido", "Levemente llovizna"]
    tokens = tokenizar_lote(textos)
    assert tokens == [["apenas", "hay", "ruido"], ["levemente", "llovizna"]]
