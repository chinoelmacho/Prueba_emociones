import importlib

def test_package_imports():
    assert importlib.import_module("prueba_emociones") is not None
