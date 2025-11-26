import pytest


@pytest.fixture
def intensificadores():
    return ["muy", "sumamente", "extremadamente"]


@pytest.fixture
def atenuadores():
    return ["poco", "apenas", "levemente"]


@pytest.fixture
def negadores():
    return ["no", "nunca", "negacion"]


@pytest.fixture
def objetivos_emocionales(intensificadores, atenuadores, negadores):
    return intensificadores + atenuadores + negadores
