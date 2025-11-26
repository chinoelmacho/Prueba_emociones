import numpy as np
from scipy import sparse

from prueba_emociones.features.feature_pipeline import FeaturePipeline, build_feature_pipeline
from prueba_emociones.features.intensity_flags import IntensityFlagTransformer


def test_intensity_transformer_counts():
    transformer = IntensityFlagTransformer()
    texts = [
        "Muy feliz y super emocionado, nada malo",
        "No estoy triste, pero tampoco estoy muy alegre",
        "Es increíblemente bueno y absolutamente genial",
    ]

    result = transformer.transform(texts)

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[:, 0], np.array([2.0, 1.0, 2.0]))
    np.testing.assert_array_equal(result[:, 1], np.array([1.0, 2.0, 0.0]))
    assert list(transformer.get_feature_names_out()) == ["intensifier_count", "negation_count"]


def test_pipeline_shapes_and_columns():
    texts = [
        "Me siento muy feliz hoy",
        "No estoy seguro, pero extremadamente interesado",
        "La película no fue nada buena",
    ]

    pipeline = FeaturePipeline()
    matrix = pipeline.fit_transform(texts)

    assert sparse.issparse(matrix)
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] == pipeline.vectorizer.transform(texts).shape[1] + 2

    feature_names = pipeline.get_feature_names_out()
    assert "intensifier_count" in feature_names
    assert "negation_count" in feature_names
    assert any("feliz" in name for name in feature_names)


def test_pipeline_is_deterministic_with_seed():
    texts = ["muy feliz", "no tan malo"]

    pipeline_a = build_feature_pipeline()
    pipeline_b = build_feature_pipeline()

    rng_seed = 123
    np.random.seed(rng_seed)
    matrix_a = pipeline_a.fit_transform(texts)

    np.random.seed(rng_seed)
    matrix_b = pipeline_b.fit_transform(texts)

    assert matrix_a.shape == matrix_b.shape
    assert matrix_a.nnz == matrix_b.nnz
    np.testing.assert_array_equal(matrix_a.toarray(), matrix_b.toarray())
