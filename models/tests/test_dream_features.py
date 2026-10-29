from dream_features import NUMERIC_FEATURE_NAMES, numeric_feature_matrix, numeric_features_for_text


def test_numeric_features_for_text_length_matches_names():
    features = numeric_features_for_text("I walked through a dark forest and saw a monster")
    assert len(features) == len(NUMERIC_FEATURE_NAMES)
    assert all(isinstance(f, float) for f in features)


def test_numeric_features_for_text_empty_string():
    features = numeric_features_for_text('')
    assert len(features) == len(NUMERIC_FEATURE_NAMES)


def test_numeric_feature_matrix_shape():
    matrix = numeric_feature_matrix(["a short dream", "another short dream here"])
    assert matrix.shape == (2, len(NUMERIC_FEATURE_NAMES))


def test_numeric_feature_matrix_empty_list():
    matrix = numeric_feature_matrix([])
    assert matrix.shape == (0, 0) or matrix.shape[0] == 0
