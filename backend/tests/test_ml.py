import math

from ml import GENRES, predict_genre, get_model


def test_genres_list_is_nonempty_and_sorted():
    assert len(GENRES) > 0
    assert GENRES == sorted(GENRES)


def test_get_model_returns_same_cached_instance():
    assert get_model() is get_model()


def test_predict_genre_shape():
    result = predict_genre("I flew over an enchanted castle with a dragon and a wizard")
    assert set(result.keys()) == {'genre', 'confidence', 'all_scores'}
    assert result['genre'] in GENRES
    assert 0.0 <= result['confidence'] <= 1.0
    assert set(result['all_scores'].keys()) == set(GENRES)


def test_predict_genre_scores_sum_to_one():
    result = predict_genre("A shadowy monster chased me through a dark abandoned house")
    assert math.isclose(sum(result['all_scores'].values()), 1.0, abs_tol=0.02)


def test_predict_genre_confidence_matches_best_score():
    result = predict_genre("A spy mission went wrong and I was being hunted by assassins")
    assert result['all_scores'][result['genre']] == result['confidence']
