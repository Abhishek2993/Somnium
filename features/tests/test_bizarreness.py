import numpy as np
import pytest

from bizarreness import bizarreness_score, cosine_distance


def test_cosine_distance_identical_vectors_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_vectors_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_opposite_vectors_is_two():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-9)


def test_cosine_distance_zero_vector_returns_one():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_distance(a, b) == 1.0


def test_bizarreness_score_empty_text():
    result = bizarreness_score('')
    assert result == {'per_pair': [], 'mean': 0.0, 'max': 0.0, 'num_sentences': 0}


def test_bizarreness_score_single_sentence():
    result = bizarreness_score("I was walking through the forest.")
    assert result['num_sentences'] == 1
    assert result['per_pair'] == []
    assert result['mean'] == 0.0


@pytest.mark.slow
def test_bizarreness_score_coherent_vs_disjointed_text():
    coherent = (
        "I was walking through a quiet forest. The trees were tall and green. "
        "Birds were singing in the branches above me."
    )
    disjointed = (
        "I was walking through a quiet forest. Suddenly a calculator exploded into jazz music. "
        "My grandmother's teeth turned into a fleet of tiny submarines."
    )
    coherent_score = bizarreness_score(coherent)
    disjointed_score = bizarreness_score(disjointed)
    assert disjointed_score['mean'] > coherent_score['mean']
