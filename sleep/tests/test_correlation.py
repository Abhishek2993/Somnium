import pytest

from correlation import simple_correlation, sleep_log_vs_feature


def test_simple_correlation_perfect_positive():
    result = simple_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert result['r'] == pytest.approx(1.0)
    assert result['n'] == 4


def test_simple_correlation_perfect_negative():
    result = simple_correlation([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0])
    assert result['r'] == pytest.approx(-1.0)


def test_simple_correlation_too_few_points_returns_neutral():
    result = simple_correlation([1.0], [1.0])
    assert result == {'r': 0.0, 'p_value': 1.0, 'n': 1}
    assert simple_correlation([], []) == {'r': 0.0, 'p_value': 1.0, 'n': 0}


def test_simple_correlation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simple_correlation([1.0, 2.0], [1.0])


def test_sleep_log_vs_feature_skips_missing_values():
    dreams = [
        {'text': 'aaaaa', 'sleep_quality': 5},
        {'text': 'b', 'sleep_quality': None},
        {'text': 'c', 'sleep_quality': 1},
        {'text': 'd'},  # key entirely missing
    ]
    result = sleep_log_vs_feature(dreams, 'sleep_quality', feature_fn=lambda t: len(t))
    assert result['n'] == 2


def test_sleep_log_vs_feature_no_matching_rows_returns_neutral():
    dreams = [{'text': 'a'}, {'text': 'b', 'sleep_quality': None}]
    result = sleep_log_vs_feature(dreams, 'sleep_quality', feature_fn=lambda t: len(t))
    assert result == {'r': 0.0, 'p_value': 1.0, 'n': 0}
