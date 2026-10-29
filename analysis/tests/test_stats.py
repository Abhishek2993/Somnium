import numpy as np

from stats import bootstrap_ci, bootstrap_diff_ci, cohens_d


def test_cohens_d_zero_for_identical_groups():
    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_cohens_d_positive_when_a_greater_than_b():
    d = cohens_d([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
    assert d > 0


def test_cohens_d_negative_when_a_less_than_b():
    d = cohens_d([1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
    assert d < 0


def test_cohens_d_zero_for_too_few_samples():
    assert cohens_d([1.0], [1.0, 2.0, 3.0]) == 0.0
    assert cohens_d([], []) == 0.0


def test_cohens_d_zero_for_zero_variance_groups():
    assert cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 0.0


def test_bootstrap_ci_empty_data():
    result = bootstrap_ci([])
    assert result['n'] == 0
    assert result['point_estimate'] == 0.0


def test_bootstrap_ci_contains_point_estimate_for_constant_data():
    result = bootstrap_ci([3.0, 3.0, 3.0, 3.0], n_resamples=200)
    assert result['point_estimate'] == 3.0
    assert result['low'] == 3.0
    assert result['high'] == 3.0


def test_bootstrap_ci_mean_is_plausible():
    data = [float(x) for x in range(1, 21)]  # 1..20, mean=10.5
    result = bootstrap_ci(data, n_resamples=2000, seed=1)
    assert result['point_estimate'] == 10.5
    assert result['low'] < 10.5 < result['high']


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    data = [1.0, 5.0, 3.0, 9.0, 2.0]
    first = bootstrap_ci(data, n_resamples=500, seed=7)
    second = bootstrap_ci(data, n_resamples=500, seed=7)
    assert first == second


def test_bootstrap_diff_ci_empty_group_returns_zero():
    result = bootstrap_diff_ci([], [1.0, 2.0])
    assert result['point_estimate'] == 0.0


def test_bootstrap_diff_ci_captures_true_difference():
    a = [float(x) for x in range(10, 21)]   # mean 15
    b = [float(x) for x in range(0, 11)]    # mean 5
    result = bootstrap_diff_ci(a, b, n_resamples=2000, seed=1)
    assert result['point_estimate'] == 10.0
    assert result['low'] < 10.0 < result['high']


def test_bootstrap_diff_ci_supports_custom_statistic():
    a = [1.0, 2.0, 3.0, 100.0]
    b = [1.0, 2.0, 3.0, 4.0]
    result = bootstrap_diff_ci(a, b, statistic=np.median, n_resamples=500, seed=1)
    assert result['point_estimate'] == 0.0
