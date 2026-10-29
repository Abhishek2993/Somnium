from compare_dream_types import build_synthetic_comparison_groups, compare_dream_types, feature_values_by_group
from dream_features import NUMERIC_FEATURE_NAMES


def test_build_synthetic_comparison_groups_has_three_groups_and_real_text():
    groups = build_synthetic_comparison_groups()
    assert set(groups.keys()) == {'nightmare', 'lucid', 'general'}
    assert len(groups['nightmare']) == 8
    assert len(groups['lucid']) == 8
    assert len(groups['general']) == 30 * 8 - 16
    assert all(isinstance(t, str) and t for t in groups['nightmare'])


def test_feature_values_by_group_shape():
    groups = {'a': ["one two three four", "five six seven"], 'b': ["eight nine"]}
    values = feature_values_by_group(groups)
    assert set(values.keys()) == {'a', 'b'}
    assert set(values['a'].keys()) == set(NUMERIC_FEATURE_NAMES)
    assert len(values['a']['narrative_length']) == 2
    assert len(values['b']['narrative_length']) == 1


def test_compare_dream_types_covers_every_pair_and_feature():
    groups = build_synthetic_comparison_groups()
    results = compare_dream_types(groups, n_resamples=100)

    assert set(results.keys()) == set(NUMERIC_FEATURE_NAMES)
    one_feature = results['hvdc_aggression']
    assert set(one_feature.keys()) == {'general_vs_lucid', 'general_vs_nightmare', 'lucid_vs_nightmare'}
    for pair_result in one_feature.values():
        assert 'cohens_d' in pair_result
        assert 'bootstrap_diff_ci' in pair_result


def test_compare_dream_types_nightmares_score_higher_on_aggression_than_lucid():
    groups = build_synthetic_comparison_groups()
    results = compare_dream_types(groups, feature_names=['hvdc_aggression'], n_resamples=500)
    d = results['hvdc_aggression']['lucid_vs_nightmare']['cohens_d']
    assert d < 0  # lucid - nightmare < 0, i.e. nightmare has more aggression words
