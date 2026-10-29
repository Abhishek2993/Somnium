from compare_dream_types import build_synthetic_comparison_groups
from emotion_arcs import aggregate_emotion_arcs


def test_aggregate_emotion_arcs_shape():
    groups = build_synthetic_comparison_groups()
    results = aggregate_emotion_arcs(groups, n_resamples=100)

    assert set(results.keys()) == {'nightmare', 'lucid', 'general'}
    for group_result in results.values():
        assert set(group_result.keys()) == {'start', 'end', 'slope', 'n'}
        for metric in ('start', 'end', 'slope'):
            assert set(group_result[metric].keys()) == {'point_estimate', 'low', 'high', 'confidence', 'n'}


def test_aggregate_emotion_arcs_counts_match_group_sizes():
    groups = {'a': ["I was happy. It was wonderful."], 'b': ["I was scared. It was terrible.", "Another scary one."]}
    results = aggregate_emotion_arcs(groups, n_resamples=50)
    assert results['a']['n'] == 1
    assert results['b']['n'] == 2


def test_aggregate_emotion_arcs_nightmare_end_valence_is_lower_than_lucid():
    groups = build_synthetic_comparison_groups()
    results = aggregate_emotion_arcs(groups, n_resamples=500)
    assert results['nightmare']['end']['point_estimate'] < results['lucid']['end']['point_estimate']
