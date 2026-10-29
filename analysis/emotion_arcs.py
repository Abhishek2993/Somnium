"""
emotion_arcs.py — aggregates emotion_arc() (start/end/slope) per dream
group, with a bootstrap CI for each. Same synthetic-corpus caveat as
compare_dream_types.py -- see its module docstring.
"""
import json
import os

import paths  # noqa: F401

from emotion import emotion_arc
from stats import bootstrap_ci


def aggregate_emotion_arcs(groups: dict[str, list[str]], n_resamples: int = 2000, seed: int = 42) -> dict:
    """{group_name: {'start': ci, 'end': ci, 'slope': ci, 'n': n}}."""
    results = {}
    for name, texts in groups.items():
        arcs = [emotion_arc(t) for t in texts]
        starts = [a['start'] for a in arcs]
        ends = [a['end'] for a in arcs]
        slopes = [a['slope'] for a in arcs]
        results[name] = {
            'start': bootstrap_ci(starts, n_resamples=n_resamples, seed=seed),
            'end': bootstrap_ci(ends, n_resamples=n_resamples, seed=seed),
            'slope': bootstrap_ci(slopes, n_resamples=n_resamples, seed=seed),
            'n': len(texts),
        }
    return results


def main():
    from compare_dream_types import build_synthetic_comparison_groups

    groups = build_synthetic_comparison_groups()
    results = aggregate_emotion_arcs(groups)

    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(paths.REPORTS_DIR, 'emotion_arcs_synthetic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out_path}')
    for name, metrics in results.items():
        print(f"{name} (n={metrics['n']}): "
              f"start={metrics['start']['point_estimate']:.3f}, "
              f"end={metrics['end']['point_estimate']:.3f}, "
              f"slope={metrics['slope']['point_estimate']:.3f}")


if __name__ == '__main__':
    main()
