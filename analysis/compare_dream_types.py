"""
compare_dream_types.py — compares nightmare / lucid / general dreams on
every features/ numeric feature, with Cohen's d and a bootstrap CI for
the difference in means, for every pair of groups.

Data source: there is no real gold-labeled Reddit data yet (0 rows in
data/gold_template.csv -- see data/README.md), so "nightmare"/"lucid"/
"general" here means the Nightmare/Lucid/everything-else genres in
backend/ml.py's synthetic corpus -- a sanity check of this analysis
code, NOT a real finding about nightmares or lucid dreams. Re-run this
against data/gold_template.csv's gold_nightmare/gold_lucid columns
once real labels exist, and replace any numbers quoted from this file.
"""
import json
import os

import paths  # noqa: F401

from dream_features import NUMERIC_FEATURE_NAMES, numeric_feature_matrix
from stats import bootstrap_diff_ci, cohens_d


def build_synthetic_comparison_groups() -> dict[str, list[str]]:
    import ml
    texts, labels = ml.get_training_corpus()
    groups: dict[str, list[str]] = {'nightmare': [], 'lucid': [], 'general': []}
    for text, label in zip(texts, labels):
        if label == 'Nightmare':
            groups['nightmare'].append(text)
        elif label == 'Lucid':
            groups['lucid'].append(text)
        else:
            groups['general'].append(text)
    return groups


def feature_values_by_group(groups: dict[str, list[str]]) -> dict[str, dict]:
    """{group_name: {feature_name: [values...]}}."""
    result = {}
    for name, texts in groups.items():
        matrix = numeric_feature_matrix(texts)
        result[name] = {
            feature: matrix[:, i].tolist() if len(texts) else []
            for i, feature in enumerate(NUMERIC_FEATURE_NAMES)
        }
    return result


def compare_dream_types(groups: dict[str, list[str]], feature_names: list[str] | None = None,
                         n_resamples: int = 2000, seed: int = 42) -> dict:
    """
    {feature_name: {"group_a_vs_group_b": {"cohens_d", "bootstrap_diff_ci"}}}
    for every pair of groups, for every requested feature.
    """
    feature_names = feature_names or NUMERIC_FEATURE_NAMES
    values_by_group = feature_values_by_group(groups)
    group_names = sorted(groups.keys())

    results: dict = {}
    for feature in feature_names:
        pairwise = {}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                a = values_by_group[g1][feature]
                b = values_by_group[g2][feature]
                pairwise[f'{g1}_vs_{g2}'] = {
                    'cohens_d': cohens_d(a, b),
                    'bootstrap_diff_ci': bootstrap_diff_ci(a, b, n_resamples=n_resamples, seed=seed),
                }
        results[feature] = pairwise
    return results


def main():
    groups = build_synthetic_comparison_groups()
    print({name: len(texts) for name, texts in groups.items()})
    results = compare_dream_types(groups)

    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(paths.REPORTS_DIR, 'compare_dream_types_synthetic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
