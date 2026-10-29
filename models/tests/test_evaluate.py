import json
import os

from evaluate import save_report, stratified_cv_report

TEXTS = [f"dream number {i}" for i in range(20)]
LABELS = (['A'] * 10) + (['B'] * 10)


def _majority_fit(texts, labels, seed=42):
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]


def _majority_predict(model, texts):
    return [model] * len(texts)


def test_stratified_cv_report_shape_and_keys():
    report = stratified_cv_report(_majority_fit, _majority_predict, TEXTS, LABELS, n_splits=5, seed=42)
    assert set(report.keys()) == {'macro_f1', 'per_class_report', 'confusion_matrix', 'labels', 'n_splits', 'seed'}
    assert report['labels'] == ['A', 'B']
    assert report['n_splits'] == 5
    assert len(report['confusion_matrix']) == 2
    assert len(report['confusion_matrix'][0]) == 2


def test_stratified_cv_report_perfect_classifier_scores_one():
    # Texts carry a real, content-based signal ("apple" vs "banana") so a
    # non-cheating classifier can generalize to a held-out fold -- unlike
    # memorizing exact training texts, which can't work under CV.
    texts = [f"apple dream {i}" for i in range(10)] + [f"banana dream {i}" for i in range(10)]
    labels = ['A'] * 10 + ['B'] * 10

    def perfect_fit(texts, labels, seed=42):
        return None

    def perfect_predict(model, texts):
        return ['A' if 'apple' in t else 'B' for t in texts]

    report = stratified_cv_report(perfect_fit, perfect_predict, texts, labels, n_splits=5, seed=42)
    assert report['macro_f1'] == 1.0


def test_stratified_cv_report_is_deterministic_for_fixed_seed():
    first = stratified_cv_report(_majority_fit, _majority_predict, TEXTS, LABELS, n_splits=5, seed=7)
    second = stratified_cv_report(_majority_fit, _majority_predict, TEXTS, LABELS, n_splits=5, seed=7)
    assert first['confusion_matrix'] == second['confusion_matrix']


def test_save_report_writes_metrics_json_and_confusion_matrix_csv(tmp_path):
    report = stratified_cv_report(_majority_fit, _majority_predict, TEXTS, LABELS, n_splits=5, seed=42)
    save_report('unit_test_model', report, out_dir=str(tmp_path))

    metrics_path = tmp_path / 'unit_test_model_metrics.json'
    cm_path = tmp_path / 'unit_test_model_confusion_matrix.csv'
    assert metrics_path.exists()
    assert cm_path.exists()

    with open(metrics_path) as f:
        metrics = json.load(f)
    assert 'macro_f1' in metrics
    assert 'per_class_report' in metrics

    with open(cm_path) as f:
        lines = f.read().strip().splitlines()
    assert lines[0] == ',A,B'
