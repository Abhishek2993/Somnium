import numpy as np

from classifier import (
    STAGES, evaluate_predictions, run_subject_wise_evaluation,
    subject_wise_split, train_stage_classifier,
)

STAGE_MEANS = {
    'W':   [1, 1, 1, 1, 5],
    'N1':  [1, 2, 3, 2, 2],
    'N2':  [3, 2, 2, 4, 1],
    'N3':  [5, 1, 1, 1, 1],
    'REM': [1, 1, 4, 1, 3],
}


def make_synthetic_sleep_data(n_subjects: int = 10, epochs_per_stage: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y, subjects = [], [], []
    for subj in range(n_subjects):
        for stage in STAGES:
            mean = STAGE_MEANS[stage]
            for _ in range(epochs_per_stage):
                X.append(rng.normal(loc=mean, scale=0.3))
                y.append(stage)
                subjects.append(f'subj{subj}')
    return np.array(X), np.array(y), np.array(subjects)


def test_subject_wise_split_never_puts_same_subject_in_both_sets():
    _, _, subjects = make_synthetic_sleep_data()
    train_idx, test_idx = subject_wise_split(subjects, test_size=0.3, seed=42)
    train_subjects = set(subjects[train_idx])
    test_subjects = set(subjects[test_idx])
    assert train_subjects.isdisjoint(test_subjects)


def test_subject_wise_split_is_deterministic_for_fixed_seed():
    _, _, subjects = make_synthetic_sleep_data()
    train_a, test_a = subject_wise_split(subjects, seed=7)
    train_b, test_b = subject_wise_split(subjects, seed=7)
    assert list(train_a) == list(train_b)
    assert list(test_a) == list(test_b)


def test_train_stage_classifier_predicts_known_labels():
    X, y, _ = make_synthetic_sleep_data(epochs_per_stage=10)
    clf = train_stage_classifier(X, y, seed=42)
    predictions = clf.predict(X)
    assert set(predictions) <= set(STAGES)


def test_evaluate_predictions_perfect_match():
    y_true = ['W', 'N1', 'N2', 'N3', 'REM']
    metrics = evaluate_predictions(y_true, y_true)
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_f1'] == 1.0
    assert metrics['cohen_kappa'] == 1.0


def test_run_subject_wise_evaluation_on_well_separated_synthetic_data():
    X, y, subjects = make_synthetic_sleep_data(n_subjects=10, epochs_per_stage=20, seed=42)
    metrics = run_subject_wise_evaluation(X, y, subjects, test_size=0.3, seed=42)

    assert set(metrics.keys()) == {
        'accuracy', 'macro_f1', 'cohen_kappa', 'n_train', 'n_test', 'n_train_subjects', 'n_test_subjects',
    }
    # Stage clusters are well-separated (mean distance >> noise scale), so a
    # held-out-subject classifier should do well, not just memorize subjects.
    assert metrics['accuracy'] > 0.8
    assert metrics['n_train'] + metrics['n_test'] == len(y)


def test_run_subject_wise_evaluation_is_deterministic_for_fixed_seed():
    X, y, subjects = make_synthetic_sleep_data()
    first = run_subject_wise_evaluation(X, y, subjects, seed=5)
    second = run_subject_wise_evaluation(X, y, subjects, seed=5)
    assert first == second
