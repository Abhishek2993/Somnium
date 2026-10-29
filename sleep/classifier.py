"""
classifier.py — baseline sleep-stage classifier (W, N1, N2, N3, REM)
on band-power features (sleep/features.py), with a subject-wise
train/test split (no subject's epochs appear in both splits) and
accuracy / macro-F1 / Cohen's kappa.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def subject_wise_split(subject_ids: list, test_size: float = 0.2, seed: int = RANDOM_SEED):
    """Indices for one subject-wise train/test split (no subject in both)."""
    subject_ids = np.asarray(subject_ids)
    indices = np.arange(len(subject_ids))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(indices, groups=subject_ids))
    return train_idx, test_idx


def train_stage_classifier(X, y, seed: int = RANDOM_SEED) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def run_subject_wise_evaluation(X, y, subject_ids: list, test_size: float = 0.2, seed: int = RANDOM_SEED) -> dict:
    """Fits on the train subjects, evaluates on held-out subjects. Returns
    evaluate_predictions()'s metrics plus n_train/n_test/n_train_subjects/n_test_subjects."""
    X = np.asarray(X)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)

    train_idx, test_idx = subject_wise_split(subject_ids, test_size=test_size, seed=seed)
    clf = train_stage_classifier(X[train_idx], y[train_idx], seed=seed)
    y_pred = clf.predict(X[test_idx])

    metrics = evaluate_predictions(y[test_idx], y_pred)
    metrics['n_train'] = int(len(train_idx))
    metrics['n_test'] = int(len(test_idx))
    metrics['n_train_subjects'] = int(len(set(subject_ids[train_idx])))
    metrics['n_test_subjects'] = int(len(set(subject_ids[test_idx])))
    return metrics
