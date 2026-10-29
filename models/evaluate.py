"""
evaluate.py — stratified 5-fold cross-validation for any model exposing
fit(texts, labels) -> model and predict(model, texts) -> labels. Saves
macro-F1, a per-class report, and a confusion matrix to reports/.

Usage:
    backend/venv/bin/python3 models/evaluate.py
"""
import csv
import json
import os

import paths  # noqa: F401

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42


def stratified_cv_report(fit_fn, predict_fn, texts: list[str], labels: list[str],
                          n_splits: int = 5, seed: int = RANDOM_SEED) -> dict:
    """
    Runs stratified K-fold CV, refitting fit_fn on each training fold and
    predicting on the held-out fold, then aggregates out-of-fold
    predictions into one macro-F1 / per-class report / confusion matrix.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for train_idx, test_idx in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        model = fit_fn(train_texts, train_labels, seed=seed)
        predictions = predict_fn(model, test_texts)

        y_true_all.extend(test_labels)
        y_pred_all.extend(predictions)

    label_order = sorted(set(labels))
    return {
        'macro_f1': f1_score(y_true_all, y_pred_all, average='macro', labels=label_order, zero_division=0),
        'per_class_report': classification_report(
            y_true_all, y_pred_all, labels=label_order, output_dict=True, zero_division=0,
        ),
        'confusion_matrix': confusion_matrix(y_true_all, y_pred_all, labels=label_order).tolist(),
        'labels': label_order,
        'n_splits': n_splits,
        'seed': seed,
    }


def save_report(name: str, report: dict, out_dir: str | None = None) -> None:
    out_dir = out_dir or paths.REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, f'{name}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'macro_f1': report['macro_f1'],
            'per_class_report': report['per_class_report'],
            'n_splits': report['n_splits'],
            'seed': report['seed'],
        }, f, indent=2)

    cm_path = os.path.join(out_dir, f'{name}_confusion_matrix.csv')
    with open(cm_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + report['labels'])
        for label, row in zip(report['labels'], report['confusion_matrix']):
            writer.writerow([label] + row)


def main():
    import ml
    import train_logreg_tfidf_features
    import train_sbert_logreg

    texts, labels = ml.get_training_corpus()
    print(f'Evaluating on {len(texts)} examples across {len(set(labels))} genres '
          f'(the synthetic corpus in backend/ml.py -- see models/README.md).')

    runs = {
        'baseline_tfidf_linearsvc': (
            lambda tr_t, tr_l, seed: ml.build_baseline_pipeline().fit(tr_t, tr_l),
            lambda model, te_t: list(model.predict(te_t)),
        ),
        'logreg_tfidf_features': (train_logreg_tfidf_features.fit, train_logreg_tfidf_features.predict),
        'sbert_logreg': (train_sbert_logreg.fit, train_sbert_logreg.predict),
    }

    for name, (fit_fn, predict_fn) in runs.items():
        print(f'Running {name} ...')
        report = stratified_cv_report(fit_fn, predict_fn, texts, labels)
        save_report(name, report)
        print(f'  {name}: macro_f1={report["macro_f1"]:.3f}')


if __name__ == '__main__':
    main()
