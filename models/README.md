# models/

Compares the existing TF-IDF+LinearSVC genre baseline against two new
model types on **equal footing**, via stratified 5-fold cross-validation.

## What data this actually evaluates on

There is still no gold-labeled Reddit dream data (`data/gold_template.csv`
has 0 rows — see `data/README.md`). The only labeled genre dataset that
exists anywhere in this project right now is the **synthetic corpus
hardcoded in `backend/ml.py`** (30 genres x ~8 hand-written sentences
each = ~240 examples) — the same corpus the existing baseline classifier
already trains on. `models/evaluate.py` runs all three model types on
that corpus via `ml.get_training_corpus()`, so the comparison is real
(an actual 5-fold CV run, not invented numbers) but **not yet a
comparison on real dream text**. Re-run `evaluate.py` once real,
gold-labeled genre data exists and treat today's numbers as a sanity
check that the pipeline works, not as a genre-classification benchmark.

## Models

| Model | File | Status |
|---|---|---|
| (baseline) TF-IDF + LinearSVC | `backend/ml.py` (existing, unchanged) | Real |
| (a) TF-IDF + `features/` numeric features + LogisticRegression | `train_logreg_tfidf_features.py` | Real, tested |
| (b) `all-MiniLM-L6-v2` sentence embeddings + LogisticRegression | `train_sbert_logreg.py` | Real, tested |
| (c) Fine-tuned DistilBERT (optional, `ENABLE_DISTILBERT`) | `train_distilbert.py` | **Written but not run** — no GPU in this dev environment, and CPU fine-tuning is slow even on ~240 examples. The disabled-by-default path (raising `RuntimeError`) is tested; the actual fine-tuning path is not. |

## Results (real run, synthetic corpus, stratified 5-fold CV, seed 42)

| Model | Macro-F1 |
|---|---|
| Baseline: TF-IDF + LinearSVC | 0.159 |
| (a) TF-IDF + features + LogisticRegression | 0.132 |
| (b) all-MiniLM-L6-v2 + LogisticRegression | 0.598 |

All three scores are low in absolute terms because there are only ~8
examples per genre (240 examples / 30 classes) — 5-fold CV means each
fold trains on roughly 6 examples per class and tests on ~2. The
relative gap is still a real, useful signal from this run: pretrained
sentence embeddings (b) generalize far better than either TF-IDF
variant at this sample size, which TF-IDF's sparse per-class vocabulary
can't learn well from so few examples. These numbers will shift once
real, gold-labeled genre data exists (see "What data this actually
evaluates on" above) — re-run `evaluate.py` then and replace this table.

## Running it

```bash
backend/venv/bin/python3 models/evaluate.py          # trains+CVs all 3 non-DistilBERT models, writes reports/
backend/venv/bin/python3 -m pytest models             # unit tests (add -m "not slow" to skip the SBERT-backed ones)
```

`evaluate.py` writes `reports/<model_name>_metrics.json` (macro-F1 +
per-class precision/recall/F1) and `reports/<model_name>_confusion_matrix.csv`
for each model.
