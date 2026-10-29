# features/

Pure, independently-testable feature extractors for a single dream
text. Nothing here talks to the database or the Flask app directly —
`backend/app.py` (Phase 7) imports from this package to build the
`/api/analyze` feature profile.

| Module | What it computes | Status |
|---|---|---|
| [emotion.py](emotion.py) | Per-sentence VADER sentiment valence + emotion arc (start/end/slope) | Real, tested |
| [sensory.py](sensory.py) | Sensory-word ratios (visual/auditory/tactile/olfactory/gustatory) | Real, tested; lexicon is hand-curated, not from a published word list (TODO) |
| [hvdc.py](hvdc.py) | Simplified Hall-Van de Castle-style counts (characters, aggression/friendliness, misfortune/good fortune, settings) | Real, tested; **approximation only** — not the validated human-coded HVdC system |
| [narrative.py](narrative.py) | Length, dominant tense, first-person ratio | Real, tested |
| [embeddings.py](embeddings.py) | Shared `all-MiniLM-L6-v2` sentence-embedding loader | Real, tested (shared with `models/train_sbert_logreg.py` in Phase 4) |
| [bizarreness.py](bizarreness.py) | Cosine distance between consecutive sentence embeddings | Real, tested |
| [topics.py](topics.py) | BERTopic theme topics across a corpus | Real, tested on tiny synthetic corpora only — not yet run on real dream text (0 collected so far) |
| [lucidity.py](lucidity.py) | Lucid-dream probability | **Keyword heuristic, not a trained classifier** — no gold `gold_lucid` labels exist yet. `train_lucid_classifier()` is the hook to replace this once they do. |
| [nightmare.py](nightmare.py) | Nightmare vs. pleasant classification | **Heuristic from emotion_arc + hvdc counts, not a trained classifier** — same reason as lucidity.py. `train_nightmare_classifier()` is the hook. Thresholds (0.4/0.6) are arbitrary, not calibrated against gold data. |

## Why two modules are heuristics, not classifiers

`lucidity.py` and `nightmare.py` need labeled examples to train a real
classifier on, and the only labels this project treats as ground truth
are the hand-labeled `gold_lucid`/`gold_nightmare` columns in
`data/gold_template.csv` (see `data/README.md`) — which has 0 rows
until a real Reddit collection + manual labeling pass happens. Until
then these two modules fall back to transparent, documented heuristics
rather than a classifier silently "trained" on nothing. Both files
expose a `train_*_classifier(texts, labels)` function ready to be
called the moment gold labels exist.

## Running the tests

```bash
backend/venv/bin/python3 -m pytest features                 # everything, including slow/model-backed tests
backend/venv/bin/python3 -m pytest features -m "not slow"    # skip the sentence-transformers/BERTopic-backed tests
```

`bizarreness.py` and `topics.py` need `sentence-transformers` (+ CPU
torch) and `bertopic` (+ `umap-learn`) respectively — see
`backend/requirements.txt`.
