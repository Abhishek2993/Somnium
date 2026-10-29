# analysis/

| File | What it does | Status |
|---|---|---|
| [stats.py](stats.py) | Cohen's d + bootstrap CI (point estimate, single-group, and group-difference) | Real, tested |
| [compare_dream_types.py](compare_dream_types.py) | Nightmare vs. lucid vs. general, every `features/` numeric feature, pairwise Cohen's d + bootstrap CI | Real statistics, run on synthetic data (see caveat) |
| [emotion_arcs.py](emotion_arcs.py) | Emotion arc (start/end/slope) aggregated per group, with bootstrap CIs | Real statistics, run on synthetic data (see caveat) |
| [posting_time.py](posting_time.py) | Hour-of-day histogram of Reddit post times | Written, not yet run — 0 rows in `reddit_dreams` so far |
| [umap_embeddings.py](umap_embeddings.py) | Interactive UMAP (Plotly HTML) of sentence embeddings, colored by genre | Written; blocked on `umap-learn` finishing install in this environment |

## The nightmare/lucid/general caveat

Same situation as `models/README.md`: there's no gold-labeled Reddit
data yet (`data/gold_template.csv` has 0 rows). So "nightmare" /
"lucid" / "general" here means the **Nightmare** (n=8) / **Lucid**
(n=8) / everything-else (n=224) genres in `backend/ml.py`'s synthetic
corpus — real computations, run for real, but on hand-written
single-sentence examples, not real dream reports. Treat every number
below as a sanity check that the analysis code works, not a finding
about nightmares or lucid dreams. Re-run both scripts against
`data/gold_template.csv`'s `gold_nightmare`/`gold_lucid` columns once
real gold labels exist, and replace this table.

### Real results from this run (seed 42, 2000 bootstrap resamples)

- **Aggression word count** (`hvdc_aggression`): nightmare vs. lucid,
  Cohen's d = **-0.76** (nightmare examples use more aggression-coded
  words; bootstrap 95% CI for lucid-minus-nightmare mean difference:
  [-0.50, 0.00]).
- **Emotion-arc end valence** (`emotion_end`): lucid vs. nightmare,
  Cohen's d = **+1.20** (lucid examples end on a more positive VADER
  valence than nightmare examples; bootstrap 95% CI for the
  lucid-minus-nightmare difference: [0.09, 0.66], excludes 0).
- **Emotion arc slope is 0.0 and start == end for every group**,
  because every entry in the synthetic corpus is a single sentence —
  `emotion_arc()` can't compute a trajectory over just one sentence
  (see `features/emotion.py`). This feature only becomes meaningful on
  real, multi-sentence dream narratives once they're collected — it's
  not a bug, it's a real limitation of the synthetic sanity-check data.

Full output: `reports/compare_dream_types_synthetic.json`,
`reports/emotion_arcs_synthetic.json` (gitignored; regenerate with
`python3 analysis/compare_dream_types.py` / `analysis/emotion_arcs.py`).

## Running it

```bash
backend/venv/bin/python3 analysis/compare_dream_types.py
backend/venv/bin/python3 analysis/emotion_arcs.py
backend/venv/bin/python3 analysis/posting_time.py       # no-ops until reddit_dreams has rows
backend/venv/bin/python3 analysis/umap_embeddings.py     # needs umap-learn
backend/venv/bin/python3 -m pytest analysis -m "not slow"
```
