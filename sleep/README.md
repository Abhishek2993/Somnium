# sleep/ (ENABLE_SLEEP_MODULE, default off)

EEG sleep-stage classification on the public Sleep-EDF Expanded
dataset, kept deliberately separate from the dream-text pipeline.
**Read [docs/biology_link.md](../docs/biology_link.md) first** — it
explains why this module exists and, just as importantly, what it does
not claim (the text and EEG datasets are unpaired).

| File | What it does | Status |
|---|---|---|
| [features.py](features.py) | Band power (delta/theta/alpha/sigma/beta) per 30s epoch, via Welch PSD | Real, tested on synthetic sine-wave signals with known dominant frequencies |
| [classifier.py](classifier.py) | Baseline stage classifier (W/N1/N2/N3/REM), subject-wise split, accuracy/macro-F1/Cohen's kappa | Real, tested on synthetic per-subject data (no subject ever in both train and test) |
| [hypnogram.py](hypnogram.py) | Plots a stage sequence as a Plotly hypnogram | Real, tested |
| [correlation.py](correlation.py) | Pearson correlation between a sleep-log field and a dream-text feature | Real, tested — descriptive only, no medical claims (see docs/biology_link.md) |
| [download_sleep_edf.py](download_sleep_edf.py) | Downloads real Sleep-EDF Expanded recordings via `mne` | **Not run here** — the full corpus is several GB and this is a one-time, user-initiated download; `mne` itself isn't even installed in this dev environment yet (see the pin note in `backend/requirements.txt`) |

## Why nothing here has run on real EEG

The baseline classifier (plain `LogisticRegression` on band-power
features) doesn't need a GPU — the real blocker is that downloading
and processing real Sleep-EDF recordings is a multi-GB,
multi-hour-or-more undertaking, not something to trigger automatically
in a dev session. `mne` is pinned in `backend/requirements.txt` for
when that happens, but hasn't been installed or verified here. Every function
in `features.py` and `classifier.py` takes plain numpy arrays (not an
`mne.Raw` object), specifically so they can be — and are — tested
against synthetic signals with a fixed seed, with no real EEG data or
`mne` import required for the tests to run.

To actually run this against real data:

```bash
backend/venv/bin/python3 sleep/download_sleep_edf.py --subjects 0 1 2
# then load the downloaded EDF with mne, extract each channel as a
# numpy array + its sampling rate, and pass that into features.py /
# classifier.py -- a loader script tying mne's reader to these
# functions is a natural next addition once real data is in hand.
```

## Optional sleep-log fields

When `ENABLE_SLEEP_MODULE=true` (see `backend/.env`), `backend/db.py`
adds two nullable columns to the existing `dreams` table —
`sleep_duration_hours` and `sleep_quality` (1-5) — and `backend/app.py`'s
`POST /api/dreams` accepts them. `correlation.py` computes a simple
Pearson correlation between one of these and any dream-text feature
for one user's own journal. This is small-sample, self-reported,
descriptive statistics — not a diagnostic tool.

## Running the tests

```bash
backend/venv/bin/python3 -m pytest sleep
```
