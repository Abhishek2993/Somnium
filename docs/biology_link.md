# The biology link: what this module is, and what it is not

`sleep/` (gated behind `ENABLE_SLEEP_MODULE`, default **off**) adds EEG
sleep-stage classification on the public Sleep-EDF Expanded dataset,
alongside — never combined with — Somnium's dream-text pipeline.

## The literature relation this module is motivated by

A consistent finding across decades of sleep research (going back to
Aserinsky & Kleitman's original REM discovery, and refined since by
researchers like Nielsen, Hobson, and others) is that dream reports
collected after waking from REM sleep tend to be:

- **Longer** — more words, more narrative content, than reports from NREM awakenings.
- **More vivid** — richer sensory detail.
- **More bizarre** — more discontinuous, implausible, or physically impossible content.

This is why `features/bizarreness.py` (sentence-embedding discontinuity)
and `features/sensory.py` (sensory-word ratios) exist in this project
at all: they're text-side proxies for exactly the properties REM-sleep
dream research says should vary with sleep stage.

## What this module explicitly does NOT do

**The text dataset (Reddit/DreamBank dream reports) and the EEG dataset
(Sleep-EDF Expanded) are completely unpaired.** No Sleep-EDF subject
ever wrote a Reddit post, and no Reddit poster's EEG is in Sleep-EDF.
There is no shared subject, no shared night, no way to join one dataset
to the other. `sleep/classifier.py` predicts sleep stage from EEG band
power; `features/*.py` predicts text properties from dream text. These
are two independent pipelines run on two independent populations.

**Nothing in this codebase decodes a dream's content from EEG, or
predicts EEG patterns from dream text.** If you see those two pipelines
discussed together, the connection is the *literature finding above* —
a published, general correlation between sleep stage and reported dream
properties across many independent studies — not a claim this specific
code can detect in a specific user's data. Reusing this project's own
features/sensory scores as a stand-in for "how REM-like" a dream was is
a hypothesis for someone to go test with real paired data; it is not
something this repository has verified to be true.

## Optional sleep-log fields (`sleep_duration_hours`, `sleep_quality`)

When `ENABLE_SLEEP_MODULE` is on, a dream entry can optionally record
how long the user slept and a 1-5 self-reported quality rating (see
`backend/db.py`'s `_migrate_sleep_log_columns`). `sleep/correlation.py`
computes a plain Pearson correlation between one of these fields and a
dream-text feature, for one user's own journal — this is descriptive
statistics on a single person's small, self-reported sample, not a
clinical or diagnostic measurement, and it is not statistically
powered to support any claim of the form "X affects sleep" or
"Y affects dreaming." **No medical or diagnostic claims should ever be
attached to this correlation, in this codebase or in anything built on
top of it.**

## Status

Not run end-to-end in this environment: the full Sleep-EDF Expanded
corpus is several GB and this is a one-time, user-initiated download,
so `sleep/download_sleep_edf.py` was written but not executed here
(see `sleep/README.md`). `sleep/features.py` and `sleep/classifier.py`
are tested against synthetic EEG-like signals with fixed seeds, not
real recordings -- note the baseline classifier itself (logistic
regression on band-power features) doesn't need a GPU either way.
