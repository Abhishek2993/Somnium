# Data collection

## Sources

| Source | Script | Status |
|---|---|---|
| r/dreams (main) | [collect_reddit.py](collect_reddit.py) | not yet run — needs Reddit API credentials |
| r/LucidDreaming | [collect_reddit.py](collect_reddit.py) | not yet run — needs Reddit API credentials |
| r/Nightmares | [collect_reddit.py](collect_reddit.py) | not yet run — needs Reddit API credentials |
| Local DreamBank export (optional) | [dreambank_loader.py](dreambank_loader.py) | not yet run — needs a DreamBank file path, and the parser has not been verified against a real export (see the TODO in that file) |

Reddit posts are stored in the `reddit_dreams` table in `somnium.db`, keyed by
`post_id`; DreamBank entries go in `dreambank_dreams`, keyed by a content
hash. Both tables are defined in [`backend/db.py`](../backend/db.py).

**Dates collected:** none yet — this environment has network access but no
Reddit API credentials. To run a real collection:

```bash
cp backend/.env.example backend/.env
# fill in REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT
# (create an app at https://www.reddit.com/prefs/apps, type "script")
backend/venv/bin/python3 data/collect_reddit.py
```

**Counts collected:** none yet (0 rows in `reddit_dreams`, 0 in
`dreambank_dreams`). This file will be updated with real row counts per
subreddit, and a before/after-filter breakdown, once a collection run has
actually happened — no placeholder numbers are reported here.

## Filtering applied during collection

A post is dropped (never inserted) if any of the following hold:

- The author account is gone (`submission.author is None`) — a strong
  signal the post was deleted or removed.
- The cleaned body is empty or is a `[deleted]`/`[removed]` marker.
- The cleaned body is under 50 words.
- `langdetect` does not classify the cleaned body as English.
- The post's `post_id` or content hash (title+body, case/whitespace
  normalized) already exists in `reddit_dreams` — this is how
  cross-subreddit duplicates/crossposts are caught, in addition to the
  natural per-subreddit `post_id` uniqueness.

See [`text_cleaning.py`](text_cleaning.py) for the filter implementations
and their unit tests in [`tests/test_text_cleaning.py`](tests/test_text_cleaning.py).

## Resumability

`collect_reddit.py` appends every post id it looks at (whether it passed
the filters or not) to `data/post_ids/<subreddit>.txt` as it goes. On the
next run, ids already in that file are skipped without being
re-processed. The database's `UNIQUE` constraints on `post_id` and
`content_hash` are a second, independent safety net against duplicate
rows even if the checkpoint files are lost.

Only the collection scripts and these post-id lists (bare Reddit post
IDs, no text) are committed to git. Raw collected text lives only in the
local, gitignored `somnium.db` — see the root `.gitignore`
(`data/raw/`, `data/*.db`).

## Known limitations

- **Reddit listing cap.** `subreddit.new()` / `subreddit.top()` only
  expose roughly the most recent/highest-scoring ~1000 posts per listing
  via PRAW — there is no way to fetch a subreddit's *entire* history
  through the public API. The collected sample is therefore biased toward
  recent and highly-upvoted posts, not a random sample of all dreams ever
  posted.
- **Selection bias.** People who post dreams to Reddit, and specifically
  to r/dreams, r/LucidDreaming or r/Nightmares, are not representative of
  dreamers in general: they are Reddit users, skew toward people
  already interested in lucid dreaming or who had a dream vivid/disturbing
  enough to want to share it, and are self-selecting for willingness to
  write a long public post. Nightmare and lucid-dream content is
  systematically over-represented relative to "ordinary" dreams, by
  construction of which subreddits are being sampled.
- **Flair/subreddit as label is weak, not ground truth.** A post's
  subreddit or flair (e.g. "Nightmare") reflects how the *poster*
  categorized it, not an independent judgment — see `weak_labels.py`
  (Phase 2, not yet built) and the gold-label set sampled from this data
  for the only labels used in reported metrics.
- **Posting time ≠ dream time.** `created_utc` is when the Reddit post was
  submitted, which can be hours or days after the dream itself. Any
  time-of-day analysis (Phase 5) must state this explicitly rather than
  treating posting time as dream time.
- **Non-English filter is approximate.** `langdetect` is a statistical
  guesser and will misclassify some short or code-mixed posts in either
  direction.

## Ethics and Reddit API terms

- **No usernames are ever stored.** `collect_reddit.py` reads
  `submission.author` only to check whether the account still exists
  (a signal the post itself is still live); the username value is never
  written into `build_record()`'s output or the database.
- Only publicly-posted, non-deleted/non-removed submissions are stored,
  and only the post id, subreddit, flair, title, cleaned body, score,
  comment count, and word count — no comments, no other user activity,
  no cross-referencing against a user's other posts.
- Collection must be run with credentials from a Reddit "script" app
  registered under the researcher's own account, and must comply with
  Reddit's [Data API Terms](https://www.redditinc.com/policies/data-api-terms)
  and [User Agreement](https://www.redditinc.com/policies/user-agreement) —
  in particular around rate limits (handled automatically by
  `praw`/`prawcore`), non-commercial research use, and a user's right to
  delete their content (a post deleted on Reddit after collection will
  not be automatically removed from `reddit_dreams`; this dataset is a
  point-in-time snapshot, not a live mirror).
- This dataset is for research/educational analysis of dream *text*
  (genre, emotion, bizarreness, etc.) — it is not used to identify,
  profile, or re-contact any individual.
