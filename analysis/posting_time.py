"""
posting_time.py — hour-of-day histogram of when dreams were POSTED to
Reddit.

CAVEAT (read this before drawing any conclusion from this file):
`created_utc` is when the Reddit POST was submitted, not when the
dream was dreamed. A person can post hours or days after waking. Any
pattern found here is a posting-behavior pattern, not a dream-timing
pattern -- never describe this as "when people dream."
"""
import json
import os
from collections import Counter
from datetime import datetime, timezone

import paths  # noqa: F401

POSTING_TIME_CAVEAT = (
    "created_utc is when the Reddit post was submitted, not when the dream "
    "happened. This describes posting behavior, not dream timing."
)


def posting_hour_histogram(rows: list[dict]) -> dict[int, int]:
    """{hour_of_day (0-23, UTC): post_count}, all 24 hours present even if 0."""
    counts = Counter(
        datetime.fromtimestamp(row['created_utc'], tz=timezone.utc).hour for row in rows
    )
    return {hour: counts.get(hour, 0) for hour in range(24)}


def posting_hour_histogram_by_subreddit(rows: list[dict]) -> dict[str, dict[int, int]]:
    by_subreddit: dict[str, list[dict]] = {}
    for row in rows:
        by_subreddit.setdefault(row['subreddit'], []).append(row)
    return {subreddit: posting_hour_histogram(group) for subreddit, group in by_subreddit.items()}


def main():
    import db

    db.init_db()
    rows = db.get_reddit_dreams()
    if not rows:
        print(
            'No rows in reddit_dreams (0 collected so far -- see data/README.md). '
            'Nothing to analyze. Run data/collect_reddit.py first.'
        )
        return

    print(POSTING_TIME_CAVEAT)
    histogram = posting_hour_histogram(rows)
    by_subreddit = posting_hour_histogram_by_subreddit(rows)

    os.makedirs(paths.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(paths.REPORTS_DIR, 'posting_time.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'caveat': POSTING_TIME_CAVEAT, 'overall': histogram, 'by_subreddit': by_subreddit}, f, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
