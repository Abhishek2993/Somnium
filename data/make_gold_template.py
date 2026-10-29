"""
make_gold_template.py — sample 200 random collected dreams into
data/gold_template.csv for manual labeling.

All metrics reported in this project (model comparisons, per-class
reports, nightmare/lucid/general comparisons) are computed ONLY on the
hand-labeled gold set that comes out of this file -- never on the weak
subreddit/flair labels from weak_labels.py.

Usage:
    backend/venv/bin/python3 data/make_gold_template.py
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))

import db  # noqa: E402

SAMPLE_SIZE = 200
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_template.csv')

CSV_FIELDS = [
    'id', 'source', 'source_id', 'text',
    'gold_genre', 'gold_lucid', 'gold_nightmare', 'gold_notes',
]


def fetch_candidate_dreams() -> list[dict]:
    """All collected dreams, normalised to a common {source, source_id, text} shape."""
    candidates = []
    for row in db.get_reddit_dreams():
        candidates.append({
            'source': 'reddit',
            'source_id': row['post_id'],
            'text': (row['title'] + '\n\n' + row['body']).strip(),
        })
    for row in db.get_dreambank_dreams():
        candidates.append({
            'source': 'dreambank',
            'source_id': str(row['id']),
            'text': row['text'],
        })
    return candidates


def sample_gold_template(dreams: list[dict], n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> list[dict]:
    """
    Deterministically sample up to n dreams (sorted by source_id first so
    the sample is reproducible regardless of DB row order), and number
    them 1..len for use as a stable labeling id.
    """
    pool = sorted(dreams, key=lambda d: (d['source'], d['source_id']))
    rng = random.Random(seed)
    sample = rng.sample(pool, k=min(n, len(pool)))
    return [{'id': i + 1, **row} for i, row in enumerate(sample)]


def write_gold_csv(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'id': row['id'],
                'source': row['source'],
                'source_id': row['source_id'],
                'text': row['text'],
                'gold_genre': '',
                'gold_lucid': '',
                'gold_nightmare': '',
                'gold_notes': '',
            })


def main():
    db.init_db()
    candidates = fetch_candidate_dreams()
    if not candidates:
        print(
            'No collected dreams found in reddit_dreams/dreambank_dreams -- '
            'run data/collect_reddit.py (and/or data/dreambank_loader.py) first. '
            'Writing an empty gold_template.csv with headers only.'
        )
    sample = sample_gold_template(candidates)
    if 0 < len(sample) < SAMPLE_SIZE:
        print(f'Only {len(sample)} candidate dreams available; sampled all of them '
              f'(requested {SAMPLE_SIZE}).')
    write_gold_csv(sample, OUTPUT_PATH)
    print(f'Wrote {len(sample)} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
