import csv

import make_gold_template as mgt


def _dreams(n):
    return [{'source': 'reddit', 'source_id': f'p{i:03d}', 'text': f'dream number {i}'} for i in range(n)]


def test_sample_gold_template_caps_at_requested_size():
    sample = mgt.sample_gold_template(_dreams(10), n=5, seed=1)
    assert len(sample) == 5
    assert [row['id'] for row in sample] == [1, 2, 3, 4, 5]


def test_sample_gold_template_returns_all_when_fewer_than_n():
    sample = mgt.sample_gold_template(_dreams(3), n=200, seed=1)
    assert len(sample) == 3


def test_sample_gold_template_is_deterministic_for_fixed_seed():
    dreams = _dreams(20)
    first = mgt.sample_gold_template(dreams, n=5, seed=42)
    second = mgt.sample_gold_template(dreams, n=5, seed=42)
    assert [row['source_id'] for row in first] == [row['source_id'] for row in second]


def test_sample_gold_template_differs_across_seeds_for_large_pool():
    dreams = _dreams(50)
    a = mgt.sample_gold_template(dreams, n=5, seed=1)
    b = mgt.sample_gold_template(dreams, n=5, seed=2)
    assert [row['source_id'] for row in a] != [row['source_id'] for row in b]


def test_write_gold_csv_round_trip(tmp_path):
    sample = mgt.sample_gold_template(_dreams(3), n=3, seed=1)
    out = tmp_path / 'gold.csv'
    mgt.write_gold_csv(sample, str(out))

    with open(out, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert set(rows[0].keys()) == set(mgt.CSV_FIELDS)
    assert all(r['gold_genre'] == '' for r in rows)


def test_fetch_candidate_dreams_combines_reddit_and_dreambank(tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'gold_test.db'))
    db.init_db()
    db.insert_reddit_dream({
        'post_id': 'r1', 'subreddit': 'dreams', 'created_utc': 1.0, 'flair': None,
        'title': 'Title', 'body': 'Body text', 'score': 1, 'num_comments': 0,
        'word_count': 2, 'content_hash': 'hash-r1',
    })
    db.insert_dreambank_dream({
        'source_label': 'series.txt', 'text': 'A dreambank dream', 'word_count': 3,
        'content_hash': 'hash-d1',
    })

    candidates = mgt.fetch_candidate_dreams()

    sources = {c['source'] for c in candidates}
    assert sources == {'reddit', 'dreambank'}
    assert len(candidates) == 2
