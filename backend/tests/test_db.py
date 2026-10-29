import config
import db


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'test_somnium.db'))
    db.init_db()


def test_init_db_creates_dreams_table(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    with db.get_conn() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'dreams' in tables


def test_insert_and_get_dream_by_id(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    dream_id = db.insert_dream({
        'date': '2026-01-01', 'title': 'Test', 'text': 'I flew over a castle',
        'genre': 'Fantasy', 'confidence': 0.9, 'mood': 'happy', 'type': 'Lucid',
        'keywords': ['castle', 'fly'], 'entities': [{'text': 'Paris', 'label': 'GPE'}],
    })
    dream = db.get_dream_by_id(dream_id)
    assert dream['title'] == 'Test'
    assert dream['keywords'] == ['castle', 'fly']
    assert dream['entities'] == [{'text': 'Paris', 'label': 'GPE'}]


def test_get_dream_by_id_missing_returns_none(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    assert db.get_dream_by_id(999) is None


def test_get_all_dreams_returns_all_inserted_rows(tmp_path, monkeypatch):
    # created_at has second resolution, so two inserts in the same test can
    # tie -- assert membership/count rather than strict recency order.
    _fresh_db(tmp_path, monkeypatch)
    first_id = db.insert_dream({'date': '2026-01-01', 'text': 'first'})
    second_id = db.insert_dream({'date': '2026-01-02', 'text': 'second'})
    dreams = db.get_all_dreams()
    assert {d['id'] for d in dreams} == {first_id, second_id}
    assert len(dreams) == 2


def test_delete_dream_removes_row_and_reports_success(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    dream_id = db.insert_dream({'date': '2026-01-01', 'text': 'to be deleted'})
    assert db.delete_dream(dream_id) is True
    assert db.get_dream_by_id(dream_id) is None
    assert db.delete_dream(dream_id) is False


def _reddit_record(**overrides):
    base = dict(
        post_id='abc123', subreddit='dreams', created_utc=1700000000.0,
        flair='Nightmare', title='A dream', body='I walked through a hallway.',
        score=5, num_comments=1, word_count=5, content_hash='hash1',
    )
    base.update(overrides)
    return base


def test_insert_reddit_dream_and_fetch(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    row_id = db.insert_reddit_dream(_reddit_record())
    assert row_id is not None
    rows = db.get_reddit_dreams()
    assert len(rows) == 1
    assert rows[0]['post_id'] == 'abc123'
    assert 'author' not in rows[0]


def test_insert_reddit_dream_ignores_duplicate_post_id(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    db.insert_reddit_dream(_reddit_record())
    dup = db.insert_reddit_dream(_reddit_record(content_hash='different-hash'))
    assert dup is None
    assert len(db.get_reddit_dreams()) == 1


def test_insert_reddit_dream_ignores_duplicate_content_hash(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    db.insert_reddit_dream(_reddit_record())
    dup = db.insert_reddit_dream(_reddit_record(post_id='different-id'))
    assert dup is None
    assert len(db.get_reddit_dreams()) == 1


def test_get_reddit_dreams_filters_by_subreddit(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    db.insert_reddit_dream(_reddit_record(post_id='a', content_hash='ha', subreddit='dreams'))
    db.insert_reddit_dream(_reddit_record(post_id='b', content_hash='hb', subreddit='Nightmares'))
    assert {r['post_id'] for r in db.get_reddit_dreams('dreams')} == {'a'}


def test_get_reddit_post_ids(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    db.insert_reddit_dream(_reddit_record(post_id='a', content_hash='ha'))
    db.insert_reddit_dream(_reddit_record(post_id='b', content_hash='hb'))
    assert db.get_reddit_post_ids() == {'a', 'b'}


def test_insert_dreambank_dream_and_fetch_dedupes_by_hash(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    record = {'source_label': 'series1.txt', 'text': 'a long dream text', 'word_count': 4, 'content_hash': 'xyz'}
    first = db.insert_dreambank_dream(record)
    second = db.insert_dreambank_dream(record)
    assert first is not None
    assert second is None
    assert len(db.get_dreambank_dreams()) == 1


def test_sleep_log_columns_absent_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_SLEEP_MODULE', False)
    _fresh_db(tmp_path, monkeypatch)
    with db.get_conn() as conn:
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(dreams)')}
    assert 'sleep_duration_hours' not in columns
    assert 'sleep_quality' not in columns


def test_sleep_log_columns_added_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_SLEEP_MODULE', True)
    _fresh_db(tmp_path, monkeypatch)
    with db.get_conn() as conn:
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(dreams)')}
    assert 'sleep_duration_hours' in columns
    assert 'sleep_quality' in columns


def test_sleep_log_migration_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_SLEEP_MODULE', True)
    _fresh_db(tmp_path, monkeypatch)
    db.init_db()  # calling init_db() again must not raise (duplicate ADD COLUMN)
    with db.get_conn() as conn:
        columns = [row['name'] for row in conn.execute('PRAGMA table_info(dreams)')]
    assert columns.count('sleep_quality') == 1


def test_insert_dream_stores_sleep_fields_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_SLEEP_MODULE', True)
    _fresh_db(tmp_path, monkeypatch)
    dream_id = db.insert_dream({
        'date': '2026-01-01', 'text': 'a dream',
        'sleep_duration_hours': 7.5, 'sleep_quality': 4,
    })
    dream = db.get_dream_by_id(dream_id)
    assert dream['sleep_duration_hours'] == 7.5
    assert dream['sleep_quality'] == 4


def test_insert_dream_ignores_sleep_fields_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_SLEEP_MODULE', False)
    _fresh_db(tmp_path, monkeypatch)
    dream_id = db.insert_dream({
        'date': '2026-01-01', 'text': 'a dream',
        'sleep_duration_hours': 7.5, 'sleep_quality': 4,
    })
    dream = db.get_dream_by_id(dream_id)
    assert 'sleep_duration_hours' not in dream
    assert 'sleep_quality' not in dream
