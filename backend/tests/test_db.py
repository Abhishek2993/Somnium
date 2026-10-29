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
