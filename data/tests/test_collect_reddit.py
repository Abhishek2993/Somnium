from types import SimpleNamespace

import collect_reddit as cr

LONG_ENGLISH_BODY = (
    "I was walking through a long hallway that kept stretching further away "
    "every time I took a step forward, and the lights above me flickered in "
    "a strange rhythm while I tried to find a door that would let me out of "
    "this endless corridor before something behind me caught up with me."
)


def _fake_submission(**overrides):
    base = dict(
        id='abc123',
        subreddit='dreams',
        created_utc=1700000000.0,
        link_flair_text='Nightmare',
        title='A long hallway',
        selftext=LONG_ENGLISH_BODY,
        score=12,
        num_comments=3,
        author=SimpleNamespace(name='someuser'),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_record_shape_and_no_username_field():
    record = cr.build_record(_fake_submission())
    assert record['post_id'] == 'abc123'
    assert record['subreddit'] == 'dreams'
    assert record['word_count'] == cr.word_count(record['body'])
    assert 'author' not in record
    assert 'username' not in record


def test_build_record_cleans_body():
    submission = _fake_submission(selftext=LONG_ENGLISH_BODY + ' https://example.com/spoiler')
    record = cr.build_record(submission)
    assert 'http' not in record['body']


def test_is_deleted_account_true_when_author_none():
    assert cr.is_deleted_account(_fake_submission(author=None)) is True


def test_is_deleted_account_false_when_author_present():
    assert cr.is_deleted_account(_fake_submission()) is False


def test_should_store_rejects_deleted_account():
    submission = _fake_submission(author=None)
    record = cr.build_record(submission)
    assert cr.should_store(submission, record) is False


def test_should_store_rejects_short_body():
    submission = _fake_submission(selftext='too short')
    record = cr.build_record(submission)
    assert cr.should_store(submission, record, min_words=50) is False


def test_should_store_accepts_valid_post():
    submission = _fake_submission()
    record = cr.build_record(submission)
    assert cr.should_store(submission, record, min_words=10) is True


def test_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, 'POST_IDS_DIR', str(tmp_path))
    assert cr.load_checkpoint('dreams') == set()
    cr.append_checkpoint('dreams', 'id1')
    cr.append_checkpoint('dreams', 'id2')
    assert cr.load_checkpoint('dreams') == {'id1', 'id2'}


def test_checkpoint_is_per_subreddit(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, 'POST_IDS_DIR', str(tmp_path))
    cr.append_checkpoint('dreams', 'id1')
    cr.append_checkpoint('nightmares', 'id2')
    assert cr.load_checkpoint('dreams') == {'id1'}
    assert cr.load_checkpoint('nightmares') == {'id2'}
