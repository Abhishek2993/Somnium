"""
Flask test-client tests for /api/health and /api/analyze. Deliberately
does NOT exercise /api/dreams (POST/GET/DELETE) here -- those persist
to the real somnium.db via db.DB_PATH, and this test module imports
the real `app` module (triggering its real init_db()/model-loading
side effects), so writing through it would pollute the real database.
db.py's own CRUD is already covered against a temp DB in test_db.py.
"""
import config
from app import app as flask_app


def _client():
    flask_app.config['TESTING'] = True
    return flask_app.test_client()


def test_health():
    response = _client().get('/api/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_analyze_requires_text():
    response = _client().post('/api/analyze', json={'text': ''})
    assert response.status_code == 400


def test_analyze_returns_expected_shape_without_feature_profile(monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_FEATURE_PROFILE', False)
    response = _client().post('/api/analyze', json={'text': 'I flew over an enchanted castle with a dragon'})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body.keys()) == {
        'title', 'fixed_text', 'genre', 'confidence', 'all_scores', 'keywords', 'entities', 'disclaimer',
    }
    assert body['genre']
    assert 'feature_profile' not in body


def test_analyze_includes_feature_profile_when_enabled(monkeypatch):
    monkeypatch.setattr(config, 'ENABLE_FEATURE_PROFILE', True)
    response = _client().post(
        '/api/analyze',
        json={'text': 'I realized I was dreaming and flew over an enchanted castle with a dragon'},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert 'feature_profile' in body
    profile = body['feature_profile']
    assert set(profile.keys()) == {'emotion_arc', 'bizarreness', 'sensory_ratios', 'lucid_probability'}
    assert profile['lucid_probability'] > 0


def test_analyze_disclaimer_is_non_diagnostic():
    response = _client().post('/api/analyze', json={'text': 'I was falling through a dark void'})
    body = response.get_json()
    disclaimer = body['disclaimer'].lower()
    assert 'not' in disclaimer
    assert 'medical' in disclaimer or 'diagnostic' in disclaimer
