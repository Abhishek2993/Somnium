"""
app.py — Flask REST API for Somnium Dream Journal
Run: python app.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS

import config
from db import init_db, insert_dream, get_all_dreams, delete_dream, get_dream_by_id
from nlp import extract_keywords, get_named_entities, fix_grammar, generate_title
from ml import predict_genre

app = Flask(__name__)
CORS(app)  # Allow frontend (file:// or localhost) to call the API

NON_DIAGNOSTIC_DISCLAIMER = (
    "This analysis is for personal reflection and entertainment only. "
    "It is not medical, psychological, or diagnostic advice."
)

# Initialise DB on startup
init_db()

# Pre-load NLP + ML models on startup (avoids first-request lag)
print("⏳ Loading NLP model...")
from nlp import NLP  # triggers spaCy load
print("⏳ Training / loading ML classifier...")
from ml import get_model; get_model()
print("✅ Somnium backend ready.")


# ── Health check ───────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '2.0.0'})


# ── Analyse dream (NLP + ML) ───────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'text is required'}), 400

    # NLP pipeline
    fixed        = fix_grammar(text)
    keywords     = extract_keywords(fixed)
    entities     = get_named_entities(fixed)

    # ML genre prediction
    prediction   = predict_genre(fixed)
    genre        = prediction['genre']
    confidence   = prediction['confidence']
    all_scores   = prediction['all_scores']

    # Generate evocative title
    title        = generate_title(fixed, genre)

    response = {
        'title':       title,
        'fixed_text':  fixed,
        'genre':       genre,
        'confidence':  confidence,
        'all_scores':  all_scores,
        'keywords':    keywords,
        'entities':    entities,
        'disclaimer':  NON_DIAGNOSTIC_DISCLAIMER,
    }

    # Optional feature profile (emotion arc, bizarreness, sensory ratios,
    # lucid probability) -- additive, behind ENABLE_FEATURE_PROFILE so the
    # existing response shape is unchanged when it's off.
    if config.ENABLE_FEATURE_PROFILE:
        from feature_profile import build_feature_profile
        response['feature_profile'] = build_feature_profile(fixed)

    return jsonify(response)


# ── Save dream ─────────────────────────────────────────────────────────────────
@app.route('/api/dreams', methods=['POST'])
def save_dream():
    data = request.get_json(force=True)
    required = ['date', 'text']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    dream_id = insert_dream({
        'date':       data.get('date'),
        'title':      data.get('title', ''),
        'text':       data.get('text'),
        'genre':      data.get('genre', 'Surreal'),
        'confidence': data.get('confidence', 0.0),
        'mood':       data.get('mood', ''),
        'type':       data.get('type', 'Normal'),
        'keywords':   data.get('keywords', []),
        'entities':   data.get('entities', []),
        # Optional sleep-log fields, only persisted when ENABLE_SLEEP_MODULE
        # is on (see db.insert_dream / docs/biology_link.md).
        'sleep_duration_hours': data.get('sleep_duration_hours'),
        'sleep_quality':        data.get('sleep_quality'),
    })
    dream = get_dream_by_id(dream_id)
    return jsonify({'id': dream_id, 'dream': dream}), 201


# ── Get all dreams ─────────────────────────────────────────────────────────────
@app.route('/api/dreams', methods=['GET'])
def list_dreams():
    dreams = get_all_dreams()
    return jsonify(dreams)


# ── Delete dream ───────────────────────────────────────────────────────────────
@app.route('/api/dreams/<int:dream_id>', methods=['DELETE'])
def remove_dream(dream_id):
    if delete_dream(dream_id):
        return jsonify({'deleted': dream_id})
    return jsonify({'error': 'Not found'}), 404


# ── Get single dream ───────────────────────────────────────────────────────────
@app.route('/api/dreams/<int:dream_id>', methods=['GET'])
def get_dream(dream_id):
    dream = get_dream_by_id(dream_id)
    if dream:
        return jsonify(dream)
    return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    print("🌙 Starting Somnium backend on http://localhost:5000")
    app.run(debug=True, port=5000)
