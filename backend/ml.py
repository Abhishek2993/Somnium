"""
ml.py — scikit-learn genre classifier for Somnium
TF-IDF vectoriser + LinearSVC pipeline
Trained on a synthetic corpus of genre-labelled sentences at startup.
"""
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'genre_model.pkl')

# ── Synthetic training corpus ─────────────────────────────────────────────────
# Each entry: (text, genre)
_CORPUS = [
    # Fantasy
    ("I flew over an enchanted castle with dragons and a wizard casting spells", "Fantasy"),
    ("A mystical elf guided me through an ancient magical kingdom", "Fantasy"),
    ("I discovered a fairy village hidden beneath a glowing mushroom forest", "Fantasy"),
    ("A sorcerer turned the entire realm to gold with a flick of his wand", "Fantasy"),
    ("Unicorns galloped across the sky while pixies sang enchanted songs", "Fantasy"),
    ("The dragon breathed fire that transformed into magical butterflies", "Fantasy"),
    ("I wielded a sword of light to defeat an evil warlock in enchanted ruins", "Fantasy"),
    ("Potions bubbled in the wizard tower as spells glowed on parchment", "Fantasy"),

    # Horror
    ("A shadowy monster chased me through a dark abandoned house", "Horror"),
    ("Ghosts screamed in the crumbling mansion as blood dripped from the walls", "Horror"),
    ("I was trapped by a demon that wore a human face in the darkness", "Horror"),
    ("Zombies flooded the streets while I hid in a cellar full of corpses", "Horror"),
    ("A vampire watched from the shadows while sinister laughter echoed", "Horror"),
    ("Something evil lurked behind every door as I crept through the terror", "Horror"),
    ("A knife-wielding figure chased me through fog-filled streets at night", "Horror"),
    ("Dead figures rose from graves as I tried to scream but no sound came", "Horror"),

    # Adventure
    ("I explored a jungle island searching for buried treasure with a map", "Adventure"),
    ("My ship sailed across the ocean towards uncharted mountains", "Adventure"),
    ("I climbed a cliff face to discover ancient ruins hidden in the wilderness", "Adventure"),
    ("A quest took me through forests rivers and deserts to rescue someone", "Adventure"),
    ("I parachuted into a remote valley and had to survive alone", "Adventure"),
    ("Riding horses across vast plains we discovered a hidden civilisation", "Adventure"),
    ("An expedition through the caves revealed a lost underground world", "Adventure"),
    ("I navigated rapids on a raft fighting against the raging river", "Adventure"),

    # Mystery
    ("A secret door led to a hidden room full of mysterious clues", "Mystery"),
    ("I played detective investigating a puzzle with hidden suspects", "Mystery"),
    ("Someone was leaving cryptic ciphers around an unknown city", "Mystery"),
    ("The maze had no exit and every door revealed a new secret conspiracy", "Mystery"),
    ("I had to solve a riddle before the truth about the crime was revealed", "Mystery"),
    ("Whispers pointed to a suspect in the dark underground investigation", "Mystery"),
    ("The detective uncovered a hidden layer beneath the apparent crime scene", "Mystery"),
    ("Clues scattered across the museum pointed to an impossible conspiracy", "Mystery"),

    # Sci-Fi
    ("I travelled through space to a distant planet orbiting two stars", "Sci-Fi"),
    ("Robots marched across the neon city as AI took control of the future", "Sci-Fi"),
    ("An alien spacecraft landed and communicated through pulsing light", "Sci-Fi"),
    ("I stepped through a dimension portal into a cyberpunk world of chrome", "Sci-Fi"),
    ("The android asked if it had a soul while the galaxy burned outside", "Sci-Fi"),
    ("Lasers lit up the asteroid belt during an interstellar war", "Sci-Fi"),
    ("Teleporting between planets I witnessed technology that bent physics", "Sci-Fi"),
    ("A time machine took me to the far future where machines governed all", "Sci-Fi"),

    # Romance
    ("I danced under the stars with someone I deeply loved", "Romance"),
    ("A tender kiss at sunset made my heart overflow with passion", "Romance"),
    ("We shared a long embrace in a rose garden full of affection", "Romance"),
    ("My soulmate appeared in the dream and our hearts connected instantly", "Romance"),
    ("A wedding on a hilltop filled with warmth love and blossoming flowers", "Romance"),
    ("I longed for them and felt their presence warm beside me all night", "Romance"),
    ("Hand in hand we walked through a field of blooming roses at dusk", "Romance"),
    ("The feeling of being deeply cherished wrapped around me like warmth", "Romance"),

    # Thriller
    ("A spy mission went wrong and I was being hunted by assassins", "Thriller"),
    ("I uncovered a government conspiracy and had to run before capture", "Thriller"),
    ("Dodging snipers I carried classified information through a war zone", "Thriller"),
    ("An undercover operation turned deadly when the betrayal was revealed", "Thriller"),
    ("Explosives were planted everywhere and I had seconds to escape", "Thriller"),
    ("A mole in the agency had leaked my identity to enemy operatives", "Thriller"),
    ("Armed men chased my car through narrow streets at high speed", "Thriller"),
    ("I defused a bomb seconds before it would have destroyed the city", "Thriller"),

    # Surreal
    ("I floated through melting walls while colours morphed into sounds", "Surreal"),
    ("Everything shrank and expanded impossibly as I drifted through void", "Surreal"),
    ("Flying without wings I passed through layers of shifting dimensions", "Surreal"),
    ("The sky was liquid and I swam upward through infinite rainbow fog", "Surreal"),
    ("Objects morphed into other objects endlessly in a loop with no logic", "Surreal"),
    ("I watched myself from outside my body as reality distorted around me", "Surreal"),
    ("Gravity reversed and buildings became impossible stairs to nowhere", "Surreal"),
    ("A bizarre parade of giant talking clocks led me into a mirror world", "Surreal"),

    # Spiritual
    ("Angels descended in a beam of divine light filling me with peace", "Spiritual"),
    ("I meditated in a sacred temple and felt my soul leave my body", "Spiritual"),
    ("A spiritual guide led me to enlightenment in a cosmic realm of light", "Spiritual"),
    ("The cosmos spoke to me and I understood the karma of all existence", "Spiritual"),
    ("I transcended physical form and merged with a holy divine consciousness", "Spiritual"),
    ("Prayers echoed in a celestial space as sacred geometry surrounded me", "Spiritual"),
    ("My soul ascended through clouds toward a luminous divine presence", "Spiritual"),
    ("I heard the universe breathe and felt at peace with all living things", "Spiritual"),

    # Comedy
    ("A clown chased me with a giant rubber chicken through a silly maze", "Comedy"),
    ("Everything was absurdly funny and everyone laughed at bizarre pranks", "Comedy"),
    ("I accidentally tripped into a cartoon world full of hilarious nonsense", "Comedy"),
    ("A ridiculous series of mishaps turned a simple errand into chaos", "Comedy"),
    ("Everyone in the dream wore silly costumes at an absurd party", "Comedy"),
    ("I gave a speech and everything went comically wrong in front of crowds", "Comedy"),
    ("A dog drove a car while cats filed paperwork in a giggling office", "Comedy"),
    ("The most ridiculous situation unfolded and I could not stop laughing", "Comedy"),
]

GENRES = sorted(set(label for _, label in _CORPUS))


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)),
    ])


def _train() -> Pipeline:
    texts, labels = zip(*_CORPUS)
    model = _build_pipeline()
    model.fit(texts, labels)
    return model


def load_or_train() -> Pipeline:
    """Load cached model or train a fresh one."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    model = _train()
    joblib.dump(model, MODEL_PATH)
    return model


# Global model — loaded once at import time
_MODEL: Pipeline | None = None


def get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_or_train()
    return _MODEL


def predict_genre(text: str) -> dict:
    """
    Predict genre and return confidence score.
    Returns {genre, confidence, all_scores}
    """
    model = get_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    genre = classes[best_idx]
    confidence = round(float(proba[best_idx]), 3)
    all_scores = {c: round(float(p), 3) for c, p in zip(classes, proba)}
    return {
        'genre': genre,
        'confidence': confidence,
        'all_scores': all_scores,
    }


def retrain_with_dream(text: str, genre: str):
    """Incrementally re-train model when user confirms a genre (online learning stub)."""
    global _MODEL
    # Append to corpus and retrain — simple but effective for small datasets
    _CORPUS.append((text, genre))
    _MODEL = _train()
    joblib.dump(_MODEL, MODEL_PATH)
