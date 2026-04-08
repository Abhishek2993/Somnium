"""
nlp.py — spaCy + NLTK NLP pipeline for Somnium
Uses spaCy for tokenisation, lemmatisation, NER
Uses NLTK for stopword filtering
"""
import re
import spacy
import nltk
from nltk.corpus import stopwords

# ── Bootstrap NLTK data (first run only) ──────────────────────────────────────
for pkg in ('stopwords', 'punkt'):
    try:
        nltk.data.find(f'corpora/{pkg}' if pkg == 'stopwords' else f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Load spaCy model ───────────────────────────────────────────────────────────
try:
    NLP = spacy.load('en_core_web_sm')
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    NLP = spacy.load('en_core_web_sm')

STOP_WORDS = set(stopwords.words('english')) | {
    'dream', 'dreamt', 'dreamed', 'see', 'saw', 'feel', 'felt',
    'go', 'went', 'come', 'came', 'know', 'think', 'thought',
    'like', 'just', 'suddenly', 'then', 'everything', 'something',
    'somehow', 'maybe', 'seemed', 'started', 'began', 'could', 'would'
}

# POS tags we care about for keywords
GOOD_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ'}


def fix_grammar(text: str) -> str:
    """Basic grammar normalisation."""
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\.(\S)', r'. \1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def extract_keywords(text: str, top_n: int = 14) -> list[str]:
    """
    Use spaCy to lemmatise tokens, then filter by POS + stopwords.
    Returns de-duped keyword list.
    """
    doc = NLP(text.lower())
    seen = set()
    keywords = []
    for token in doc:
        lemma = token.lemma_.strip()
        if (
            token.pos_ in GOOD_POS
            and len(lemma) > 3
            and lemma not in STOP_WORDS
            and lemma.isalpha()
            and lemma not in seen
        ):
            seen.add(lemma)
            keywords.append(lemma)
    return keywords[:top_n]


def get_named_entities(text: str) -> list[dict]:
    """
    Run spaCy NER and return filtered entities
    (people, places, objects, works of art, etc.)
    """
    doc = NLP(text)
    allowed = {'PERSON', 'GPE', 'LOC', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'EVENT', 'FAC'}
    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ in allowed and ent.text not in seen:
            seen.add(ent.text)
            entities.append({'text': ent.text, 'label': ent.label_})
    return entities


def generate_title(text: str, genre: str) -> str:
    """Generate an evocative dream title using spaCy + genre prefix."""
    prefixes = {
        'Fantasy':   ['The Enchanted', 'A Realm of', 'Whispers of', 'Beyond the'],
        'Horror':    ['Shadows of', 'The Dark', 'Echoes of', 'Fear in'],
        'Adventure': ['Beyond the', 'Journey to', 'The Quest for', 'Lost in'],
        'Mystery':   ['Secrets of', 'The Hidden', 'Riddles of', "The Unknown"],
        'Sci-Fi':    ['Signal from', 'Across the', 'The Last', 'Echoes of'],
        'Romance':   ['Hearts in', 'A Dream of', 'Where Love', 'Longing for'],
        'Thriller':  ['Run from', 'Behind the', 'No Escape from', 'Chasing'],
        'Surreal':   ['Drifting Through', 'The Fluid', 'Lost in', 'Falling into'],
        'Spiritual': ['Light of', 'The Sacred', 'Echoes of the', 'Journey to'],
        'Comedy':    ['The Absurd', 'Much Ado About', 'Laughing Through', 'The Bizarre'],
    }
    doc = NLP(text)
    nouns = [t.lemma_.capitalize() for t in doc if t.pos_ in ('NOUN', 'PROPN') and len(t.lemma_) > 3]
    noun = nouns[0] if nouns else 'the Unknown'
    import random
    prefix_list = prefixes.get(genre, prefixes['Surreal'])
    return f"{random.choice(prefix_list)} {noun}"
