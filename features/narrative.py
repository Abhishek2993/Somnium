"""
narrative.py — basic narrative-structure features: length, dominant
verb tense, and first-person pronoun ratio.
"""
import re

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

for pkg, path in (
    ('punkt', 'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
):
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

PAST_TAGS = {'VBD', 'VBN'}
PRESENT_TAGS = {'VBP', 'VBZ', 'VBG'}

FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ''))


def dominant_tense(text: str) -> str:
    """'past', 'present', or 'unknown' (no tensed verbs found)."""
    if not text or not text.strip():
        return 'unknown'
    tags = [tag for _, tag in pos_tag(word_tokenize(text))]
    past = sum(1 for t in tags if t in PAST_TAGS)
    present = sum(1 for t in tags if t in PRESENT_TAGS)
    if past == 0 and present == 0:
        return 'unknown'
    return 'past' if past >= present else 'present'


def first_person_ratio(text: str) -> float:
    tokens = [w.lower() for w in _WORD_RE.findall(text or '')]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS) / len(tokens)


def narrative_profile(text: str) -> dict:
    return {
        'length': word_count(text),
        'tense': dominant_tense(text),
        'first_person_ratio': first_person_ratio(text),
    }
