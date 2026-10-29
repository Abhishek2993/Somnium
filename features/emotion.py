"""
emotion.py — per-sentence emotion (sentiment valence) scores and the
overall emotion arc of a dream (start, end, slope).

Uses NLTK's VADER lexicon for valence (compound score in [-1, 1]).
This is sentiment valence (positive <-> negative), not a discrete
multi-class emotion model (joy/fear/anger/...) -- that would need a
labeled emotion dataset we don't have. Documented here rather than
silently overclaiming.
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

for pkg, path in (('vader_lexicon', 'sentiment/vader_lexicon'), ('punkt', 'tokenizers/punkt'),
                   ('punkt_tab', 'tokenizers/punkt_tab')):
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

_SIA = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def sentence_emotion_scores(text: str) -> list[dict]:
    """Per-sentence VADER scores: [{'sentence', 'compound', 'pos', 'neg', 'neu'}, ...]."""
    analyzer = _get_analyzer()
    scores = []
    for sentence in split_sentences(text):
        polarity = analyzer.polarity_scores(sentence)
        scores.append({
            'sentence': sentence,
            'compound': polarity['compound'],
            'pos': polarity['pos'],
            'neg': polarity['neg'],
            'neu': polarity['neu'],
        })
    return scores


def emotion_arc(text: str) -> dict:
    """
    The emotional trajectory of a dream, in terms of VADER compound
    valence per sentence: {'start', 'end', 'slope', 'scores'}.

    - start/end: compound valence of the first/last sentence.
    - slope: least-squares slope of compound valence over sentence
      index (sentence 0, 1, 2, ...) -- positive means the dream grows
      more positive over its course, negative means it darkens.
    All three are 0.0 for empty text or a single sentence (slope only).
    """
    scores = sentence_emotion_scores(text)
    compounds = [s['compound'] for s in scores]

    if not compounds:
        return {'start': 0.0, 'end': 0.0, 'slope': 0.0, 'scores': []}

    start = compounds[0]
    end = compounds[-1]
    slope = _linear_slope(compounds)
    return {'start': start, 'end': end, 'slope': slope, 'scores': compounds}


def _linear_slope(values: list[float]) -> float:
    """Least-squares slope of values against their index (0, 1, 2, ...)."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return numerator / denominator
