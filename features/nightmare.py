"""
nightmare.py — nightmare-vs-pleasant signal for a dream text.

Like lucidity.py, there is no gold-labeled nightmare/pleasant data yet
(data/gold_template.csv has 0 rows), so nightmare_probability() is a
transparent heuristic built from emotion_arc() (emotion.py) and the
aggression/misfortune vs. friendliness/good_fortune counts (hvdc.py),
not a trained classifier. The 0.4/0.6 thresholds in classify_dream_type()
are arbitrary placeholders, not calibrated against any ground truth --
train_nightmare_classifier() is the hook to replace this once gold
labels exist.
"""
import math

from emotion import emotion_arc
from hvdc import hvdc_counts


def nightmare_probability(text: str) -> float:
    """Heuristic probability in [0, 1] that a dream reads as a nightmare."""
    if not text or not text.strip():
        return 0.5

    arc = emotion_arc(text)
    counts = hvdc_counts(text)
    n = max(counts['word_count'], 1)

    negative_signal = -arc['end']
    aggression_misfortune_ratio = (counts['aggression'] + counts['misfortune']) / n
    friendliness_good_ratio = (counts['friendliness'] + counts['good_fortune']) / n

    raw_score = negative_signal + 10 * (aggression_misfortune_ratio - friendliness_good_ratio)
    return 1 / (1 + math.exp(-raw_score))


def classify_dream_type(text: str, nightmare_threshold: float = 0.6, pleasant_threshold: float = 0.4) -> str:
    """'nightmare', 'pleasant', or 'neutral', from nightmare_probability()."""
    probability = nightmare_probability(text)
    if probability >= nightmare_threshold:
        return 'nightmare'
    if probability <= pleasant_threshold:
        return 'pleasant'
    return 'neutral'


def train_nightmare_classifier(texts: list[str], labels: list[str]):
    """
    Fit a TF-IDF + LogisticRegression nightmare/pleasant classifier on
    gold labels. Not yet called anywhere -- a hook for once
    data/gold_template.csv has real gold_nightmare annotations.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if len(texts) != len(labels):
        raise ValueError('texts and labels must be the same length')
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(texts, labels)
    return model
