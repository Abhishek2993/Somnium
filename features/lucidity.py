"""
lucidity.py — lucid-dream probability for a dream text.

No gold-labeled lucid/non-lucid data exists yet (see data/gold_template.csv,
0 rows so far), so this is a transparent keyword heuristic, not a trained
classifier. Once gold_lucid labels exist, train_lucid_classifier() can be
used to fit a real supervised model and should replace this as the
primary signal -- see Phase 4's model-comparison report.
"""
import re

LUCID_PHRASES = [
    r"\bi (?:was|became|am) (?:lucid|aware (?:that )?i was dreaming)\b",
    r"\bi (?:realized|realised|knew) (?:that )?(?:i was|it was a) dream(?:ing)?\b",
    r"\blucid dream\b",
    r"\btook control of the dream\b",
    r"\b(?:i )?control(?:led)? (?:the|my) dream\b",
    r"\breality check\b",
    r"\bdream sign(?:s)?\b",
]
_LUCID_RE = [re.compile(p, re.IGNORECASE) for p in LUCID_PHRASES]


def lucid_probability(text: str) -> float:
    """
    Heuristic lucid-dream probability in [0, 1]: fraction of the known
    lucidity phrases found in the text, capped at 1.0. 0.0 for empty text.
    """
    if not text or not text.strip():
        return 0.0
    hits = sum(1 for pattern in _LUCID_RE if pattern.search(text))
    return min(1.0, hits / 2)  # 2+ distinct lucidity phrases => full confidence


def train_lucid_classifier(texts: list[str], labels: list[bool]):
    """
    Fit a TF-IDF + LogisticRegression lucid/non-lucid classifier on gold
    labels. Not yet called anywhere -- a hook for once data/gold_template.csv
    has real gold_lucid annotations (currently 0 rows).
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
