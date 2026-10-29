"""
train_sbert_logreg.py — model (b): all-MiniLM-L6-v2 sentence embeddings
(features/embeddings.py) + LogisticRegression.
"""
import paths  # noqa: F401

from sklearn.linear_model import LogisticRegression

from embeddings import embed_sentences

RANDOM_SEED = 42


def fit(texts: list[str], labels: list[str], seed: int = RANDOM_SEED) -> dict:
    """Returns a model dict: {'clf': fitted classifier}."""
    X = embed_sentences(texts)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, labels)
    return {'clf': clf}


def predict(model: dict, texts: list[str]) -> list[str]:
    X = embed_sentences(texts)
    return list(model['clf'].predict(X))
