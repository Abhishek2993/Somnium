"""
train_logreg_tfidf_features.py — model (a): TF-IDF + the features/
numeric feature vector (dream_features.py), fed into LogisticRegression.
Compared against the existing TF-IDF+LinearSVC baseline and the SBERT
model in models/evaluate.py.
"""
import paths  # noqa: F401

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dream_features import numeric_feature_matrix

RANDOM_SEED = 42


def _combine(tfidf_matrix, texts):
    numeric = csr_matrix(numeric_feature_matrix(texts))
    return hstack([tfidf_matrix, numeric]).tocsr()


def fit(texts: list[str], labels: list[str], seed: int = RANDOM_SEED) -> dict:
    """Returns a model dict: {'tfidf': fitted vectorizer, 'clf': fitted classifier}."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=1, max_features=5000, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(texts)
    X = _combine(X_tfidf, texts)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, labels)
    return {'tfidf': tfidf, 'clf': clf}


def predict(model: dict, texts: list[str]) -> list[str]:
    X_tfidf = model['tfidf'].transform(texts)
    X = _combine(X_tfidf, texts)
    return list(model['clf'].predict(X))
