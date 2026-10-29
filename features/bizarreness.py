"""
bizarreness.py — a dream's bizarreness score: cosine distance between
consecutive sentence embeddings (all-MiniLM-L6-v2 via embeddings.py).
The idea: the more a dream "jumps" between semantically unrelated
sentences, the higher its average/peak bizarreness.
"""
import numpy as np

from embeddings import embed_sentences
from emotion import split_sentences


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return 1.0 - similarity


def bizarreness_score(text: str) -> dict:
    """
    {'per_pair': [...], 'mean': ..., 'max': ..., 'num_sentences': n}.
    A dream with fewer than 2 sentences has no consecutive pair to
    compare, so all scores are 0.0.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return {'per_pair': [], 'mean': 0.0, 'max': 0.0, 'num_sentences': len(sentences)}

    embeddings = embed_sentences(sentences)
    distances = [
        cosine_distance(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]
    return {
        'per_pair': distances,
        'mean': float(np.mean(distances)),
        'max': float(np.max(distances)),
        'num_sentences': len(sentences),
    }
