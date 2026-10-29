"""
correlation.py — a simple Pearson correlation between a sleep-log
field (duration/quality) and a dream-text feature. Descriptive only:
reports r, a p-value, and n -- NO diagnostic or medical interpretation,
and this file must never be used to generate one (see docs/biology_link.md).
"""
from scipy.stats import pearsonr


def simple_correlation(x: list[float], y: list[float]) -> dict:
    """{'r', 'p_value', 'n'}. r=0.0/p_value=1.0 if fewer than 2 paired points."""
    if len(x) != len(y):
        raise ValueError('x and y must be the same length')
    if len(x) < 2:
        return {'r': 0.0, 'p_value': 1.0, 'n': len(x)}

    r, p_value = pearsonr(x, y)
    return {'r': float(r), 'p_value': float(p_value), 'n': len(x)}


def sleep_log_vs_feature(sleep_log_dreams: list[dict], sleep_log_field: str, feature_fn) -> dict:
    """
    sleep_log_dreams: dream dicts with a `text` field and optionally
    `sleep_log_field` (e.g. 'sleep_quality' or 'sleep_duration_hours').
    Rows missing that field are skipped. feature_fn(text) -> float.
    """
    pairs = [
        (dream[sleep_log_field], feature_fn(dream['text']))
        for dream in sleep_log_dreams
        if dream.get(sleep_log_field) is not None
    ]
    if not pairs:
        return {'r': 0.0, 'p_value': 1.0, 'n': 0}
    x, y = zip(*pairs)
    return simple_correlation(list(x), list(y))
