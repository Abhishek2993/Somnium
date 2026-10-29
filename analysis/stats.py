"""
stats.py — effect size (Cohen's d) and bootstrap confidence interval
helpers used by compare_dream_types.py and emotion_arcs.py.
"""
import numpy as np

DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d (pooled-SD standardized mean difference, a - b). 0.0 if
    either group has fewer than 2 values or pooled SD is 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_ci(data: list[float], statistic=np.mean, n_resamples: int = DEFAULT_N_RESAMPLES,
                  confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> dict:
    """Bootstrap CI for `statistic` over `data`. {'point_estimate','low','high','confidence','n'}."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return {'point_estimate': 0.0, 'low': 0.0, 'high': 0.0, 'confidence': confidence, 'n': 0}

    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot_stats, [alpha, 1 - alpha])
    return {
        'point_estimate': float(statistic(data)),
        'low': float(low),
        'high': float(high),
        'confidence': confidence,
        'n': len(data),
    }


def bootstrap_diff_ci(a: list[float], b: list[float], statistic=np.mean,
                       n_resamples: int = DEFAULT_N_RESAMPLES, confidence: float = DEFAULT_CONFIDENCE,
                       seed: int = DEFAULT_SEED) -> dict:
    """Bootstrap CI for statistic(a) - statistic(b) (independent resampling of each group)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {'point_estimate': 0.0, 'low': 0.0, 'high': 0.0, 'confidence': confidence, 'n_a': len(a), 'n_b': len(b)}

    rng = np.random.default_rng(seed)
    diffs = np.array([
        statistic(rng.choice(a, size=len(a), replace=True)) - statistic(rng.choice(b, size=len(b), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(diffs, [alpha, 1 - alpha])
    return {
        'point_estimate': float(statistic(a) - statistic(b)),
        'low': float(low),
        'high': float(high),
        'confidence': confidence,
        'n_a': len(a),
        'n_b': len(b),
    }
